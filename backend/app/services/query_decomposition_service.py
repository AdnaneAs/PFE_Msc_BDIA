import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from app.services.llm_service import get_answer_from_llm
from app.services.embedding_service import generate_embedding
from app.services.vector_db_service import query_documents_advanced

# Configure logging
logger = logging.getLogger(__name__)

class QueryDecomposer:
    """
    Service for decomposing complex queries into simpler sub-queries
    and processing them independently before synthesizing the final answer.
    """
    
    def __init__(self):
        self.decomposition_cache = {}  # Cache for decomposed queries
        
    def _create_decomposition_prompt(self, query: str) -> str:
        """
        Create a prompt to analyze and potentially decompose a query
        
        Args:
            query: The original user query
            
        Returns:
            str: Formatted prompt for query analysis
        """
        prompt = f"""You are a query analysis expert. Analyze the following query and determine if it needs to be broken down into smaller, more specific questions.

INSTRUCTIONS:
1. If the query is SIMPLE (single concept, straightforward question), respond with:
   CLASSIFICATION: SIMPLE
   QUERY: [original query]

2. If the query is COMPLEX (multiple concepts, compound questions, requires multiple steps), respond with:
   CLASSIFICATION: COMPLEX
   SUB_QUERIES:
   1. [first specific sub-question]
   2. [second specific sub-question]
   3. [additional sub-questions as needed]

GUIDELINES:
- Break down queries that ask about multiple topics
- Separate questions that require different types of information
- Split compound questions connected by "and", "or", "also"
- Keep sub-queries focused and specific
- Maintain the original intent and context

QUERY TO ANALYZE: "{query}"

ANALYSIS:"""
        
        return prompt
    
    def _create_synthesis_prompt(self, original_query: str, sub_queries: List[str], sub_answers: List[str]) -> str:
        """
        Create a prompt to synthesize sub-answers into a comprehensive final answer
        
        Args:
            original_query: The original user query
            sub_queries: List of sub-queries
            sub_answers: List of corresponding answers
            
        Returns:
            str: Formatted prompt for answer synthesis
        """
        qa_pairs = []
        for i, (sub_q, sub_a) in enumerate(zip(sub_queries, sub_answers), 1):
            qa_pairs.append(f"Q{i}: {sub_q}\nA{i}: {sub_a}")
        
        qa_text = "\n\n".join(qa_pairs)
        
        prompt = f"""You are tasked with synthesizing multiple sub-answers into a comprehensive response to the original question.

ORIGINAL QUESTION: "{original_query}"

SUB-QUESTIONS AND ANSWERS:
{qa_text}

INSTRUCTIONS:
1. Create a coherent, comprehensive answer that addresses the original question
2. Integrate information from all relevant sub-answers
3. Maintain logical flow and avoid redundancy
4. If any sub-answers indicate lack of information, mention this appropriately
5. Include relevant citations and sources when available
6. Ensure the final answer directly responds to the original question

SYNTHESIZED ANSWER:"""
        
        return prompt
    
    def _parse_decomposition_response(self, response: str) -> Tuple[bool, List[str]]:
        """
        Parse the LLM response for query decomposition
        
        Args:
            response: Raw LLM response
            
        Returns:
            tuple: (is_complex, list_of_queries)
        """
        try:
            lines = response.strip().split('\n')
            
            # Look for classification
            classification = None
            for line in lines:
                if 'CLASSIFICATION:' in line.upper():
                    classification = line.split(':', 1)[1].strip().upper()
                    break
            
            if classification == 'SIMPLE':
                # Find the query line
                for line in lines:
                    if 'QUERY:' in line.upper():
                        query = line.split(':', 1)[1].strip()
                        return False, [query]
                # Fallback: return original query
                return False, []
                
            elif classification == 'COMPLEX':
                # Extract sub-queries
                sub_queries = []
                in_sub_queries = False
                
                for line in lines:
                    if 'SUB_QUERIES:' in line.upper():
                        in_sub_queries = True
                        continue
                    
                    if in_sub_queries and line.strip():
                        # Remove numbering and clean up
                        clean_line = line.strip()
                        if clean_line[0].isdigit() and '.' in clean_line:
                            clean_line = clean_line.split('.', 1)[1].strip()
                        if clean_line:
                            sub_queries.append(clean_line)
                
                return True, sub_queries
            
            # Fallback: treat as simple
            return False, []
            
        except Exception as e:
            logger.error(f"Error parsing decomposition response: {e}")
            return False, []
    
    async def decompose_query(self, query: str, model_config: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Analyze and potentially decompose a query
        
        Args:
            query: The user query to analyze
            model_config: LLM configuration for decomposition
            
        Returns:
            tuple: (is_complex, list_of_sub_queries or [original_query])
        """
        # Check cache first
        cache_key = f"decomp_{hash(query.strip().lower())}"
        if cache_key in self.decomposition_cache:
            logger.info(f"Using cached decomposition for query: {query[:50]}...")
            return self.decomposition_cache[cache_key]
        
        logger.info(f"Analyzing query complexity: {query[:100]}...")
        
        try:
            # Create decomposition prompt
            decomp_prompt = self._create_decomposition_prompt(query)
            
            # Use a lightweight model for decomposition if available
            decomp_config = model_config.copy() if model_config else {}
            if not model_config or not model_config.get('model'):
                # Default to a fast model for decomposition
                decomp_config = {
                    'provider': 'ollama',
                    'model': 'llama3.2:latest'
                }
            
            # Get decomposition analysis
            response, model_info = get_answer_from_llm(
                question="",  # We're using a custom prompt
                context_documents=[],  # No context needed for decomposition
                model_config=decomp_config,
                custom_prompt=decomp_prompt
            )
            
            logger.info(f"Decomposition analysis completed using {model_info}")
            logger.debug(f"Decomposition response: {response[:200]}...")
            
            # Parse the response
            is_complex, queries = self._parse_decomposition_response(response)
            
            # If parsing failed or no queries, fall back to original
            if not queries:
                queries = [query]
                is_complex = False
            
            # Cache the result
            result = (is_complex, queries)
            self.decomposition_cache[cache_key] = result
            
            logger.info(f"Query classified as {'COMPLEX' if is_complex else 'SIMPLE'}")
            if is_complex:
                logger.info(f"Decomposed into {len(queries)} sub-queries: {[q[:50] for q in queries]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query decomposition: {e}")
            # Fallback to treating as simple query
            return False, [query]
    
    async def process_sub_query(self, sub_query: str, model_config: Dict[str, Any] = None, 
                              max_sources: int = 5, search_strategy: str = "semantic") -> Dict[str, Any]:
        """
        Process a single sub-query through the RAG pipeline
        
        Args:
            sub_query: The sub-query to process
            model_config: LLM configuration
            max_sources: Maximum number of sources to retrieve
            search_strategy: Search strategy to use
            
        Returns:
            dict: Sub-query results including answer, sources, and metrics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing sub-query: {sub_query[:100]}...")
              # Generate embedding for the sub-query using current model
            query_embedding = generate_embedding(sub_query)
            
            # Get current embedding model info
            from app.services.embedding_service import get_current_model_info
            current_model = get_current_model_info()
            model_name = current_model.get("name", "all-MiniLM-L6-v2")
            
            # Query documents using model-specific collection
            query_results = query_documents_advanced(
                query_embedding=query_embedding,
                query_text=sub_query,
                n_results=max_sources,
                search_strategy=search_strategy,
                model_name=model_name
            )
            
            documents = query_results["documents"]
            metadatas = query_results["metadatas"]
            relevance_scores = query_results.get("relevance_scores", [])
            
            if not documents:
                logger.warning(f"No relevant documents found for sub-query: {sub_query[:50]}...")
                return {
                    "sub_query": sub_query,
                    "answer": "No relevant information found for this specific question.",
                    "sources": [],
                    "num_sources": 0,
                    "relevance_scores": [],
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # Get answer from LLM
            answer, model_info = get_answer_from_llm(sub_query, documents, model_config)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Sub-query processed in {processing_time}ms with {len(documents)} sources")
            
            return {
                "sub_query": sub_query,
                "answer": answer,
                "sources": metadatas,
                "num_sources": len(documents),
                "relevance_scores": relevance_scores,
                "processing_time_ms": processing_time,
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"Error processing sub-query '{sub_query[:50]}...': {e}")
            return {
                "sub_query": sub_query,
                "answer": f"Error processing this question: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "relevance_scores": [],
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def synthesize_answers(self, original_query: str, sub_results: List[Dict[str, Any]], 
                               model_config: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Synthesize multiple sub-answers into a comprehensive final answer
        
        Args:
            original_query: The original user query
            sub_results: List of sub-query results
            model_config: LLM configuration for synthesis
            
        Returns:
            tuple: (synthesized_answer, model_info)
        """
        try:
            logger.info(f"Synthesizing {len(sub_results)} sub-answers for query: {original_query[:50]}...")
            
            # Extract sub-queries and answers
            sub_queries = [result["sub_query"] for result in sub_results]
            sub_answers = [result["answer"] for result in sub_results]
            
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(original_query, sub_queries, sub_answers)
            
            # Use the same model config as the original query
            synthesis_config = model_config if model_config else {
                'provider': 'ollama',
                'model': 'llama3.2:latest'
            }
            
            # Get synthesized answer
            synthesized_answer, model_info = get_answer_from_llm(
                question="",  # Using custom prompt
                context_documents=[],  # Context is in the prompt
                model_config=synthesis_config,
                custom_prompt=synthesis_prompt
            )
            
            logger.info(f"Answer synthesis completed using {model_info}")
            
            return synthesized_answer, model_info
            
        except Exception as e:
            logger.error(f"Error synthesizing answers: {e}")
            # Fallback: concatenate sub-answers
            fallback_answer = f"Based on the analysis of your question:\n\n"
            for i, result in enumerate(sub_results, 1):
                fallback_answer += f"{i}. {result['answer']}\n\n"
            
            return fallback_answer.strip(), "synthesis_fallback"


# Global instance
query_decomposer = QueryDecomposer()
