from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from app.models.query_models import QueryRequest, QueryResponse, LLMStatusResponse, StreamingQueryRequest, DecomposedQueryResponse, SubQueryResult
from app.services.embedding_service import generate_embedding
from app.services.vector_db_service import query_documents, query_documents_advanced, get_query_suggestions
from app.services.llm_service import get_answer_from_llm, get_llm_status, get_available_models
from app.services.metrics_service import QueryMetricsTracker
from app.services.query_decomposition_service import query_decomposer
import time
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Query the documents with a question"
)
async def query(request: QueryRequest):
    """
    Query the documents with a question and get an answer from the LLM
    """
    # Initialize metrics tracker
    metrics = QueryMetricsTracker(request.question, request.config_for_model)
    
    try:
        # Enhanced logging of the query request
        logger.info(f"Received query request - Question: '{request.question}'")
        if request.config_for_model:
            logger.info(f"Model configuration: {json.dumps(request.config_for_model)}")
        
        # Generate embedding for the question using current model
        metrics.mark_retrieval_start()
        
        # Get current embedding model info
        from app.services.embedding_service import get_current_model_info
        current_model = get_current_model_info()
        model_name = current_model.get("name", "all-MiniLM-L6-v2")
        
        query_embedding = generate_embedding(request.question)
        
        # Use advanced query processing with model-specific collection
        query_results = query_documents_advanced(
            query_embedding=query_embedding,
            query_text=request.question,
            n_results=request.max_sources or 5,
            search_strategy=request.search_strategy or "semantic",
            model_name=model_name
        )
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        relevance_scores = query_results.get("relevance_scores", [])
        search_strategy = query_results.get("search_strategy", "semantic")
        
        # Mark the end of retrieval and log metrics
        metrics.mark_retrieval_end(len(documents))
        logger.info(f"Document retrieval completed using '{search_strategy}', found {len(documents)} relevant documents")
        
        # Log relevance metrics
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            max_relevance = max(relevance_scores)
            logger.info(f"Relevance scores - Avg: {avg_relevance:.3f}, Max: {max_relevance:.3f}")
        
        if not documents:
            logger.warning(f"No relevant documents found for query: '{request.question}'")
            metrics.complete()
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question. Please try a different question or upload more documents.",
                sources=[],
                query_time_ms=metrics.complete()["total_time_ms"],
                num_sources=0
            )
            
        # Get answer from LLM using the specified model configuration
        metrics.mark_llm_start()
        answer, model_info = get_answer_from_llm(request.question, documents, request.config_for_model)
        
        # Mark the end of LLM processing and log metrics
        metrics.mark_llm_end(model_info, len(answer) if answer else 0)
        
        # Log the complete response for better tracking
        logger.info(f"Query completed - Model: {model_info}")
        
        # Get final metrics and complete the tracking
        final_metrics = metrics.complete()
        
        # Return response with enhanced metrics
        return QueryResponse(
            answer=answer,
            sources=metadatas,
            query_time_ms=final_metrics["total_time_ms"],
            retrieval_time_ms=final_metrics["retrieval_time_ms"],
            llm_time_ms=final_metrics["llm_time_ms"],
            num_sources=len(documents),
            model=model_info,
            search_strategy=search_strategy,
            # Calculate average and top relevance (scores are now 0-100)
            average_relevance=round(sum(relevance_scores) / len(relevance_scores), 1) if relevance_scores else None,
            top_relevance=round(max(relevance_scores), 1) if relevance_scores else None
        )
        
    except Exception as e:
        # Enhanced error logging
        logger.error(f"Error processing query: '{request.question}'", exc_info=True)
        logger.error(f"Error details: {str(e)}")
        
        # Track the error in metrics
        metrics.mark_error(str(e))
        
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post(
    "/stream",
    summary="Stream a response to a query"
)
async def stream_query(request: QueryRequest):
    """
    Query the documents and stream the response from the LLM as it's generated
    """
    # Initialize metrics tracker
    metrics = QueryMetricsTracker(request.question, request.config_for_model)
    
    raise HTTPException(status_code=501, detail="Streaming is no longer supported. Please use the standard query endpoint.")


@router.get(
    "/status",
    response_model=LLMStatusResponse,
    summary="Get the status of the LLM service"
)
async def get_status():
    """
    Get the current status of the LLM service
    """
    try:
        status = get_llm_status()
        return LLMStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting LLM status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting LLM status: {str(e)}")


@router.get(
    "/stream",
    summary="Stream a response to a query (GET method for EventSource)"
)
async def stream_query_get(
    question: str, 
    model_config: str = Query(None, description="JSON string with model configuration")
):
    """
    Query the documents and stream the response from the LLM as it's generated.
    This endpoint uses GET to work with EventSource.
    """
    # Parse model_config if provided
    config = {}
    if model_config:
        try:
            config = json.loads(model_config)
            logger.info(f"Parsed model config: {config}")
        except json.JSONDecodeError:
            logger.error(f"Invalid model config JSON: {model_config}")
    
    # We reuse our POST endpoint by creating a request object
    request = QueryRequest(question=question, model_config=config)
    return await stream_query(request)


@router.get(
    "/models",
    summary="Get available LLM models"
)
async def get_models():
    """
    Get a list of available LLM models
    """
    try:
        models = get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")


@router.post(
    "/decomposed",
    response_model=DecomposedQueryResponse,
    summary="Query with intelligent decomposition for complex questions"
)
async def query_with_decomposition(request: QueryRequest):
    """
    Enhanced query endpoint that automatically decomposes complex questions
    into sub-queries for better accuracy and comprehensive answers.
    """
    start_time = time.time()
    decomposition_time_ms = None
    synthesis_time_ms = None
    
    try:
        logger.info(f"Received decomposed query request - Question: '{request.question}'")
        logger.info(f"Decomposition enabled: {request.use_decomposition}")
        
        if request.config_for_model:
            logger.info(f"Model configuration: {json.dumps(request.config_for_model)}")
        
        # Step 1: Query Decomposition (if enabled)
        if request.use_decomposition:
            decomp_start = time.time()
            is_complex, sub_queries = await query_decomposer.decompose_query(
                request.question, 
                request.config_for_model
            )
            decomposition_time_ms = int((time.time() - decomp_start) * 1000)
            
            logger.info(f"Query decomposition completed in {decomposition_time_ms}ms")
        else:
            is_complex = False
            sub_queries = [request.question]
            logger.info("Query decomposition disabled, treating as simple query")
        
        # Step 2: Process each sub-query
        sub_results = []
        for sub_query in sub_queries:
            sub_result = await query_decomposer.process_sub_query(
                sub_query=sub_query,
                model_config=request.config_for_model,
                max_sources=request.max_sources or 5,
                search_strategy=request.search_strategy or "semantic"
            )
            
            # Convert to SubQueryResult model
            sub_query_result = SubQueryResult(
                sub_query=sub_result["sub_query"],
                answer=sub_result["answer"],
                sources=sub_result["sources"],
                num_sources=sub_result["num_sources"],
                relevance_scores=sub_result["relevance_scores"],
                processing_time_ms=sub_result["processing_time_ms"],
                model_info=sub_result.get("model_info")
            )
            sub_results.append(sub_query_result)
        
        # Step 3: Synthesize final answer (if decomposed)
        if is_complex and len(sub_results) > 1:
            synthesis_start = time.time()
            sub_results_dict = [
                {
                    "sub_query": sr.sub_query,
                    "answer": sr.answer,
                    "sources": sr.sources,
                    "num_sources": sr.num_sources,
                    "relevance_scores": sr.relevance_scores
                }
                for sr in sub_results
            ]
            
            final_answer, synthesis_model = await query_decomposer.synthesize_answers(
                request.question,
                sub_results_dict,
                request.config_for_model
            )
            synthesis_time_ms = int((time.time() - synthesis_start) * 1000)
            
            logger.info(f"Answer synthesis completed in {synthesis_time_ms}ms using {synthesis_model}")
        else:
            # For simple queries, use the single answer
            final_answer = sub_results[0].answer if sub_results else "No answer could be generated."
            synthesis_model = sub_results[0].model_info if sub_results else "unknown"
        
        # Calculate aggregate metrics
        total_query_time_ms = int((time.time() - start_time) * 1000)
        
        # Collect all unique sources
        all_sources = []
        all_relevance_scores = []
        for sub_result in sub_results:
            all_sources.extend(sub_result.sources)
            all_relevance_scores.extend(sub_result.relevance_scores)
        
        # Remove duplicate sources based on document ID
        unique_sources = []
        seen_docs = set()
        for source in all_sources:
            doc_id = source.get('document_id', str(source))
            if doc_id not in seen_docs:
                unique_sources.append(source)
                seen_docs.add(doc_id)
        
        # Calculate average relevance
        avg_relevance = round(sum(all_relevance_scores) / len(all_relevance_scores), 1) if all_relevance_scores else None
        top_relevance = round(max(all_relevance_scores), 1) if all_relevance_scores else None
        
        logger.info(f"Decomposed query completed in {total_query_time_ms}ms")
        logger.info(f"Generated {len(sub_results)} sub-queries, final answer length: {len(final_answer)}")
        
        return DecomposedQueryResponse(
            original_query=request.question,
            is_decomposed=is_complex,
            sub_queries=[sr.sub_query for sr in sub_results],
            sub_results=sub_results,
            final_answer=final_answer,
            total_query_time_ms=total_query_time_ms,
            decomposition_time_ms=decomposition_time_ms,
            synthesis_time_ms=synthesis_time_ms,
            model=synthesis_model if is_complex else (sub_results[0].model_info if sub_results else "unknown"),
            search_strategy=request.search_strategy or "semantic",
            total_sources=len(unique_sources),
            average_relevance=avg_relevance,
            top_relevance=top_relevance
        )
        
    except Exception as e:
        logger.error(f"Error in decomposed query processing: {e}")
        
        # Fallback to regular query processing
        try:
            logger.info("Falling back to regular query processing")
            regular_response = await query(request)
            
            # Convert to decomposed format
            return DecomposedQueryResponse(
                original_query=request.question,
                is_decomposed=False,
                sub_queries=[request.question],
                sub_results=[
                    SubQueryResult(
                        sub_query=request.question,
                        answer=regular_response.answer,
                        sources=regular_response.sources,
                        num_sources=regular_response.num_sources,
                        relevance_scores=[],
                        processing_time_ms=regular_response.query_time_ms,
                        model_info=regular_response.model
                    )
                ],
                final_answer=regular_response.answer,
                total_query_time_ms=regular_response.query_time_ms,
                decomposition_time_ms=None,
                synthesis_time_ms=None,
                model=regular_response.model,
                search_strategy=regular_response.search_strategy,
                total_sources=regular_response.num_sources,
                average_relevance=regular_response.average_relevance,
                top_relevance=regular_response.top_relevance
            )
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {fallback_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Query processing failed: {str(e)}. Fallback also failed: {str(fallback_error)}"
            )