"""
Base Agent Class for Audit System
=================================

Provides the foundation for all agents in the audit conformity system.
Each agent inherits from this base and implements specific functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

from .audit_types import AgentState, AgentType, WorkflowStatus

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the audit system"""
    
    def __init__(self, agent_type: AgentType, llm_config: Dict[str, Any]):
        """
        Initialize the base agent
        
        Args:
            agent_type: Type of agent (orchestrator, planner, etc.)
            llm_config: LLM configuration for this agent
        """
        self.agent_type = agent_type
        self.llm_config = llm_config
        self.agent_id = f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        
        # logger.info(f"Initialized {agent_type.value} agent with ID: {self.agent_id}")
    
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute the agent's main functionality
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after agent execution
        """
        pass
    
    def log_message(self, state: AgentState, message: str, level: str = "info") -> AgentState:
        """
        Log a message to the shared state
        
        Args:
            state: Current workflow state
            message: Message to log
            level: Log level (info, warning, error)
            
        Returns:
            Updated state with logged message
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "level": level,
            "message": message
        }
        
        state["messages"].append(log_entry)
        state["updated_at"] = timestamp
        
        # Also log to Python logger
        getattr(logger, level)(f"[{self.agent_id}] {message}")
        
        return state
    
    def log_error(self, state: AgentState, error: str) -> AgentState:
        """
        Log an error to the shared state
        
        Args:
            state: Current workflow state
            error: Error message
            
        Returns:
            Updated state with logged error
        """
        state["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "error": error
        })
        
        return self.log_message(state, f"ERROR: {error}", "error")
    
    def update_status(self, state: AgentState, status: WorkflowStatus) -> AgentState:
        """
        Update the workflow status
        
        Args:
            state: Current workflow state
            status: New workflow status
            
        Returns:
            Updated state with new status
        """
        old_status = state["status"]
        state["status"] = status
        state["updated_at"] = datetime.now().isoformat()
        
        return self.log_message(
            state, 
            f"Status changed from {old_status.value} to {status.value}"
        )
    
    async def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the configured LLM with a prompt
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
        """
        # Import here to avoid circular imports
        from app.services.llm_service import get_llm_response
        
        try:
            response = await get_llm_response(
                prompt=prompt,
                system_prompt=system_prompt,
                provider=self.llm_config.get("provider", "ollama"),
                model=self.llm_config.get("model", "llama3.2:latest"),
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 2000)            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    async def retrieve_documents(self, query: str, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the multimodal RAG system
        
        Args:
            query: Search query
            document_id: Optional specific document ID to search within
            
        Returns:
            List of relevant document chunks
        """        # Import here to avoid circular imports
        from app.services.vector_db_service import query_documents
        from app.services.embedding_service import generate_embedding
        
        try:
            # Generate embedding for the query
            query_embedding = generate_embedding(query)
            
            # Call query_documents with proper parameters
            results = query_documents(
                query_embedding=query_embedding,
                n_results=10,
                collection_name="documents"
            )
              # Filter by document_id if specified
            if document_id and results.get("documents"):
                # logger.info(f"Filtering documents by document_id: {document_id}")
                # logger.info(f"Available metadatas sample: {results.get('metadatas', [])[:2]}")
                
                filtered_docs = []
                filtered_metadatas = []
                filtered_distances = []
                
                for i, metadata in enumerate(results.get("metadatas", [])):
                    if metadata:
                        # Check multiple possible fields for document identification
                        metadata_doc_id = (metadata.get("document_id") or 
                                         metadata.get("doc_id") or 
                                         metadata.get("source") or 
                                         metadata.get("file_name") or
                                         metadata.get("filename"))
                        # logger.debug(f"Checking metadata {i}: {metadata_doc_id} vs {document_id}")
                        
                        if str(metadata_doc_id) == str(document_id):
                            if i < len(results.get("documents", [])):
                                filtered_docs.append(results["documents"][i])
                            if i < len(results.get("metadatas", [])):
                                filtered_metadatas.append(results["metadatas"][i])
                            if i < len(results.get("distances", [])):
                                filtered_distances.append(results["distances"][i])
                
                # logger.info(f"Filtered documents: {len(filtered_docs)} out of {len(results.get('documents', []))}")
                
                results = {
                    "documents": filtered_docs,
                    "metadatas": filtered_metadatas,
                    "distances": filtered_distances
                }
            else:
                # logger.info(f"No document_id filter specified, returning all {len(results.get('documents', []))} documents")
                pass
            
            # Convert to list format expected by agents
            document_list = []
            if results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    doc_data = {
                        "content": doc,
                        "metadata": results.get("metadatas", [{}])[i] if i < len(results.get("metadatas", [])) else {},
                        "distance": results.get("distances", [1.0])[i] if i < len(results.get("distances", [])) else 1.0
                    }
                    document_list.append(doc_data)
            
            return document_list
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            # Return empty list instead of raising to allow workflow to continue
            return []
    
    def validate_state(self, state: AgentState) -> bool:
        """
        Validate that the state has required fields for this agent
        
        Args:
            state: State to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        required_fields = [
            "workflow_id", "status", "created_at", "updated_at",
            "messages", "errors"
        ]
        
        for field in required_fields:
            if field not in state:
                logger.error(f"Missing required field in state: {field}")
                return False
        
        return True
