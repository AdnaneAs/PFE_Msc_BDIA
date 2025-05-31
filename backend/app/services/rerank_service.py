"""
BGE Reranker Service
==================

This service implements BGE (BAAI General Embedding) reranking to enhance RAG pipeline
by providing better ranking of retrieved documents based on query-document relevance.

BGE Reranker improves retrieval quality by:
1. Taking initial vector search results (top 20-50)
2. Using cross-encoder models to score query-document pairs
3. Reranking based on actual relevance scores
4. Returning top-k most relevant documents

This significantly improves retrieval quality over vector similarity alone.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

class BGEReranker:
    """
    BGE Cross-Encoder Reranker for improving document retrieval quality
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize BGE reranker
        
        Args:
            model_name: Name of the BGE reranker model to use
                      Options: 'BAAI/bge-reranker-base', 'BAAI/bge-reranker-large'
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configurations
        self.model_configs = {
            "BAAI/bge-reranker-base": {
                "display_name": "BGE Reranker Base",
                "description": "Fast and efficient reranking model",
                "model_size": "~400MB",
                "speed": "Fast"
            },
            "BAAI/bge-reranker-large": {
                "display_name": "BGE Reranker Large", 
                "description": "Higher quality reranking with more parameters",
                "model_size": "~1.2GB",
                "speed": "Slower"
            },
            "BAAI/bge-reranker-v2-m3": {
                "display_name": "BGE Reranker v2 M3",
                "description": "Latest multilingual reranking model",
                "model_size": "~600MB", 
                "speed": "Medium"
            }
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the BGE reranker model"""
        try:
            logger.info(f"Loading BGE reranker model: {self.model_name}")
            start_time = time.time()
            
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"BGE reranker loaded in {load_time:.2f}s on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BGE reranker {self.model_name}: {e}")
            # Fallback to base model
            try:
                fallback_model = "BAAI/bge-reranker-base"
                logger.info(f"Trying fallback model: {fallback_model}")
                self.model = CrossEncoder(fallback_model, device=self.device)
                self.model_name = fallback_model
                logger.info("Fallback BGE reranker loaded successfully")
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback reranker: {fallback_e}")
                self.model = None
    
    def is_available(self) -> bool:
        """Check if reranker is available"""
        return self.model is not None
    
    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        top_k: int = 5,
        return_scores: bool = True
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Rerank documents based on query relevance using BGE reranker
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            metadatas: Optional metadata for each document
            top_k: Number of top documents to return
            return_scores: Whether to return relevance scores
            
        Returns:
            Tuple of (reranked_documents, reranked_metadatas, relevance_scores)
        """
        if not self.is_available():
            logger.warning("BGE reranker not available, returning original order")
            top_k = min(top_k, len(documents))
            return (
                documents[:top_k],
                metadatas[:top_k] if metadatas else [{}] * top_k,
                [0.5] * top_k  # Default neutral scores
            )
        
        if not documents:
            return [], [], []
        
        try:
            start_time = time.time()
            
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = [[query, doc] for doc in documents]
            
            # Get relevance scores from BGE reranker
            logger.info(f"Reranking {len(documents)} documents with BGE reranker")
            
            # Predict relevance scores
            scores = self.model.predict(query_doc_pairs)
            
            # Convert to list if numpy array
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            # Create tuples of (document, metadata, score) for sorting
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            doc_score_tuples = list(zip(documents, metadatas, scores))
            
            # Sort by relevance score (descending)
            doc_score_tuples.sort(key=lambda x: x[2], reverse=True)
            
            # Extract top_k results
            top_k = min(top_k, len(doc_score_tuples))
            top_results = doc_score_tuples[:top_k]
            
            reranked_docs = [item[0] for item in top_results]
            reranked_metadatas = [item[1] for item in top_results]
            relevance_scores = [float(item[2]) for item in top_results]
            
            rerank_time = time.time() - start_time
            
            # Log reranking results
            logger.info(f"Reranking completed in {rerank_time:.3f}s")
            logger.info(f"Top relevance score: {max(relevance_scores):.4f}")
            logger.info(f"Average relevance: {np.mean(relevance_scores):.4f}")
            
            # Add reranking metadata
            for i, metadata in enumerate(reranked_metadatas):
                metadata['rerank_score'] = relevance_scores[i]
                metadata['rerank_model'] = self.model_name
                metadata['rerank_position'] = i + 1
            
            return reranked_docs, reranked_metadatas, relevance_scores
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original order on error
            top_k = min(top_k, len(documents))
            return (
                documents[:top_k],
                metadatas[:top_k] if metadatas else [{}] * top_k,
                [0.5] * top_k
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current reranker model"""
        config = self.model_configs.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "display_name": config.get("display_name", self.model_name),
            "description": config.get("description", "BGE reranking model"),
            "model_size": config.get("model_size", "Unknown"),
            "speed": config.get("speed", "Unknown"),
            "device": self.device,            "available": self.is_available()
        }
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Rerank documents using the BGE cross-encoder model
        
        Args:
            query: The query string
            documents: List of document strings to rerank
            top_k: Number of top documents to return (if None, returns all)
            
        Returns:
            List of tuples (document, relevance_score) sorted by relevance
        """
        if top_k is None:
            top_k = len(documents)
            
        reranked_docs, _, relevance_scores = self.rerank_documents(query, documents, None, top_k)
        return list(zip(reranked_docs, relevance_scores))


def is_reranker_available() -> bool:
    """
    Check if BGE reranker is available and can be used
    
    Returns:
        bool: True if reranker is available, False otherwise
    """
    try:
        import sentence_transformers
        return True
    except ImportError:
        logger.warning("sentence-transformers not available for BGE reranking")
        return False


# Global reranker instance
_global_reranker = None

def get_reranker(model_name: str = "BAAI/bge-reranker-base") -> BGEReranker:
    """
    Get or create a global BGE reranker instance
    
    Args:
        model_name: Name of the BGE reranker model
        
    Returns:
        BGEReranker: The reranker instance
    """
    global _global_reranker
    
    if _global_reranker is None or _global_reranker.model_name != model_name:
        logger.info(f"Initializing BGE reranker: {model_name}")
        _global_reranker = BGEReranker(model_name)
    
    return _global_reranker

def rerank_search_results(
    query: str,
    search_results: Dict[str, Any],
    top_k: int = 5,
    reranker_model: str = "BAAI/bge-reranker-base"
) -> Dict[str, Any]:
    """
    Rerank search results using BGE reranker
    
    Args:
        query: Original search query
        search_results: Results from vector search (ChromaDB format)
        top_k: Number of top results to return after reranking
        reranker_model: BGE reranker model to use
        
    Returns:
        Dict: Reranked search results in ChromaDB format
    """
    # Get reranker
    reranker = get_reranker(reranker_model)
    
    if not reranker.is_available():
        logger.warning("Reranker not available, returning original results")
        return search_results
    
    # Extract documents and metadata from search results
    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    distances = search_results.get("distances", [[]])[0]
    ids = search_results.get("ids", [[]])[0]
    
    if not documents:
        return search_results
    
    # Rerank documents
    reranked_docs, reranked_metadatas, relevance_scores = reranker.rerank_documents(
        query=query,
        documents=documents,
        metadatas=metadatas,
        top_k=top_k,
        return_scores=True
    )
    
    # Reconstruct the results format
    # Note: We lose the original distances and ids after reranking
    # We'll create new ones based on reranked order
    reranked_ids = []
    reranked_distances = []
    
    for i, metadata in enumerate(reranked_metadatas):
        # Try to find original ID
        original_id = "reranked_" + str(i)
        for j, orig_meta in enumerate(metadatas):
            if orig_meta.get('filename') == metadata.get('filename') and \
               orig_meta.get('chunk_index') == metadata.get('chunk_index'):
                original_id = ids[j] if j < len(ids) else original_id
                break
        
        reranked_ids.append(original_id)
        
        # Convert relevance score to distance-like metric (lower is better)
        # BGE reranker scores are typically in range [-10, 10], we normalize
        distance = max(0.0, (1.0 - (relevance_scores[i] + 10) / 20))
        reranked_distances.append(distance)
    
    return {
        "documents": [reranked_docs],
        "metadatas": [reranked_metadatas],
        "distances": [reranked_distances],
        "ids": [reranked_ids],
        "relevance_scores": relevance_scores,
        "search_strategy": "semantic_with_reranking",
        "reranker_model": reranker_model
    }

def get_available_rerankers() -> List[Dict[str, Any]]:
    """
    Get list of available BGE reranker models
    
    Returns:
        List of dictionaries with reranker information
    """
    reranker = BGEReranker()  # Create temporary instance to get configs
    
    models = []
    for model_name, config in reranker.model_configs.items():
        models.append({
            "model_name": model_name,
            "display_name": config["display_name"],
            "description": config["description"],
            "model_size": config["model_size"],
            "speed": config["speed"]
        })
    
    return models
