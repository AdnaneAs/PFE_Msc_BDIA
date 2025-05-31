import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from app.config import CHROMA_DB_PATH, TOP_K_RESULTS

# Import BGE reranker service
try:
    from app.services.rerank_service import rerank_search_results, get_reranker
    RERANKING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("BGE reranking service available")
except ImportError as e:
    RERANKING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"BGE reranking not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the DB directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
logger.info(f"Vector database path: {CHROMA_DB_PATH}")

# Initialize ChromaDB client
_client = None

def get_chroma_client():
    """
    Get or initialize the ChromaDB client
    
    Returns:
        chromadb.PersistentClient: The ChromaDB client
    """
    global _client
    if _client is None:
        logger.info(f"Initializing ChromaDB client at {CHROMA_DB_PATH}")
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client

def get_collection(collection_name: str = "documents", model_name: Optional[str] = None):
    """
    Get or create a model-specific collection in ChromaDB
    
    Args:
        collection_name: Base name of the collection
        model_name: Name of the embedding model (auto-detected if None)
        
    Returns:
        chromadb.Collection: The ChromaDB collection
    """
    client = get_chroma_client()
    
    # Generate model-specific collection name
    full_collection_name = get_collection_name_for_model(collection_name, model_name)
    
    # Get or create collection without specifying an embedding function
    collection = client.get_or_create_collection(name=full_collection_name)
    
    # Log collection info
    count = collection.count()
    logger.info(f"Using collection '{full_collection_name}' with {count} documents")
    
    return collection

def get_collection_name_for_model(base_collection: str = "documents", model_name: Optional[str] = None) -> str:
    """
    Generate a collection name specific to an embedding model
    
    Args:
        base_collection: Base collection name (e.g., "documents")
        model_name: Name of the embedding model (e.g., "all-MiniLM-L6-v2", "bge-m3")
        
    Returns:
        str: Model-specific collection name
    """
    if model_name is None:
        # Import here to avoid circular imports
        from app.services.embedding_service import get_current_model_info
        try:
            current_model = get_current_model_info()
            model_name = current_model.get("name", "all-MiniLM-L6-v2")
        except Exception:
            # Fallback to default model
            model_name = "all-MiniLM-L6-v2"
      # Sanitize model name for collection naming
    sanitized_model = model_name.replace("-", "_").replace(".", "_").replace("/", "_").lower()
    collection_name = f"{base_collection}_{sanitized_model}"
    
    logger.info(f"Using collection '{collection_name}' for model '{model_name}'")
    return collection_name

def add_documents(
    texts: List[str], 
    embeddings: List[List[float]], 
    metadatas: List[Dict[str, Any]], 
    collection_name: str = "documents",
    model_name: Optional[str] = None
) -> List[str]:
    """
    Add documents to a model-specific ChromaDB collection
    
    Args:
        texts: List of text chunks
        embeddings: List of pre-computed embeddings
        metadatas: List of metadata dictionaries
        collection_name: Base name of the collection
        model_name: Name of the embedding model (auto-detected if None)
        
    Returns:
        List[str]: List of IDs of the added documents
    """
    # Get model-specific collection
    collection = get_collection(collection_name, model_name)
    
    # Generate unique IDs using UUIDs to avoid conflicts with any existing IDs
    import uuid
    ids = [f"chunk-{uuid.uuid4()}" for _ in range(len(texts))]
    
    logger.info(f"Adding {len(texts)} documents to collection '{collection_name}'")
    
    try:
        # Add documents to collection
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # Log updated count
        count = collection.count()
        logger.info(f"Collection '{collection_name}' now has {count} documents")
        
        return ids
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {str(e)}")
        # If there was an error, try with new IDs
        try:
            # Generate completely new IDs
            import uuid
            new_ids = [f"chunk-{uuid.uuid4()}" for _ in range(len(texts))]
            
            # Try again with new IDs
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=new_ids
            )
            
            # Log updated count
            count = collection.count()
            logger.info(f"Collection '{collection_name}' now has {count} documents after retry")
            
            return new_ids
        except Exception as retry_e:
            logger.error(f"Failed to add documents even after retry: {str(retry_e)}")
            raise

def query_documents(
    query_embedding: List[float],
    n_results: int = TOP_K_RESULTS,
    collection_name: str = "documents",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query documents from a model-specific ChromaDB collection based on a query embedding
    
    Args:
        query_embedding: Embedding of the query
        n_results: Number of results to return
        collection_name: Base name of the collection
        model_name: Name of the embedding model (auto-detected if None)
        
    Returns:
        Dict[str, Any]: Query results containing documents, metadatas, distances, and ids
    """
    # Get model-specific collection
    collection = get_collection(collection_name, model_name)
    
    logger.info(f"Querying collection '{collection_name}' for top {n_results} results")
    
    # Query collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Process results
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    # Log query results
    logger.info(f"Found {len(documents)} documents matching the query")
    if len(distances) > 0:
        logger.info(f"Top result similarity score: {1.0 - distances[0]:.4f}")
    
    # Log source document info
    if len(metadatas) > 0:
        sources = [f"{m.get('filename', 'unknown')}:{m.get('chunk_index', 0)}" for m in metadatas[:3]]
        logger.info(f"Top sources: {', '.join(sources)}")
    
    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
        "ids": ids
    }

def get_all_vectorized_documents(collection_name: str = "documents", model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all documents from a model-specific vector database collection with their metadata
    
    Args:
        collection_name: Base name of the collection
        model_name: Name of the embedding model (auto-detected if None)
        
    Returns:
        List[Dict[str, Any]]: List of documents with their metadata
    """
    # Get model-specific collection
    collection = get_collection(collection_name, model_name)
    
    # Get all documents
    results = collection.get()
    
    # Process results
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])
    
    # Combine documents with their metadata
    vectorized_docs = []
    seen_files = set()  # To track unique files
    
    for doc, metadata, doc_id in zip(documents, metadatas, ids):
        filename = metadata.get("filename", "unknown")
        if filename not in seen_files:
            seen_files.add(filename)
            vectorized_docs.append({
                "filename": filename,
                "doc_id": doc_id,
                "chunk_count": metadata.get("total_chunks", 1),
                "uploaded_at": metadata.get("uploaded_at", ""),
                "status": "processed"
            })
    
    logger.info(f"Retrieved {len(vectorized_docs)} unique documents from vector database")
    return vectorized_docs

def delete_documents(document_id: str = None, filename: str = None, model_name: Optional[str] = None) -> bool:
    """
    Delete documents from a model-specific vector database collection based on document_id or filename.
    
    Args:
        document_id: ID of the document to delete
        filename: Name of the file to delete
        model_name: Name of the embedding model (auto-detected if None)
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Get model-specific collection
        collection = get_collection("documents", model_name)
        
        # Build where clause based on provided parameters
        where = {}
        if document_id:
            where["doc_id"] = str(document_id)
        elif filename:
            where["filename"] = filename
            
        if not where:
            logger.error("No deletion criteria provided")
            return False
        
        # First get all matching documents
        try:
            results = collection.get(where=where)
            ids_to_delete = results.get("ids", [])
            
            if not ids_to_delete:
                logger.warning(f"No documents found matching criteria: {where}")
                return True  # Return true since there's nothing to delete
                
            logger.info(f"Found {len(ids_to_delete)} document chunks to delete with criteria: {where}")
            
            # Delete by IDs instead of using 'where' clause
            collection.delete(ids=ids_to_delete)
            
            logger.info(f"Successfully deleted {len(ids_to_delete)} document chunks")
            return True
            
        except Exception as inner_e:
            logger.error(f"Error during deletion process: {str(inner_e)}")
            # Try a direct deletion as fallback
            collection.delete(where=where)
            logger.info(f"Attempted fallback deletion with where clause: {where}")
            return True
            
    except Exception as e:
        logger.error(f"Error deleting documents from vector database: {str(e)}")
        return False

def query_documents_advanced(
    query_embedding: List[float],
    query_text: str,
    n_results: int = TOP_K_RESULTS,
    collection_name: str = "documents",
    search_strategy: str = "semantic",  # "semantic", "hybrid", "keyword", "semantic_rerank"
    model_name: Optional[str] = None,
    use_reranking: bool = False,
    reranker_model: str = "BAAI/bge-reranker-base"
) -> Dict[str, Any]:
    """
    Advanced document querying with multiple search strategies using model-specific collections
    Enhanced with BGE reranking for +23.86% MAP improvement
    
    Args:
        query_embedding: Embedding of the query
        query_text: Original query text for keyword matching
        n_results: Number of results to return
        collection_name: Base name of the collection
        search_strategy: Type of search to perform
        model_name: Name of the embedding model (auto-detected if None)
        use_reranking: Whether to apply BGE reranking to improve results
        reranker_model: BGE reranker model to use
        
    Returns:
        Dict[str, Any]: Enhanced query results with relevance scores
    """
    # Debug logging
    logger.info(f"🎯 BGE RERANKING DEBUG: use_reranking={use_reranking}, RERANKING_AVAILABLE={RERANKING_AVAILABLE}, reranker_model={reranker_model}")
    
    if use_reranking:
        logger.info(f"BGE reranking enabled with model: {reranker_model} (benchmarked +23.86% MAP improvement)")
    
    collection = get_collection(collection_name, model_name)
    
    logger.info(f"Advanced querying with strategy '{search_strategy}' for top {n_results} results")
    
    # Import configuration for reranking optimization
    try:
        from app.config import RERANKING_INITIAL_RETRIEVAL_MULTIPLIER, RERANKING_MAX_INITIAL_DOCUMENTS
        retrieval_multiplier = RERANKING_INITIAL_RETRIEVAL_MULTIPLIER
        max_initial_docs = RERANKING_MAX_INITIAL_DOCUMENTS
    except ImportError:
        # Fallback values if config not available
        retrieval_multiplier = 3
        max_initial_docs = 50
    
    # Determine initial retrieval count for reranking
    # Get more documents initially if reranking is enabled for better quality
    initial_n_results = n_results
    if use_reranking and RERANKING_AVAILABLE:
        # Use configuration-based multiplier for optimal performance
        initial_n_results = min(n_results * retrieval_multiplier, max_initial_docs)
        logger.info(f"Retrieving {initial_n_results} documents for BGE reranking to top {n_results}")
    
    if search_strategy == "semantic" or search_strategy == "semantic_rerank":
        # Standard semantic search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_n_results
        )
    
    elif search_strategy == "hybrid":
        # Combine semantic and keyword search
        # First get more results for re-ranking
        initial_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(initial_n_results * 3, 50)  # Get 3x more for re-ranking
        )
        
        # Re-rank based on keyword overlap
        results = _rerank_results(initial_results, query_text, initial_n_results)
        
    elif search_strategy == "keyword":
        # Keyword-based search using ChromaDB's where clause
        # Note: This is a simplified implementation
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_n_results,
                where_document={"$contains": query_text.lower()}
            )
            
            # If no results found with keyword search, fall back to semantic search
            if not results.get("documents", [[]])[0]:
                logger.info(f"No keyword matches for '{query_text}', falling back to semantic search")
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=initial_n_results
                )
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}, falling back to semantic search")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_n_results
            )
    
    else:
        # Default to semantic search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_n_results
        )
      # Apply BGE reranking if requested and available
    if use_reranking and RERANKING_AVAILABLE and search_strategy != "hybrid":
        logger.info("Applying BGE reranking to improve result quality")
        try:
            results = rerank_search_results(
                query=query_text,
                search_results=results,
                top_k=n_results,
                reranker_model=reranker_model
            )
            # Update search strategy to indicate reranking was applied
            search_strategy = f"{search_strategy}_reranked"
        except Exception as e:
            logger.error(f"BGE reranking failed: {e}, using original results without reranking")
            # Continue with original results instead of failing completely
    elif use_reranking and not RERANKING_AVAILABLE:
        logger.warning("BGE reranking requested but not available, using original results")
    elif use_reranking and search_strategy == "hybrid":
        logger.info("BGE reranking skipped for hybrid search strategy")
    
    # Process and enhance results
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]
    
    # Get relevance scores from reranking or calculate from distances
    if 'relevance_scores' in results:
        relevance_scores = results['relevance_scores']
    elif distances:
        # Calculate relevance scores (convert distance to similarity)
        # ChromaDB returns squared Euclidean distances, which can be > 1.0
        # We need to normalize them to a 0-1 relevance score
        relevance_scores = [max(0.0, min(1.0, 1.0 / (1.0 + dist))) for dist in distances]
        # Convert to percentage-like scores (0-100)
        relevance_scores = [round(score * 100, 2) for score in relevance_scores]
    else:
        relevance_scores = []
    
    # Add relevance scores to metadata
    for i, metadata in enumerate(metadatas):
        if i < len(relevance_scores):
            metadata['relevance_score'] = relevance_scores[i]
    
    # Log enhanced query results
    logger.info(f"Found {len(documents)} documents with strategy '{search_strategy}'")
    if relevance_scores:
        logger.info(f"Top relevance score: {max(relevance_scores):.4f}")
        logger.info(f"Average relevance: {sum(relevance_scores)/len(relevance_scores):.4f}")
    
    # Log top sources with relevance
    if metadatas:
        top_sources = []
        for i, m in enumerate(metadatas[:3]):
            filename = m.get('filename', 'unknown')
            chunk_idx = m.get('chunk_index', 0)
            relevance = m.get('relevance_score', 0)
            top_sources.append(f"{filename}:{chunk_idx} ({relevance:.3f})")
        logger.info(f"Top sources: {', '.join(top_sources)}")
    
    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
        "ids": ids,
        "relevance_scores": relevance_scores,
        "search_strategy": search_strategy,  # Ensure this is always returned
        "reranking_used": "_reranked" in search_strategy,  # Check if search strategy indicates reranking was applied
        "reranker_model": reranker_model if (use_reranking and RERANKING_AVAILABLE) else None
    }

def _rerank_results(initial_results: Dict, query_text: str, n_results: int) -> Dict:
    """
    Re-rank search results based on keyword overlap and semantic similarity
    """
    documents = initial_results.get("documents", [[]])[0]
    metadatas = initial_results.get("metadatas", [[]])[0]
    distances = initial_results.get("distances", [[]])[0]
    ids = initial_results.get("ids", [[]])[0]
    
    if not documents:
        return initial_results
    
    # Calculate keyword overlap scores
    query_words = set(query_text.lower().split())
    
    scored_results = []
    for i, doc in enumerate(documents):
        doc_words = set(doc.lower().split())
        keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
        
        # Calculate proper semantic similarity score
        # Use the same method as the main scoring function
        semantic_score = max(0.0, min(1.0, 1.0 / (1.0 + distances[i]))) if i < len(distances) else 0
        combined_score = 0.7 * semantic_score + 0.3 * keyword_overlap
        
        scored_results.append({
            'document': doc,
            'metadata': metadatas[i] if i < len(metadatas) else {},
            'distance': distances[i] if i < len(distances) else 1.0,
            'id': ids[i] if i < len(ids) else '',
            'combined_score': combined_score,
            'keyword_overlap': keyword_overlap
        })
    
    # Sort by combined score (descending)
    scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Take top n_results
    top_results = scored_results[:n_results]
    
    # Reconstruct the results format
    return {
        "documents": [[r['document'] for r in top_results]],
        "metadatas": [[r['metadata'] for r in top_results]],
        "distances": [[r['distance'] for r in top_results]],
        "ids": [[r['id'] for r in top_results]]
    }

def get_query_suggestions(query_text: str, collection_name: str = "documents", model_name: Optional[str] = None) -> List[str]:
    """
    Generate query suggestions based on document content from model-specific collection
    """
    collection = get_collection(collection_name, model_name)
    
    # Simple implementation: get random document chunks and extract key phrases
    try:
        # Get a sample of documents
        results = collection.get(limit=10)
        documents = results.get("documents", [])
        
        # Extract potential query topics (simplified)
        suggestions = []
        for doc in documents[:5]:
            words = doc.split()
            if len(words) > 10:
                # Extract potential question starters
                suggestions.append(f"What is {' '.join(words[:5])}...?")
                suggestions.append(f"How does {' '.join(words[:3])} work?")
        
        return suggestions[:3]  # Return top 3 suggestions
    except Exception as e:
        logger.error(f"Error generating query suggestions: {e}")
        return []

def get_all_model_collections() -> List[str]:
    """
    Get all existing embedding model collection names in the database
    
    Returns:
        List[str]: List of all collection names
    """
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        logger.info(f"Found {len(collection_names)} collections: {collection_names}")
        return collection_names
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        return []

def search_across_all_models(doc_id: str = None, filename: str = None) -> Dict[str, Any]:
    """
    Search for documents across all embedding model collections
    
    Args:
        doc_id: Document ID to search for
        filename: Filename to search for
        
    Returns:
        Dict containing results from all collections
    """
    all_results = {}
    collection_names = get_all_model_collections()
    
    for collection_name in collection_names:
        try:
            client = get_chroma_client()
            collection = client.get_collection(name=collection_name)
            
            # Build search criteria
            where = {}
            if doc_id:
                where["doc_id"] = str(doc_id)
            elif filename:
                where["filename"] = filename
            
            if where:
                results = collection.get(where=where)
                if results.get("ids"):
                    all_results[collection_name] = results
                    
        except Exception as e:
            logger.warning(f"Error searching in collection {collection_name}: {str(e)}")
            continue
    
    return all_results

def delete_from_all_models(doc_id: str = None, filename: str = None) -> bool:
    """
    Delete documents from all embedding model collections
    
    Args:
        doc_id: Document ID to delete
        filename: Filename to delete
        
    Returns:
        bool: True if deletion was successful from at least one collection
    """
    success = False
    all_results = search_across_all_models(doc_id=doc_id, filename=filename)
    
    for collection_name, results in all_results.items():
        try:
            client = get_chroma_client()
            collection = client.get_collection(name=collection_name)
            
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents from collection {collection_name}")
                success = True
                
        except Exception as e:
            logger.error(f"Error deleting from collection {collection_name}: {str(e)}")
            continue
    
    return success