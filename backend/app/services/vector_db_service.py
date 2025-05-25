import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from app.config import CHROMA_DB_PATH, TOP_K_RESULTS

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

def get_collection(collection_name: str = "documents"):
    """
    Get or create a collection in ChromaDB
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        chromadb.Collection: The ChromaDB collection
    """
    client = get_chroma_client()
    
    # Get or create collection without specifying an embedding function
    collection = client.get_or_create_collection(name=collection_name)
    
    # Log collection info
    count = collection.count()
    logger.info(f"Using collection '{collection_name}' with {count} documents")
    
    return collection

def add_documents(
    texts: List[str], 
    embeddings: List[List[float]], 
    metadatas: List[Dict[str, Any]], 
    collection_name: str = "documents"
) -> List[str]:
    """
    Add documents to ChromaDB
    
    Args:
        texts: List of text chunks
        embeddings: List of pre-computed embeddings
        metadatas: List of metadata dictionaries
        collection_name: Name of the collection
        
    Returns:
        List[str]: List of IDs of the added documents
    """
    # Get collection
    collection = get_collection(collection_name)
    
    # Get current count to use as starting index for new IDs
    # This ensures no ID conflicts with existing documents
    current_count = collection.count()
    
    # Generate unique IDs based on current count
    ids = [f"{current_count + i}" for i in range(len(texts))]
    
    logger.info(f"Adding {len(texts)} documents to collection '{collection_name}'")
    
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

def query_documents(
    query_embedding: List[float],
    n_results: int = TOP_K_RESULTS,
    collection_name: str = "documents"
) -> Dict[str, Any]:
    """
    Query documents from ChromaDB based on a query embedding
    
    Args:
        query_embedding: Embedding of the query
        n_results: Number of results to return
        collection_name: Name of the collection
        
    Returns:
        Dict[str, Any]: Query results containing documents, metadatas, distances, and ids
    """
    # Get collection
    collection = get_collection(collection_name)
    
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

def get_all_vectorized_documents(collection_name: str = "documents") -> List[Dict[str, Any]]:
    """
    Get all documents from the vector database with their metadata
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        List[Dict[str, Any]]: List of documents with their metadata
    """
    # Get collection
    collection = get_collection(collection_name)
    
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