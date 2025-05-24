import os
from typing import List, Dict, Any, Optional
import chromadb
from app.config import CHROMA_DB_PATH, TOP_K_RESULTS

# Ensure the DB directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

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
    
    # Generate IDs if not already in metadata
    ids = [str(i) for i in range(len(texts))]
    
    # Add documents to collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
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
    
    # Query collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return {
        "documents": results.get("documents", [[]])[0],
        "metadatas": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0],
        "ids": results.get("ids", [[]])[0]
    } 