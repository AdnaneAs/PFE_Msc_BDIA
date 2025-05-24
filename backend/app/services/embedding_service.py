from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME

# Load the model once at module initialization
_model = None

def get_embedding_model():
    """
    Get or initialize the embedding model
    
    Returns:
        SentenceTransformer: The embedding model
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List[List[float]]: List of embeddings (as float lists)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts)
    
    # Convert numpy arrays to native Python lists for JSON serialization
    return embeddings.tolist()

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text
    
    Args:
        text: Text to embed
        
    Returns:
        List[float]: Embedding as float list
    """
    return generate_embeddings([text])[0] 