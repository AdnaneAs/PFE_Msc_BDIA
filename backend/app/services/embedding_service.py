import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Check if CUDA is available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device: {device}")
        
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        
        # Log model information and device
        logger.info(f"Embedding model loaded successfully. Dimensions: {_model.get_sentence_embedding_dimension()}")
        logger.info(f"Using device: {_model.device}")
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
    logger.info(f"Generating embeddings for {len(texts)} text chunks")
    
    embeddings = model.encode(texts)
    
    # Log sample of first embedding (first 5 dimensions)
    if len(embeddings) > 0:
        sample = embeddings[0][:5].tolist()
        logger.info(f"Sample embedding (first 5 dimensions): {sample}...")
    
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
    logger.info(f"Generating embedding for query: {text[:50]}...")
    return generate_embeddings([text])[0] 