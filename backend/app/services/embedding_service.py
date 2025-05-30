import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL
import torch
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache to support multiple models
_model_cache = {}
_current_model_name = None

def get_available_models() -> Dict[str, Any]:
    """
    Get list of available embedding models with their configurations
    
    Returns:
        Dict[str, Any]: Available models configuration
    """
    return AVAILABLE_EMBEDDING_MODELS.copy()

def get_current_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model
    
    Returns:
        Dict[str, Any]: Current model information
    """
    global _current_model_name
    if _current_model_name and _current_model_name in AVAILABLE_EMBEDDING_MODELS:
        model_info = AVAILABLE_EMBEDDING_MODELS[_current_model_name].copy()
        model_info["is_loaded"] = _current_model_name in _model_cache
        return model_info
    else:
        # Default model info
        default_info = AVAILABLE_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL].copy()
        default_info["is_loaded"] = False
        return default_info

def set_embedding_model(model_name: str) -> Dict[str, Any]:
    """
    Set the active embedding model
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Dict[str, Any]: Result of model loading operation
    """
    global _current_model_name
    
    if model_name not in AVAILABLE_EMBEDDING_MODELS:
        available_models = list(AVAILABLE_EMBEDDING_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")
    
    try:
        # Load the model (this will cache it)
        model = get_embedding_model(model_name)
        _current_model_name = model_name
        
        model_info = AVAILABLE_EMBEDDING_MODELS[model_name].copy()
        model_info["is_loaded"] = True
        model_info["status"] = "success"
        model_info["message"] = f"Model '{model_name}' loaded successfully"
        
        logger.info(f"Active embedding model changed to: {model_name}")
        return model_info
        
    except Exception as e:
        error_msg = f"Failed to load model '{model_name}': {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "model_name": model_name
        }

def get_embedding_model(model_name: Optional[str] = None):
    """
    Get or initialize the embedding model
    
    Args:
        model_name: Optional model name to load. If None, uses current or default model.
    
    Returns:
        SentenceTransformer: The embedding model
    """
    global _model_cache, _current_model_name
    
    # Determine which model to use
    if model_name is None:
        model_name = _current_model_name or EMBEDDING_MODEL_NAME or DEFAULT_EMBEDDING_MODEL
    
    # Check if model is already cached
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    # Validate model name
    if model_name not in AVAILABLE_EMBEDDING_MODELS:
        available_models = list(AVAILABLE_EMBEDDING_MODELS.keys())
        logger.warning(f"Unknown model '{model_name}', falling back to default. Available: {available_models}")
        model_name = DEFAULT_EMBEDDING_MODEL
    
    # Load the model
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        
        model = SentenceTransformer(model_name, device=device)
        
        # Cache the model
        _model_cache[model_name] = model
        _current_model_name = model_name
        
        # Log model information
        dimensions = model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded successfully. Dimensions: {dimensions}")
        logger.info(f"Using device: {model.device}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {str(e)}")
        # Try to fall back to default model if this isn't already the default
        if model_name != DEFAULT_EMBEDDING_MODEL:
            logger.info(f"Falling back to default model: {DEFAULT_EMBEDDING_MODEL}")
            return get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        else:
            raise RuntimeError(f"Failed to load even the default embedding model: {str(e)}")

def clear_model_cache():
    """Clear the model cache to free up memory"""
    global _model_cache
    _model_cache.clear()
    logger.info("Embedding model cache cleared")

def get_model_cache_status() -> Dict[str, Any]:
    """
    Get information about cached models
    
    Returns:
        Dict[str, Any]: Cache status information
    """
    global _model_cache, _current_model_name
    return {
        "cached_models": list(_model_cache.keys()),
        "current_model": _current_model_name,
        "cache_size": len(_model_cache)
    }

def generate_embeddings(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of texts to embed
        model_name: Optional model name to use. If None, uses current active model.
        
    Returns:
        List[List[float]]: List of embeddings (as float lists)
    """
    model = get_embedding_model(model_name)
    logger.info(f"Generating embeddings for {len(texts)} text chunks using model: {model_name or _current_model_name}")
    
    embeddings = model.encode(texts)
    
    # Log sample of first embedding (first 5 dimensions)
    if len(embeddings) > 0:
        sample = embeddings[0][:5].tolist()
        logger.info(f"Sample embedding (first 5 dimensions): {sample}...")
    
    # Convert numpy arrays to native Python lists for JSON serialization
    return embeddings.tolist()

def generate_embedding(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Generate embedding for a single text
    
    Args:
        text: Text to embed
        model_name: Optional model name to use. If None, uses current active model.
        
    Returns:
        List[float]: Embedding as float list
    """
    logger.info(f"Generating embedding for query: {text[:50]}... using model: {model_name or _current_model_name}")
    return generate_embeddings([text], model_name)[0]