from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from app.services.embedding_service import (
    get_available_models,
    get_current_model_info,
    set_embedding_model,
    clear_model_cache,
    get_model_cache_status
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["embedding-models"])

@router.get(
    "/embedding/available",
    summary="Get available embedding models",
    response_description="List of available embedding models with their configurations"
)
async def get_available_embedding_models() -> Dict[str, Any]:
    """
    Get list of available embedding models with their configurations
    
    Returns:
        Dict containing available models and their specifications
    """
    try:
        models = get_available_models()
        current_model = get_current_model_info()
        
        return {
            "status": "success",
            "available_models": models,
            "current_model": current_model,
            "total_models": len(models)
        }
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.get(
    "/embedding/current",
    summary="Get current embedding model information",
    response_description="Information about the currently active embedding model"
)
async def get_current_embedding_model() -> Dict[str, Any]:
    """
    Get information about the currently active embedding model
    
    Returns:
        Dict containing current model information
    """
    try:
        current_model = get_current_model_info()
        cache_status = get_model_cache_status()
        
        return {
            "status": "success",
            "current_model": current_model,
            "cache_status": cache_status
        }
    except Exception as e:
        logger.error(f"Error getting current model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current model info: {str(e)}"
        )

@router.post(
    "/embedding/set/{model_name}",
    summary="Set active embedding model",
    response_description="Result of setting the active embedding model"
)
async def set_active_embedding_model(model_name: str) -> Dict[str, Any]:
    """
    Set the active embedding model
    
    Args:
        model_name: Name of the model to set as active
        
    Returns:
        Dict containing the result of the operation
    """
    try:
        logger.info(f"Setting active embedding model to: {model_name}")
        result = set_embedding_model(model_name)
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "Failed to set embedding model")
            )
        
        return {
            "status": "success",
            "message": f"Successfully set active model to '{model_name}'",
            "model_info": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting embedding model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set embedding model: {str(e)}"
        )

@router.post(
    "/embedding/cache/clear",
    summary="Clear embedding model cache",
    response_description="Result of clearing the model cache"
)
async def clear_embedding_model_cache() -> Dict[str, Any]:
    """
    Clear the embedding model cache to free up memory
    
    Returns:
        Dict containing the result of the cache clearing operation
    """
    try:
        logger.info("Clearing embedding model cache")
        clear_model_cache()
        
        return {
            "status": "success",
            "message": "Embedding model cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing model cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear model cache: {str(e)}"
        )

@router.get(
    "/embedding/cache/status",
    summary="Get embedding model cache status",
    response_description="Information about the current cache status"
)
async def get_embedding_cache_status() -> Dict[str, Any]:
    """
    Get information about the embedding model cache
    
    Returns:
        Dict containing cache status information
    """
    try:
        cache_status = get_model_cache_status()
        
        return {
            "status": "success",
            "cache_status": cache_status
        }
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache status: {str(e)}"
        )

@router.get(
    "/embedding/collections/status",
    summary="Get embedding collections status",
    response_description="Status of all embedding model collections in the database"
)
async def get_collections_status() -> Dict[str, Any]:
    """
    Get status and statistics for all embedding model collections
    
    Returns:
        Dict containing collection statistics and document counts
    """
    try:
        from app.services.vector_db_service import get_all_model_collections, get_chroma_client
        
        collections = get_all_model_collections()
        collection_stats = {}
        total_documents = 0
        
        client = get_chroma_client()
        
        for collection_name in collections:
            try:
                collection = client.get_collection(name=collection_name)
                count = collection.count()
                collection_stats[collection_name] = {
                    "document_count": count,
                    "status": "active"
                }
                total_documents += count
            except Exception as e:
                collection_stats[collection_name] = {
                    "document_count": 0,
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "status": "success",
            "total_collections": len(collections),
            "total_documents": total_documents,
            "collections": collection_stats,
            "current_model": get_current_model_info()
        }
    except Exception as e:
        logger.error(f"Error getting collections status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collections status: {str(e)}"
        )
