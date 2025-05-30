from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging
from app.services.embedding_service import (
    get_available_models,
    get_current_model_info,
    set_embedding_model
)
from app.services.settings_service import load_settings, update_setting, get_setting
from app.config import AVAILABLE_EMBEDDING_MODELS

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["configuration"])

@router.get(
    "/",
    summary="Get complete system configuration",
    response_description="All system configuration sections organized by category"
)
async def get_system_configuration() -> Dict[str, Any]:
    """
    Get organized system configuration including model selection and search settings
    
    Returns:
        Dict containing all configuration sections
    """
    try:        # Load user settings
        user_settings = load_settings()
        
        # Get current embedding model info
        current_embedding_model = get_current_model_info()
        available_embedding_models = get_available_models()
        
        # Get LLM configuration from settings
        current_llm_provider = user_settings.get("llm_provider", "ollama")
        current_llm_model = user_settings.get("llm_model", "llama3.2:latest")
        
        available_llm_providers = {
            "ollama": {
                "name": "ollama",
                "display_name": "Ollama (Local)",
                "description": "Local LLM inference with Ollama",
                "status": "available",  # TODO: Check actual Ollama status
                "models": [
                    "llama3.2:latest",
                    "llama3.2:1b", 
                    "llama3.2:3b",
                    "qwen2.5:latest",
                    "qwen2.5:0.5b",
                    "qwen2.5:1.5b",
                    "qwen2.5:3b",
                    "qwen2.5:7b"
                ]
            },
            "openai": {
                "name": "openai",
                "display_name": "OpenAI",
                "description": "OpenAI GPT models (API key required)",
                "status": "unavailable",  # TODO: Check API key
                "models": ["gpt-4", "gpt-3.5-turbo"]
            },
            "gemini": {
                "name": "gemini", 
                "display_name": "Google Gemini",
                "description": "Google Gemini models (API key required)",
                "status": "unavailable",  # TODO: Check API key
                "models": ["gemini-pro", "gemini-pro-vision"]
            },
            "huggingface": {
                "name": "huggingface",
                "display_name": "Hugging Face",
                "description": "Hugging Face transformers (API key required)",
                "status": "unavailable",  # TODO: Check API key
                "models": ["custom"]
            }
        }
        
        configuration = {            "model_selection": {
                "llm": {
                    "current_provider": current_llm_provider,
                    "current_model": current_llm_model,
                    "available_providers": available_llm_providers
                },
                "embedding": {
                    "current_model": current_embedding_model,
                    "available_models": available_embedding_models
                }
            },
            "search_configuration": {
                "query_decomposition": {
                    "enabled": user_settings.get("query_decomposition_enabled", False),
                    "description": "Automatically breaks down complex questions into sub-queries for more comprehensive answers. Best for multi-part questions or when you need detailed analysis."
                },
                "search_strategy": {
                    "current": user_settings.get("search_strategy", "hybrid"),
                    "options": {
                        "hybrid": {
                            "name": "hybrid",
                            "display_name": "🔍 Hybrid Search (Recommended)",
                            "description": "Combines semantic understanding with keyword matching for best results"
                        },
                        "semantic": {
                            "name": "semantic", 
                            "display_name": "🧠 Semantic Search",
                            "description": "Uses AI to understand meaning and context"
                        },
                        "keyword": {
                            "name": "keyword",
                            "display_name": "🔎 Keyword Search", 
                            "description": "Traditional text matching search"
                        }
                    }
                },                "max_sources": {
                    "current": user_settings.get("max_sources", 5),
                    "options": [3, 5, 10, 15, 20],
                    "description": "Number of relevant documents to use for generating the answer"
                }
            },
            "status": "success"
        }
        
        return configuration
        
    except Exception as e:
        logger.error(f"Error getting system configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system configuration: {str(e)}"
        )

@router.post(
    "/embedding/model",
    summary="Update embedding model selection",
    response_description="Result of embedding model change"
)
async def update_embedding_model(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Update the active embedding model
    
    Args:
        request: Dict containing 'model_name' key
        
    Returns:
        Dict containing update result
    """
    try:
        model_name = request.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")
        
        # Validate model exists
        available_models = get_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_name}' not available. Available models: {list(available_models.keys())}"
            )
          # Switch to the new model
        result = set_embedding_model(model_name)
        
        # Save to settings
        update_setting("embedding_model", model_name)
        
        return {
            "status": "success",
            "message": f"Embedding model changed to {model_name}",
            "model_info": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating embedding model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update embedding model: {str(e)}"
        )

@router.post(
    "/search/strategy",
    summary="Update search strategy",
    response_description="Result of search strategy change"
)
async def update_search_strategy(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Update the search strategy setting
    
    Args:
        request: Dict containing 'strategy' key
        
    Returns:
        Dict containing update result
    """
    try:
        strategy = request.get("strategy")
        valid_strategies = ["hybrid", "semantic", "keyword"]
        
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Must be one of: {valid_strategies}"
            )
        
        # Save to settings
        update_setting("search_strategy", strategy)
        
        return {
            "status": "success",
            "message": f"Search strategy changed to {strategy}",
            "strategy": strategy
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating search strategy: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update search strategy: {str(e)}"
        )

@router.post(
    "/search/max-sources",
    summary="Update maximum sources setting",
    response_description="Result of max sources change"
)
async def update_max_sources(request: Dict[str, int]) -> Dict[str, Any]:
    """
    Update the maximum sources setting
    
    Args:
        request: Dict containing 'max_sources' key
        
    Returns:
        Dict containing update result
    """
    try:
        max_sources = request.get("max_sources")
        valid_options = [3, 5, 10, 15, 20]
        
        if max_sources not in valid_options:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid max_sources. Must be one of: {valid_options}"
            )
        
        # Save to settings
        update_setting("max_sources", max_sources)
        
        return {
            "status": "success", 
            "message": f"Max sources changed to {max_sources}",
            "max_sources": max_sources
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating max sources: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update max sources: {str(e)}"
        )

@router.post(
    "/search/query-decomposition", 
    summary="Toggle query decomposition setting",
    response_description="Result of query decomposition toggle"
)
async def toggle_query_decomposition(request: Dict[str, bool]) -> Dict[str, Any]:
    """
    Toggle the query decomposition feature
    
    Args:
        request: Dict containing 'enabled' key
        
    Returns:
        Dict containing update result
    """
    try:
        enabled = request.get("enabled")
        
        if enabled is None:
            raise HTTPException(status_code=400, detail="enabled field is required")
        
        # Save to settings
        update_setting("query_decomposition_enabled", enabled)
        
        return {
            "status": "success",
            "message": f"Query decomposition {'enabled' if enabled else 'disabled'}",
            "enabled": enabled
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling query decomposition: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle query decomposition: {str(e)}"
        )

@router.post(
    "/llm/provider",
    summary="Update LLM provider",
    response_description="Result of LLM provider change"
)
async def update_llm_provider(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Update the LLM provider setting
    
    Args:
        request: Dict containing 'provider' key
        
    Returns:
        Dict containing update result
    """
    try:
        provider = request.get("provider")
        valid_providers = ["ollama", "openai", "gemini", "huggingface"]
        
        if provider not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Must be one of: {valid_providers}"
            )
        
        # Save to settings
        update_setting("llm_provider", provider)
        
        return {
            "status": "success",
            "message": f"LLM provider changed to {provider}",
            "provider": provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LLM provider: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update LLM provider: {str(e)}"
        )

@router.post(
    "/llm/model",
    summary="Update LLM model",
    response_description="Result of LLM model change"
)
async def update_llm_model(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Update the LLM model setting
    
    Args:
        request: Dict containing 'model' key
        
    Returns:
        Dict containing update result
    """
    try:
        model = request.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="model field is required")
        
        # Save to settings
        update_setting("llm_model", model)
        
        return {
            "status": "success",
            "message": f"LLM model changed to {model}",
            "model": model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LLM model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update LLM model: {str(e)}"
        )
