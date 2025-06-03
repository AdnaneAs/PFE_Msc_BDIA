from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from app.services.embedding_service import (
    get_available_models,
    get_current_model_info,
    set_embedding_model
)
from app.services.settings_service import (
    load_settings, 
    update_setting, 
    get_setting,
    get_available_models_by_provider,
    update_model_with_validation,
    get_ollama_models
)
from app.config import AVAILABLE_EMBEDDING_MODELS

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["configuration"])

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
        current_llm_model = user_settings.get("llm_model", "llama3.2:latest")        # Get available models by provider (uses caching)
        available_models_by_provider = get_available_models_by_provider(force_refresh=False)
        
        available_llm_providers = {}
        for provider, models in available_models_by_provider.items():
            provider_info = {
                "name": provider,
                "models": models
            }
            
            if provider == "ollama":
                provider_info.update({
                    "display_name": "Ollama (Local)",
                    "description": "Local LLM inference with Ollama",
                    "status": "available" if models else "unavailable"
                })
            elif provider == "openai":
                provider_info.update({
                    "display_name": "OpenAI",
                    "description": "OpenAI GPT models (API key required)",
                    "status": "available" if user_settings.get("api_key_openai") else "unavailable"
                })
            elif provider == "gemini":
                provider_info.update({
                    "display_name": "Google Gemini",
                    "description": "Google Gemini models (API key required)",
                    "status": "available" if user_settings.get("api_key_gemini") else "unavailable"
                })
            elif provider == "huggingface":
                provider_info.update({
                    "display_name": "Hugging Face",
                    "description": "Hugging Face transformers (API key required)",
                    "status": "available" if user_settings.get("api_key_huggingface") else "unavailable"
                })
            
            available_llm_providers[provider] = provider_info
        
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
        available_providers = get_available_models_by_provider()
        
        if provider not in available_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Must be one of: {list(available_providers.keys())}"
            )
        
        # Get current model and check compatibility
        current_model = get_setting("llm_model", "llama3.2:latest")
        
        # If switching providers, reset to first available model for that provider
        if provider != get_setting("llm_provider"):
            available_models = available_providers[provider]
            if available_models:
                new_model = available_models[0]
                success = update_model_with_validation(provider, new_model)
                if not success:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to switch to provider '{provider}' with model '{new_model}'"
                    )
                return {
                    "status": "success",
                    "message": f"LLM provider changed to {provider} with model {new_model}",
                    "provider": provider,
                    "model": new_model
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"No models available for provider '{provider}'"
                )
        
        # Same provider, just update the setting
        update_setting("llm_provider", provider)
        
        return {
            "status": "success",
            "message": f"LLM provider confirmed as {provider}",
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
    Update the LLM model setting with validation
    
    Args:
        request: Dict containing 'model' and optionally 'provider' keys
        
    Returns:
        Dict containing update result
    """
    try:
        model = request.get("model")
        provider = request.get("provider") or get_setting("llm_provider", "ollama")
        
        if not model:
            raise HTTPException(status_code=400, detail="model field is required")
        
        # Use validation function
        success = update_model_with_validation(provider, model)
        
        if not success:
            # Get available models for error message
            available_models = get_available_models_by_provider()
            provider_models = available_models.get(provider, [])
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{model}' for provider '{provider}'. Available models: {provider_models}"
            )
        
        return {
            "status": "success",
            "message": f"LLM model changed to {model} (provider: {provider})",
            "model": model,
            "provider": provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LLM model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update LLM model: {str(e)}"
        )

@router.post(
    "/api-keys/{provider}",
    summary="Store API key for a provider",
    response_description="Result of API key storage"
)
async def store_api_key(provider: str, request: Dict[str, str]) -> Dict[str, Any]:
    """
    Store API key for a specific provider (openai, gemini, huggingface)
    
    Args:
        provider: Provider name
        request: Dict containing 'api_key' key
        
    Returns:
        Dict containing storage result
    """
    try:
        valid_providers = ["openai", "gemini", "huggingface"]
        
        if provider not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Must be one of: {valid_providers}"
            )
        
        api_key = request.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail="api_key field is required")
        
        # Store API key in settings
        setting_key = f"api_key_{provider}"
        update_setting(setting_key, api_key.strip())
        
        return {
            "status": "success",
            "message": f"API key stored for {provider}",
            "provider": provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing API key for {provider}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store API key for {provider}: {str(e)}"
        )

@router.get(
    "/api-keys/status",
    summary="Get API keys status",
    response_description="Status of stored API keys"
)
async def get_api_keys_status() -> Dict[str, Any]:
    """
    Get status of stored API keys (without revealing the actual keys)
    
    Returns:
        Dict containing API keys status
    """
    try:
        settings = load_settings()
        
        status = {
            "openai": bool(settings.get("api_key_openai")),
            "gemini": bool(settings.get("api_key_gemini")),
            "huggingface": bool(settings.get("api_key_huggingface"))
        }
        
        return {
            "status": "success",
            "api_keys": status
        }
        
    except Exception as e:
        logger.error(f"Error getting API keys status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get API keys status: {str(e)}"
        )

@router.delete(
    "/api-keys/{provider}",
    summary="Clear API key for a provider",
    response_description="Result of API key clearing"
)
async def clear_api_key(provider: str) -> Dict[str, Any]:
    """
    Clear API key for a specific provider
    
    Args:
        provider: Provider name
        
    Returns:
        Dict containing clear result
    """
    try:
        valid_providers = ["openai", "gemini", "huggingface"]
        
        if provider not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Must be one of: {valid_providers}"
            )
        
        # Clear API key from settings
        setting_key = f"api_key_{provider}"
        update_setting(setting_key, None)
        
        return {
            "status": "success",
            "message": f"API key cleared for {provider}",
            "provider": provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing API key for {provider}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear API key for {provider}: {str(e)}"
        )

@router.post(
    "/reranking/toggle",
    summary="Toggle BGE reranking with persistent storage",
    response_description="Result of reranking toggle"
)
async def toggle_reranking_persistent(request: Dict[str, bool]) -> Dict[str, Any]:
    """
    Toggle BGE reranking on/off with persistent storage
    
    Args:
        request: Dict containing 'enabled' key
        
    Returns:
        Dict containing toggle result
    """
    try:
        enabled = request.get("enabled")
        
        if enabled is None:
            raise HTTPException(status_code=400, detail="enabled field is required")
        
        # Save to persistent settings
        update_setting("reranking_enabled", enabled)
        
        # Also update runtime config for immediate effect
        import app.config as config
        config.ENABLE_RERANKING_BY_DEFAULT = enabled
        
        return {
            "status": "success",
            "message": f"BGE reranking {'enabled' if enabled else 'disabled'}",
            "enabled": enabled,
            "persistent": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling reranking: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle reranking: {str(e)}"
        )

@router.get(
    "/reranking/status",
    summary="Get current reranking status"
)
async def get_reranking_status():
    """Get current reranking configuration status"""
    try:
        settings = load_settings()
        from app.config import ENABLE_RERANKING_BY_DEFAULT, DEFAULT_RERANKER_MODEL
        
        return {
            "reranking_enabled": settings.get("reranking_enabled", ENABLE_RERANKING_BY_DEFAULT),
            "default_model": DEFAULT_RERANKER_MODEL,
            "persistent": "reranking_enabled" in settings
        }
    except Exception as e:
        logger.error(f"Error getting reranking status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/llm/models/{provider}",
    summary="Get available models for a specific provider",
    response_description="List of available models for the provider"
)
async def get_models_for_provider(provider: str) -> Dict[str, Any]:
    """
    Get available models for a specific LLM provider
    
    Args:
        provider: Provider name (ollama, openai, gemini, huggingface)
        
    Returns:
        Dict containing available models for the provider
    """
    try:
        available_models = get_available_models_by_provider()
        
        if provider not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{provider}' not found. Available providers: {list(available_models.keys())}"
            )
        
        models = available_models[provider]
          # For Ollama, also include real-time availability status (cached)
        if provider == "ollama":
            fresh_models = get_ollama_models(force_refresh=False)
            return {
                "status": "success",
                "provider": provider,
                "models": models,
                "available_locally": fresh_models,
                "total_models": len(models),
                "ollama_running": len(fresh_models) > 0,
                "cached": True
            }
        
        return {
            "status": "success",
            "provider": provider,
            "models": models,
            "total_models": len(models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get models for provider: {str(e)}"
        )

@router.get(
    "/llm/models",
    summary="Get all available models organized by provider",
    response_description="All available models organized by provider"
)
async def get_all_available_models() -> Dict[str, Any]:
    """
    Get all available models organized by provider
    
    Returns:
        Dict containing all models organized by provider
    """
    try:
        available_models = get_available_models_by_provider()
          # Add real-time status for Ollama (cached)
        if "ollama" in available_models:
            ollama_models = get_ollama_models(force_refresh=False)
            available_models["ollama_status"] = {
                "running": len(ollama_models) > 0,
                "available_locally": ollama_models,
                "cached": True
            }
        
        return {
            "status": "success",
            "providers": available_models,
            "total_providers": len([k for k in available_models.keys() if not k.endswith("_status")])
        }
        
    except Exception as e:
        logger.error(f"Error getting all available models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.post(
    "/cache/clear",
    summary="Clear models cache",
    response_description="Result of cache clearing"
)
async def clear_models_cache_endpoint() -> Dict[str, Any]:
    """
    Clear models cache to force refresh
    
    Returns:
        Dict containing cache clearing result
    """
    try:
        from app.services.settings_service import clear_models_cache
        clear_models_cache()
        
        return {
            "status": "success",
            "message": "Models cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing models cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear models cache: {str(e)}"
        )

@router.get(
    "/cache/status",
    summary="Get cache status",
    response_description="Current cache status information"
)
async def get_cache_status() -> Dict[str, Any]:
    """
    Get information about the current cache status
    
    Returns:
        Dict containing cache status information
    """
    try:
        from app.services.settings_service import _ollama_models_cache, _ollama_cache_timestamp, _models_by_provider_cache, _models_cache_timestamp
        from datetime import datetime
        
        now = datetime.now()
        
        ollama_cache_age = None
        models_cache_age = None
        
        if _ollama_cache_timestamp:
            ollama_cache_age = (now - _ollama_cache_timestamp).seconds
            
        if _models_cache_timestamp:
            models_cache_age = (now - _models_cache_timestamp).seconds
        
        return {
            "status": "success",
            "ollama_cache": {
                "has_data": _ollama_models_cache is not None,
                "age_seconds": ollama_cache_age,
                "model_count": len(_ollama_models_cache) if _ollama_models_cache else 0
            },
            "models_cache": {
                "has_data": _models_by_provider_cache is not None,
                "age_seconds": models_cache_age,
                "providers": list(_models_by_provider_cache.keys()) if _models_by_provider_cache else []
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache status: {str(e)}"
        )

@router.post(
    "/llm/models/refresh",
    summary="Refresh available models cache",
    response_description="Refreshed models list"
)
async def refresh_available_models() -> Dict[str, Any]:
    """
    Force refresh the available models cache
    
    Returns:
        Dict containing refreshed models data
    """
    try:
        # Force refresh the cache
        available_models = get_available_models_by_provider(force_refresh=True)
        
        return {
            "status": "success",
            "message": "Models cache refreshed successfully",
            "providers": available_models,
            "total_providers": len(available_models),
            "refreshed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing available models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh available models: {str(e)}"
        )

# VLM Configuration Endpoints

@router.get(
    "/vlm",
    summary="Get VLM configuration",
    response_description="VLM provider, model, and available options"
)
async def get_vlm_configuration() -> Dict[str, Any]:
    """
    Get current VLM configuration and available models
    
    Returns:
        Dict containing VLM configuration
    """
    try:
        from app.services.settings_service import get_available_vlm_models
        
        # Load user settings
        user_settings = load_settings()
        
        # Get current VLM configuration
        current_vlm_provider = user_settings.get("vlm_provider", "ollama")
        current_vlm_model = user_settings.get("vlm_model", "llava:latest")
        
        # Get available VLM models by provider
        available_vlm_models = get_available_vlm_models()
        
        # Build provider info
        available_vlm_providers = {}
        for provider, models in available_vlm_models.items():
            provider_info = {
                "name": provider,
                "models": models
            }
            
            if provider == "ollama":
                provider_info.update({
                    "display_name": "Ollama (Local)",
                    "description": "Local VLM inference with models like LLaVA",
                    "status": "available" if models else "unavailable"
                })
            elif provider == "openai":
                provider_info.update({
                    "display_name": "OpenAI GPT-4 Vision",
                    "description": "OpenAI vision models (API key required)",
                    "status": "available" if user_settings.get("api_key_openai") else "unavailable"
                })
            elif provider == "gemini":
                provider_info.update({
                    "display_name": "Google Gemini Pro Vision",
                    "description": "Google Gemini vision models (API key required)",
                    "status": "available" if user_settings.get("api_key_gemini") else "unavailable"
                })
            elif provider == "huggingface":
                provider_info.update({
                    "display_name": "Hugging Face Vision",
                    "description": "HuggingFace vision models (BLIP, GIT, etc.)",
                    "status": "available"
                })
            
            available_vlm_providers[provider] = provider_info
        
        return {
            "vlm": {
                "current_provider": current_vlm_provider,
                "current_model": current_vlm_model,
                "available_providers": available_vlm_providers
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting VLM configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get VLM configuration: {str(e)}"
        )

@router.post(
    "/vlm/provider",
    summary="Update VLM provider",
    response_description="Confirmation of VLM provider update"
)
async def update_vlm_provider_endpoint(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Update the VLM provider setting
    
    Args:
        request: Dict with 'provider' key
        
    Returns:
        Confirmation message
    """
    try:
        provider = request.get("provider")
        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        
        # Validate provider
        valid_providers = ["ollama", "openai", "gemini", "huggingface"]
        if provider not in valid_providers:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid provider. Must be one of: {valid_providers}"
            )
        
        # Update setting
        update_setting("vlm_provider", provider)
        
        logger.info(f"VLM provider updated to: {provider}")
        
        return {
            "message": f"VLM provider updated to {provider}",
            "provider": provider,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating VLM provider: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update VLM provider: {str(e)}"
        )

@router.post(
    "/vlm/model",
    summary="Update VLM model",
    response_description="Confirmation of VLM model update"
)
async def update_vlm_model_endpoint(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Update the VLM model setting
    
    Args:
        request: Dict with 'model' key
        
    Returns:
        Confirmation message
    """
    try:
        model = request.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="Model is required")
        
        # Update setting
        update_setting("vlm_model", model)
        
        logger.info(f"VLM model updated to: {model}")
        
        return {
            "message": f"VLM model updated to {model}",
            "model": model,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating VLM model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update VLM model: {str(e)}"
        )
