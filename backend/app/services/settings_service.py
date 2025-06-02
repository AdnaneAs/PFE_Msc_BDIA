import json
import os
import logging
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Settings file location
SETTINGS_FILE = Path(__file__).parent.parent / "data" / "user_settings.json"

# Provider-specific model patterns for validation
PROVIDER_MODEL_PATTERNS = {
    "ollama": {
        "patterns": [r".*:.*", r"^[a-zA-Z0-9_-]+$"],  # Models with tags or simple names
        "examples": ["llama3.2:latest", "mistral:latest", "codellama:7b"]
    },
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        "patterns": [r"^gpt-.*"]
    },    "gemini": {
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.0-pro"],
        "patterns": [r"^gemini-.*"]
    },
    "huggingface": {
        "patterns": [r".*/.*"],  # Format: organization/model-name
        "examples": ["microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b-chat-hf"]
    }
}

# Available embedding models
AVAILABLE_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1",
    "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-MiniLM-L12-v2"
]

# Default settings
DEFAULT_SETTINGS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_provider": "ollama",
    "llm_model": "llama3.2:latest",
    "search_strategy": "hybrid",
    "max_sources": 5,
    "query_decomposition_enabled": False,
    "reranking_enabled": True,
    "api_key_openai": None,
    "api_key_gemini": None,
    "api_key_huggingface": None,
    "last_updated": None
}

def load_settings() -> Dict[str, Any]:
    """
    Load user settings from file, create with defaults if not exists
    
    Returns:
        Dict containing user settings
    """
    try:
        # Ensure the data directory exists
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                
            # Merge with defaults to handle new settings
            merged_settings = {**DEFAULT_SETTINGS, **settings}
            
            # Validate current model configuration
            provider = merged_settings.get("llm_provider", "ollama")
            model = merged_settings.get("llm_model", "llama3.2:latest")
            
            if not validate_model_for_provider(provider, model):
                logger.warning(f"Invalid model configuration detected, resetting to defaults")
                merged_settings["llm_provider"] = DEFAULT_SETTINGS["llm_provider"]
                merged_settings["llm_model"] = DEFAULT_SETTINGS["llm_model"]
                save_settings(merged_settings)
            
            logger.info(f"Loaded user settings from {SETTINGS_FILE}")
            return merged_settings
        else:
            # Create default settings file
            save_settings(DEFAULT_SETTINGS)
            logger.info(f"Created default settings file at {SETTINGS_FILE}")
            return DEFAULT_SETTINGS.copy()
            
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Save user settings to file with validation
    
    Args:
        settings: Dict containing settings to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate model configuration before saving
        provider = settings.get("llm_provider", "ollama")
        model = settings.get("llm_model", "llama3.2:latest")
        
        if not validate_model_for_provider(provider, model):
            logger.error(f"Cannot save invalid model configuration: {provider}/{model}")
            return False
        
        # Ensure the data directory exists
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        settings["last_updated"] = datetime.now().isoformat()
        
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved user settings to {SETTINGS_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        return False

def update_setting(key: str, value: Any) -> bool:
    """
    Update a single setting with validation
    
    Args:
        key: Setting key to update
        value: New value for the setting
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings = load_settings()
        
        # Special validation for model-related settings
        if key == "llm_model":
            provider = settings.get("llm_provider", "ollama")
            if not validate_model_for_provider(provider, value):
                logger.error(f"Invalid model '{value}' for current provider '{provider}'")
                return False
        elif key == "llm_provider":
            model = settings.get("llm_model", "llama3.2:latest")
            if not validate_model_for_provider(value, model):
                logger.error(f"Current model '{model}' is not compatible with provider '{value}'")
                return False
        elif key == "embedding_model":
            if value not in AVAILABLE_EMBEDDING_MODELS:
                logger.error(f"Invalid embedding model '{value}'. Available: {AVAILABLE_EMBEDDING_MODELS}")
                return False
        
        settings[key] = value
        return save_settings(settings)
        
    except Exception as e:
        logger.error(f"Error updating setting {key}: {str(e)}")
        return False

def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific setting value
    
    Args:
        key: Setting key to retrieve
        default: Default value if key not found
        
    Returns:
        Setting value or default
    """
    try:
        settings = load_settings()
        return settings.get(key, default)
        
    except Exception as e:
        logger.error(f"Error getting setting {key}: {str(e)}")
        return default

def reset_settings() -> bool:
    """
    Reset all settings to defaults
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        return save_settings(DEFAULT_SETTINGS.copy())
        
    except Exception as e:
        logger.error(f"Error resetting settings: {str(e)}")
        return False

# Ollama models cache
_ollama_models_cache = None
_ollama_cache_timestamp = None
_ollama_cache_duration = 300  # 5 minutes cache

def get_ollama_models(force_refresh: bool = False) -> List[str]:
    """
    Fetch available Ollama models from local installation with caching
    
    Args:
        force_refresh: Force refresh cache even if still valid
    
    Returns:
        List of available model names
    """
    global _ollama_models_cache, _ollama_cache_timestamp
    
    # Check if we have valid cached data
    if (not force_refresh and 
        _ollama_models_cache is not None and 
        _ollama_cache_timestamp is not None and
        (datetime.now() - _ollama_cache_timestamp).seconds < _ollama_cache_duration):
        return _ollama_models_cache.copy()
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            
            # Update cache
            _ollama_models_cache = models
            _ollama_cache_timestamp = datetime.now()
            
            logger.info(f"Refreshed Ollama models cache: {len(models)} models found")
            return models
        else:
            logger.warning(f"Ollama API returned status {response.status_code}")
            # Return cached data if available, otherwise empty list
            return _ollama_models_cache.copy() if _ollama_models_cache else []
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to Ollama. Using cached data if available.")
        return _ollama_models_cache.copy() if _ollama_models_cache else []
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        return _ollama_models_cache.copy() if _ollama_models_cache else []

def clear_ollama_cache():
    """Clear Ollama models cache"""
    global _ollama_models_cache, _ollama_cache_timestamp
    _ollama_models_cache = None
    _ollama_cache_timestamp = None
    logger.info("Ollama models cache cleared")

def validate_model_for_provider(provider: str, model: str) -> bool:
    """
    Validate if a model name is compatible with the given provider
    
    Args:
        provider: LLM provider name
        model: Model name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if provider not in PROVIDER_MODEL_PATTERNS:
        logger.warning(f"Unknown provider: {provider}")
        return False
    
    provider_config = PROVIDER_MODEL_PATTERNS[provider]
    
    # Check against predefined models list (for OpenAI, Gemini)
    if "models" in provider_config:
        if model in provider_config["models"]:
            return True
        else:
            logger.warning(f"Model '{model}' not in approved list for {provider}. Available: {provider_config['models']}")
            return False
    
    # Check against patterns (for Ollama, HuggingFace)
    if "patterns" in provider_config:
        import re
        for pattern in provider_config["patterns"]:
            if re.match(pattern, model):                # For Ollama, check if model is installed (uses cache)
                if provider == "ollama":
                    available_models = get_ollama_models()
                    if available_models and model not in available_models:
                        logger.warning(f"Ollama model '{model}' not installed locally. Available: {available_models}")
                        return False
                return True
        
        logger.warning(f"Model '{model}' doesn't match expected pattern for {provider}. Examples: {provider_config.get('examples', [])}")
        return False
    
    return False

# Global models cache
_models_by_provider_cache = None
_models_cache_timestamp = None
_models_cache_duration = 300  # 5 minutes cache

def get_available_models_by_provider(force_refresh: bool = False) -> Dict[str, List[str]]:
    """
    Get available models organized by provider with caching
    
    Args:
        force_refresh: Force refresh cache even if still valid
    
    Returns:
        Dict with provider as key and list of models as value
    """
    global _models_by_provider_cache, _models_cache_timestamp
    
    # Check if we have valid cached data
    if (not force_refresh and 
        _models_by_provider_cache is not None and 
        _models_cache_timestamp is not None and
        (datetime.now() - _models_cache_timestamp).seconds < _models_cache_duration):
        return _models_by_provider_cache.copy()
    
    models_by_provider = {}
    
    # Ollama - get from local installation (uses its own cache)
    ollama_models = get_ollama_models(force_refresh)
    if ollama_models:
        models_by_provider["ollama"] = ollama_models
    else:
        # Fallback to common models if Ollama not available
        models_by_provider["ollama"] = ["llama3.2:latest", "llama3.1:latest", "mistral:latest"]
    
    # OpenAI - static list
    models_by_provider["openai"] = PROVIDER_MODEL_PATTERNS["openai"]["models"]
    
    # Gemini - static list
    models_by_provider["gemini"] = PROVIDER_MODEL_PATTERNS["gemini"]["models"]
    
    # HuggingFace - provide examples (static)
    models_by_provider["huggingface"] = PROVIDER_MODEL_PATTERNS["huggingface"]["examples"]
    
    # Update cache
    _models_by_provider_cache = models_by_provider
    _models_cache_timestamp = datetime.now()
    
    return models_by_provider.copy()

def clear_models_cache():
    """Clear all models cache"""
    global _models_by_provider_cache, _models_cache_timestamp
    _models_by_provider_cache = None
    _models_cache_timestamp = None
    clear_ollama_cache()
    logger.info("All models cache cleared")

def update_model_with_validation(provider: str, model: str) -> bool:
    """
    Update LLM model with proper validation
    
    Args:
        provider: LLM provider
        model: Model name
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate provider/model combination
    if not validate_model_for_provider(provider, model):
        logger.error(f"Invalid model '{model}' for provider '{provider}'")
        return False
    
    # Load current settings
    settings = load_settings()
    
    # Update both provider and model
    settings["llm_provider"] = provider
    settings["llm_model"] = model
    
    # Save updated settings
    success = save_settings(settings)
    
    if success:
        logger.info(f"Model updated successfully: {provider}/{model}")
    else:
        logger.error(f"Failed to save model update: {provider}/{model}")
    
    return success
