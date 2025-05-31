import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Settings file location
SETTINGS_FILE = Path(__file__).parent.parent / "data" / "user_settings.json"

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
    Save user settings to file
    
    Args:
        settings: Dict containing settings to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the data directory exists
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        from datetime import datetime
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
    Update a single setting
    
    Args:
        key: Setting key to update
        value: New value for the setting
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings = load_settings()
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
