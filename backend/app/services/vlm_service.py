import os
import json
import time
import logging
import base64
from typing import Dict, Any, List, Optional, Tuple
import requests
from PIL import Image
import io

# Configure logging
logger = logging.getLogger(__name__)

# API Keys and endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Default VLM models
DEFAULT_OLLAMA_VLM = "llava:latest"
DEFAULT_OPENAI_VLM = "gpt-4o"
DEFAULT_GEMINI_VLM = "gemini-1.5-pro"
DEFAULT_HUGGINGFACE_VLM = "Salesforce/blip-image-captioning-base"

# VLM processing status
vlm_status = {
    "is_processing": False,
    "last_model_used": None,
    "last_query_time": None,
    "total_queries": 0,
    "successful_queries": 0
}

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for API calls"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

def validate_image(image_path: str) -> bool:
    """Validate if image exists and is readable"""
    try:
        if not os.path.exists(image_path):
            return False
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def describe_image(image_path: str, model_config: Dict[str, Any] = None) -> Tuple[str, str]:
    """
    Generate description for an image using configured VLM
    
    Args:
        image_path: Path to the image file
        model_config: VLM configuration
        
    Returns:
        tuple: (description, model_info)
    """
    global vlm_status
    
    # Validate image first
    if not validate_image(image_path):
        return f"Error: Invalid or missing image at {image_path}", "error"
    
    # Determine provider and model
    provider = "ollama"  # Default
    model = DEFAULT_OLLAMA_VLM
    
    if model_config:
        provider = model_config.get('provider', provider)
        model = model_config.get('model', model)
    
    # Update status
    vlm_status["is_processing"] = True
    vlm_status["last_model_used"] = f"{provider}/{model}"
    vlm_status["last_query_time"] = time.time()
    vlm_status["total_queries"] += 1
    
    logger.info(f"Describing image {image_path} with {provider}/{model}")
    
    # Retry logic (similar to LLM service)
    max_retries = 2
    retry_delay = 3
    
    for attempt in range(max_retries + 1):
        try:
            description = None
            model_info = None
            
            if provider == "openai":
                description, model_info = query_openai_vlm(image_path, model)
            elif provider == "gemini":
                description, model_info = query_gemini_vlm(image_path, model)
            elif provider == "huggingface":
                description, model_info = query_huggingface_vlm(image_path, model)
            else:  # ollama
                description, model_info = query_ollama_vlm(image_path, model)
            
            # Success
            vlm_status["is_processing"] = False
            vlm_status["successful_queries"] += 1
            return description, model_info
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"VLM error on attempt {attempt + 1}: {error_str}")
            
            # Check for retryable errors
            if ("429" in error_str or "503" in error_str or "UNAVAILABLE" in error_str):
                if attempt < max_retries:
                    logger.warning(f"Retryable VLM error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
            
            # Non-retryable or max retries reached
            vlm_status["is_processing"] = False
            return f"Error describing image: {error_str}", f"{provider}/{model} (error)"
    
    vlm_status["is_processing"] = False
    return "Error: Max retries reached for image description", f"{provider}/{model} (failed)"

def query_ollama_vlm(image_path: str, model: str) -> Tuple[str, str]:
    """Query Ollama VLM (like llava)"""
    try:
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Error: Could not encode image", f"ollama/{model} (encoding_error)"
        
        payload = {
            "model": model,
            "prompt": "Describe this image in detail. Focus on the main content, objects, text, and any important visual elements.",
            "images": [base64_image],
            "stream": False
        }
        
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        description = result.get("response", "No description generated")
        
        logger.info(f"Ollama VLM description generated: {description[:100]}...")
        return description, f"ollama/{model}"
        
    except Exception as e:
        logger.error(f"Ollama VLM error: {str(e)}")
        raise

def query_openai_vlm(image_path: str, model: str) -> Tuple[str, str]:
    """Query OpenAI Vision models"""
    try:
        import openai
        
        if not OPENAI_API_KEY:
            return "Error: OpenAI API key not configured", f"openai/{model} (no_api_key)"
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Error: Could not encode image", f"openai/{model} (encoding_error)"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail. Focus on the main content, objects, text, and any important visual elements."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        logger.info(f"OpenAI Vision description generated: {description[:100]}...")
        return description, f"openai/{model}"
        
    except Exception as e:
        logger.error(f"OpenAI Vision error: {str(e)}")
        if "429" in str(e):
            raise Exception(f"429 RATE_LIMITED: {str(e)}")
        elif "503" in str(e):
            raise Exception(f"503 UNAVAILABLE: {str(e)}")
        else:
            raise

def query_gemini_vlm(image_path: str, model: str) -> Tuple[str, str]:
    """Query Gemini Vision models"""
    try:
        from google import genai
        from google.genai import types
        
        if not GEMINI_API_KEY:
            return "Error: Gemini API key not configured", f"gemini/{model} (no_api_key)"
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Read image file
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        response = client.models.generate_content(
            model=model,
            contents=[
                "Describe this image in detail. Focus on the main content, objects, text, and any important visual elements.",
                {"mime_type": "image/jpeg", "data": image_data}
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.3
            )
        )
        
        description = response.text
        logger.info(f"Gemini Vision description generated: {description[:100]}...")
        return description, f"gemini/{model}"
        
    except Exception as e:
        logger.error(f"Gemini Vision error: {str(e)}")
        if "429" in str(e):
            raise Exception(f"429 RATE_LIMITED: {str(e)}")
        elif "503" in str(e) or "UNAVAILABLE" in str(e):
            raise Exception(f"503 UNAVAILABLE: {str(e)}")
        else:
            raise

def query_huggingface_vlm(image_path: str, model: str) -> Tuple[str, str]:
    """Query HuggingFace Vision models"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        
        # Load model and processor
        processor = BlipProcessor.from_pretrained(model)
        model_obj = BlipForConditionalGeneration.from_pretrained(model)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Generate description
        inputs = processor(image, "a photography of", return_tensors="pt")
        out = model_obj.generate(**inputs, max_length=100)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        logger.info(f"HuggingFace Vision description generated: {description[:100]}...")
        return description, f"huggingface/{model}"
        
    except Exception as e:
        logger.error(f"HuggingFace Vision error: {str(e)}")
        raise

def get_vlm_status() -> Dict[str, Any]:
    """Get current VLM service status"""
    return {
        **vlm_status,
        "available_providers": {
            "ollama": True,  # Always available if Ollama is running
            "openai": OPENAI_API_KEY is not None,
            "gemini": GEMINI_API_KEY is not None,
            "huggingface": True  # Available if transformers is installed
        }
    }

def get_available_vlm_models() -> Dict[str, List[str]]:
    """Get available VLM models for each provider"""
    return {
        "ollama": ["llava:latest", "llava:7b", "llava:13b", "llava-phi3", "bakllava"],
        "openai": ["gpt-4o", "gpt-4-vision-preview", "gpt-4o-mini"],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision"],
        "huggingface": [
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip2-opt-2.7b",
            "microsoft/git-base"
        ]
    }

logger.info("VLM Service initialized")
logger.info(f"Available providers: OpenAI={OPENAI_API_KEY is not None}, Gemini={GEMINI_API_KEY is not None}")
