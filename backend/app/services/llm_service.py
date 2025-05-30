import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import requests
import aiohttp
from functools import lru_cache
import datetime

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a dedicated logger for the LLM service
logger = logging.getLogger(__name__)

# Add a file handler to log to a file
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler(f'logs/llm_service_{datetime.datetime.now().strftime("%Y%m%d")}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Placeholder for LLM API key and endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Default models
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_GEMINI_MODEL = "gemini-pro"
DEFAULT_HUGGINGFACE_MODEL = "PleIAs/Pleias-RAG-1B"

logger.info(f"LLM Service initialized. OpenAI API configured: {OPENAI_API_KEY is not None}")
logger.info(f"Gemini API configured: {GEMINI_API_KEY is not None}")
logger.info(f"Hugging Face API configured: {HUGGINGFACE_API_KEY is not None}")
logger.info(f"Ollama endpoint: {OLLAMA_ENDPOINT}")

# Cache to store recent LLM responses
query_cache = {}
# Dict to track LLM processing status
llm_status = {
    "is_processing": False,
    "last_model_used": None,
    "last_query_time": None,
    "total_queries": 0,
    "successful_queries": 0
}

# Global model cache to avoid reloading Hugging Face models
_hf_model_cache = {}

# Model-specific configurations for Hugging Face models
HUGGINGFACE_MODEL_CONFIGS = {
    "PleIAs/Pleias-RAG-1B": {
        "prompt_template": "Question: {prompt}\nAnswer:",
        "max_new_tokens": 150,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "top_p": 0.9,
        "top_k": 50
    },
    "default": {
        "prompt_template": "{prompt}",
        "max_new_tokens": 200,
        "temperature": 0.8,
        "repetition_penalty": 1.1,
        "top_p": 0.9,
        "top_k": 50
    }
}

def make_rag_prompt(query: str, relevant_passages: List[str]) -> str:
    """
    Create an enhanced RAG prompt for LLMs with optimized instructions
    
    Args:
        query: User's question
        relevant_passages: List of relevant text passages
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Clean and join the passages with clear section markers
    cleaned_passages = []
    for i, passage in enumerate(relevant_passages):
        # Clean the passage of problematic characters
        cleaned = passage.replace("\n", " ").strip()
        # Add a section marker with reference number
        cleaned_passages.append(f"[{i+1}] {cleaned}")
    
    context_text = "\n\n".join(cleaned_passages)
    
    # Create an enhanced system prompt
    prompt = f"""You are a knowledgeable assistant answering questions based on provided reference passages.

INSTRUCTIONS:
1. Answer ONLY based on the information in the provided passages.
2. If the passages don't contain relevant information, say "I don't have enough information to answer that question."
3. Include citation numbers [1], [2], etc. when referencing specific information from the passages.
4. Provide thorough, accurate answers while explaining complex concepts clearly.
5. Use a concise, informative tone.

QUESTION: {query}

REFERENCE PASSAGES:
{context_text}

ANSWER:
"""
    
    logger.info(f"Created enhanced RAG prompt with {len(relevant_passages)} reference passages")
    logger.debug(f"Prompt first 100 chars: {prompt[:100]}...")
    
    return prompt

def create_prompt_with_context(question: str, context_documents: List[str]) -> str:
    """
    Create a prompt for the LLM by combining the question and context documents
    
    Args:
        question: User's question
        context_documents: List of relevant document chunks
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Use the enhanced RAG prompt function
    return make_rag_prompt(question, context_documents)

# Calculate a cache key based on question and context
def get_cache_key(question: str, context_documents: List[str], model_config: Dict[str, Any] = None) -> str:
    """Generate a cache key from the question, context, and model config"""
    context_hash = hash(tuple(sorted(context_documents)))
    model_hash = hash(str(model_config)) if model_config else 0
    return f"{question.strip().lower()}:{context_hash}:{model_hash}"

# Simple LRU cache for the Ollama client to avoid repeated calls for the same query
@lru_cache(maxsize=10)
def get_cached_ollama_response(prompt_hash: str):
    """Cached version of Ollama query to avoid repeated calls"""
    # This is just a key for the cache, the actual query uses the real prompt
    return None

def query_ollama_llm(prompt: str, model_config: Dict[str, Any] = None) -> tuple:
    """
    Query an Ollama LLM instance
    
    Args:
        prompt: The formatted prompt
        model_config: Configuration for the model
        
    Returns:
        tuple: (LLM response, model info)
    """
    global llm_status
    
    # Get model from config or use default
    model = DEFAULT_OLLAMA_MODEL
    if model_config and 'model' in model_config:
        model = model_config.get('model')
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_model_used"] = f"ollama/{model}"
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    logger.info(f"Querying Ollama LLM with model: {model}")
    
    # Calculate a hash for the cache
    prompt_hash = hash(prompt + model)
    
    # Check if we have a cached response
    cache_result = get_cached_ollama_response(prompt_hash)
    if cache_result is not None:
        logger.info("Using cached LLM response")
        llm_status["is_processing"] = False
        llm_status["successful_queries"] += 1
        return cache_result, f"ollama/{model} (cached)"
    
    # Enhance the Ollama prompt with system instruction
    system_instruction = "You are an AI assistant that answers questions based only on provided references. Always cite sources with [1], [2], etc."
    
    # For Ollama, we include the system instruction at the beginning of the prompt
    full_prompt = f"{system_instruction}\n\n{prompt}"
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        # Add options to speed up generation
        "options": {
            "temperature": 0.1,  # Lower temperature for more deterministic responses
            "top_p": 0.9,
            "num_predict": 512,  # Reduced from 60000 for faster responses
            "num_ctx": 2048,     # Reduced context window for faster processing
            "top_k": 40,         # Limit top-k sampling for speed
            "repeat_penalty": 1.1
        }
    }
    
    logger.info(f"LLM parameters: temperature={payload['options']['temperature']}, top_p={payload['options']['top_p']}")
    
    try:
        # Check if model is loaded first to avoid long loading times
        logger.info("Checking if model is loaded in Ollama")
        model_status_response = requests.get(f"http://localhost:11434/api/tags", timeout=2)
        
        if model_status_response.status_code == 200:
            model_list = model_status_response.json().get("models", [])
            model_loaded = any(m.get("name") == model for m in model_list)
            
            if not model_loaded:
                # If model not loaded, return informative message
                logger.warning(f"Model {model} not loaded in Ollama")
                llm_status["is_processing"] = False
                return f"The model '{model}' is not currently loaded in Ollama. Please run 'ollama pull {model}' first.", f"ollama/{model} (not loaded)"
        
        # Make the API call with increased timeout for Ollama
        start_time = time.time()
        logger.info("Sending request to Ollama API")
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)  # Increased from 30 to 60
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            processing_time = time.time() - start_time
            
            # Log sample of response
            logger.info(f"LLM responded in {processing_time:.2f} seconds")
            logger.info(f"Response sample: {result[:100]}...")
            
            # Cache successful responses
            get_cached_ollama_response.cache_clear()  # Clear old cache
            get_cached_ollama_response(prompt_hash)  # Set new cache entry
            
            # Update status
            llm_status["is_processing"] = False
            llm_status["successful_queries"] += 1
            
            return result, f"ollama/{model}"
        else:
            # If we can't reach Ollama, provide a useful error message
            logger.error(f"Ollama API error: {response.status_code}")
            llm_status["is_processing"] = False
            return f"I couldn't get a response from the language model. Status code: {response.status_code}. Please check that the Ollama service is running with the {model} model loaded.", f"ollama/{model} (error)"
    except requests.exceptions.ConnectionError:
        # Handle connection errors specifically
        logger.error("Connection error when contacting Ollama API")
        llm_status["is_processing"] = False
        return "I couldn't connect to the language model. Please check that the Ollama service is running.", f"ollama/{model} (connection error)"
    except requests.exceptions.Timeout:
        # Handle timeout errors
        logger.error("Timeout when contacting Ollama API")
        llm_status["is_processing"] = False
        return "The request to the language model timed out. Please try again later.", f"ollama/{model} (timeout)"
    except Exception as e:
        # Generic error handling
        logger.error(f"Error querying Ollama: {str(e)}")
        llm_status["is_processing"] = False
        return f"An error occurred while communicating with the language model: {str(e)}", f"ollama/{model} (error)"

def query_openai_llm(prompt: str, model_config: Dict[str, Any] = None) -> tuple:
    """
    Query OpenAI's LLM API
    
    Args:
        prompt: The formatted prompt
        model_config: Configuration for the model
        
    Returns:
        tuple: (LLM response, model info)
    """
    global llm_status
    
    # Get API key and model from config or use defaults
    api_key = OPENAI_API_KEY
    model = DEFAULT_OPENAI_MODEL
    
    if model_config:
        if 'api_key' in model_config:
            api_key = model_config.get('api_key')
        if 'model' in model_config:
            model = model_config.get('model')
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_model_used"] = f"openai/{model}"
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    logger.info(f"Querying OpenAI LLM with model: {model}")
    
    # Attempt to use OpenAI if configured
    try:
        import openai
        
        if not api_key:
            logger.error("OpenAI API key not configured")
            llm_status["is_processing"] = False
            return "OpenAI API key is not configured. Please provide an API key to use OpenAI models.", f"openai/{model} (no API key)"
        
        openai.api_key = api_key
        
        logger.info("Sending request to OpenAI API")
        start_time = time.time()
        
        # For OpenAI, we use the chat API with system and user messages
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for question-answering. Answer based only on the provided information, include citation numbers [1], [2], etc. when referencing specific information, and acknowledge when you don't have enough information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60000,
            temperature=0.1  # Lower temperature for more deterministic responses
        )
        
        processing_time = time.time() - start_time
        result = response.choices[0].message.content
        
        logger.info(f"OpenAI responded in {processing_time:.2f} seconds")
        logger.info(f"Response sample: {result[:100]}...")
        
        llm_status["is_processing"] = False
        llm_status["successful_queries"] += 1
        return result, f"openai/{model}"
    except ImportError:
        logger.error("OpenAI Python package not installed")
        llm_status["is_processing"] = False
        return "OpenAI Python package is not installed. Please install it with 'pip install openai'.", f"openai/{model} (not installed)"
    except Exception as e:
        logger.error(f"Error using OpenAI: {str(e)}")
        llm_status["is_processing"] = False
        return f"Error with OpenAI API: {str(e)}", f"openai/{model} (error)"

def query_gemini_llm(prompt: str, model_config: Dict[str, Any] = None) -> tuple:
    """
    Query Google's Gemini API
    
    Args:
        prompt: The formatted prompt
        model_config: Configuration for the model
        
    Returns:
        tuple: (LLM response, model info)
    """
    global llm_status
    
    # Get API key and model from config or use defaults
    api_key = GEMINI_API_KEY
    model = DEFAULT_GEMINI_MODEL
    
    if model_config:
        if 'api_key' in model_config:
            api_key = model_config.get('api_key')
        if 'model' in model_config:
            model = model_config.get('model')
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_model_used"] = f"gemini/{model}"
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    logger.info(f"Querying Gemini LLM with model: {model}")
    
    try:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.error("Google Generative AI package not installed")
            llm_status["is_processing"] = False
            return "Google Generative AI package is not installed. Please install it with 'pip install google-generativeai'.", f"gemini/{model} (not installed)"
        
        if not api_key:
            logger.error("Gemini API key not configured")
            llm_status["is_processing"] = False
            return "Gemini API key is not configured. Please provide an API key to use Gemini models.", f"gemini/{model} (no API key)"
        
        # Configure the Gemini API
        client = genai.Client(api_key=api_key)
        
        logger.info("Sending request to Gemini API")
        start_time = time.time()
        
        # For Gemini, we use enhanced system instructions
        system_instruction = """Answer questions based only on the provided reference passages. 
Include citation numbers [1], [2], etc. when referencing specific information.
If the information isn't in the passages, state that you don't have enough information.
Be accurate, clear, and concise in your responses."""
        
        # Generate content with the specified configuration
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=60000,
                temperature=0.5,
                system_instruction=system_instruction
            )
        )
        
        processing_time = time.time() - start_time
        result = response.text
        
        logger.info(f"Gemini responded in {processing_time:.2f} seconds")
        logger.info(f"Response sample: {result[:100]}...")
        
        llm_status["is_processing"] = False
        llm_status["successful_queries"] += 1
        return result, f"gemini/{model}"
    except Exception as e:
        logger.error(f"Error using Gemini: {str(e)}")
        llm_status["is_processing"] = False
        return f"Error with Gemini API: {str(e)}", f"gemini/{model} (error)"

def query_huggingface_llm(prompt: str, model_config: Dict[str, Any] = None) -> tuple:
    """
    Query Hugging Face models either locally (using transformers) or via API
    
    Args:
        prompt: The formatted prompt
        model_config: Configuration for the model
        
    Returns:
        tuple: (LLM response, model info)
    """
    global llm_status
    
    # Get model from config or use default
    model = DEFAULT_HUGGINGFACE_MODEL
    api_key = None  # Start with no API key
    use_local = True  # Default to local usage
    
    if model_config:
        if 'model' in model_config:
            model = model_config.get('model')
        # Only use API if explicitly provided in model_config
        if 'api_key' in model_config and model_config.get('api_key'):
            api_key = model_config.get('api_key')
            use_local = False  # If API key is provided in config, use API
        if 'use_local' in model_config:
            use_local = model_config.get('use_local', True)
    
    # If still no API key and use_local is False, fall back to environment variable
    if not use_local and not api_key:
        api_key = HUGGINGFACE_API_KEY
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_model_used"] = f"huggingface/{model}"
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    logger.info(f"Querying Hugging Face LLM with model: {model} (local: {use_local})")
    
    try:
        # Try local transformers first if requested or no API key
        if use_local:
            try:
                logger.info("Using local Hugging Face transformers")
                
                # Get or load cached model
                model_data = get_or_load_hf_model(model)
                tokenizer = model_data['tokenizer']
                model_obj = model_data['model']
                
                # Get model-specific configuration
                model_config = HUGGINGFACE_MODEL_CONFIGS.get(model, HUGGINGFACE_MODEL_CONFIGS["default"])
                
                # Format prompt using model-specific template
                formatted_prompt = model_config["prompt_template"].format(prompt=prompt)
                
                start_time = time.time()
                
                # Tokenize input (no padding needed for single input)
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Remove token_type_ids if present (not needed for most causal LM models)
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                
                # Move to same device as model
                device = next(model_obj.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate response with optimized parameters
                import torch
                with torch.no_grad():
                    outputs = model_obj.generate(
                        **inputs,
                        max_new_tokens=model_config["max_new_tokens"],
                        min_new_tokens=10,
                        do_sample=True,
                        temperature=model_config["temperature"],
                        top_p=model_config.get("top_p", 0.9),
                        top_k=model_config.get("top_k", 50),
                        repetition_penalty=model_config["repetition_penalty"],
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Check for problematic output and provide fallback
                problematic_patterns = ['__', '0/1/', '1/1/', '(1)', '(2)', '(3)']
                if (not generated_text or 
                    len(generated_text.strip()) < 10 or 
                    any(pattern in generated_text for pattern in problematic_patterns) or
                    generated_text.count('\n') > 5):  # Too many line breaks suggest formatting issues
                    
                    generated_text = "I apologize, but I'm having difficulty generating a proper response. Please try rephrasing your question or using a different model."
                
                processing_time = time.time() - start_time
                
                logger.info(f"Hugging Face (local) responded in {processing_time:.2f} seconds")
                logger.info(f"Response sample: {generated_text[:100]}...")
                
                llm_status["is_processing"] = False
                llm_status["successful_queries"] += 1
                return generated_text, f"huggingface/{model} (local)"
                
            except ImportError:
                logger.error("Transformers library not available")
                llm_status["is_processing"] = False
                return "Hugging Face transformers library is not installed. Please install it with 'pip install transformers torch accelerate' to use local models.", f"huggingface/{model} (no transformers)"
            except Exception as e:
                logger.error(f"Local model failed: {str(e)}")
                llm_status["is_processing"] = False
                return f"Local Hugging Face model failed: {str(e)}. Please ensure the transformers library is properly installed with 'pip install transformers torch accelerate'.", f"huggingface/{model} (local error)"
        
        # Use API if explicitly requested and API key is available
        elif api_key:
            logger.info("Using Hugging Face Inference API")
            
            # Hugging Face Inference API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # For text generation models, the payload format
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            start_time = time.time()
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Handle different response formats
                if isinstance(result_data, list) and len(result_data) > 0:
                    result = result_data[0].get('generated_text', '')
                elif isinstance(result_data, dict):
                    result = result_data.get('generated_text', str(result_data))
                else:
                    result = str(result_data)
                
                processing_time = time.time() - start_time
                
                logger.info(f"Hugging Face API responded in {processing_time:.2f} seconds")
                logger.info(f"Response sample: {result[:100]}...")
                
                llm_status["is_processing"] = False
                llm_status["successful_queries"] += 1
                return result, f"huggingface/{model} (api)"
            else:
                error_msg = f"Hugging Face API error: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg += f" - {error_data['error']}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                
                logger.error(error_msg)
                llm_status["is_processing"] = False
                return f"Error with Hugging Face API: {error_msg}", f"huggingface/{model} (api error)"
        else:
            llm_status["is_processing"] = False
            return "Hugging Face API key is not configured. Please provide an API key to use Hugging Face models.", f"huggingface/{model} (no api key)"
            
    except requests.exceptions.Timeout:
        logger.error("Timeout when contacting Hugging Face API")
        llm_status["is_processing"] = False
        return "The request to Hugging Face timed out. Please try again later.", f"huggingface/{model} (timeout)"
    except Exception as e:
        logger.error(f"Error using Hugging Face: {str(e)}")
        llm_status["is_processing"] = False
        return f"Error with Hugging Face: {str(e)}", f"huggingface/{model} (error)"

def get_or_load_hf_model(model_name: str):
    """Get or load a Hugging Face model from cache"""
    if model_name not in _hf_model_cache:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            logger.info(f"Loading Hugging Face model: {model_name}")
            
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Ensure pad_token is set properly
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Also ensure pad_token_id is set
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto" if model_name == "PleIAs/Pleias-RAG-1B" else None
            )
            
            _hf_model_cache[model_name] = {
                'tokenizer': tokenizer,
                'model': model
            }
            logger.info(f"Model {model_name} loaded and cached successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    return _hf_model_cache[model_name]

def warmup_hf_model(model_name: str = None):
    """Warmup a Hugging Face model by loading it into cache"""
    if model_name is None:
        model_name = DEFAULT_HUGGINGFACE_MODEL
    
    try:
        logger.info(f"Warming up Hugging Face model: {model_name}")
        pipe = get_or_load_hf_model(model_name)
        
        # Do a quick test generation to fully warm up the model
        test_result = pipe(
            "Hello, this is a test.",
            max_new_tokens=10,
            do_sample=False,
            return_full_text=False
        )
        logger.info(f"Model {model_name} warmed up successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to warm up model {model_name}: {str(e)}")
        return False

def get_answer_from_llm(question: str, context_documents: List[str], model_config: Dict[str, Any] = None, custom_prompt: str = None) -> tuple:
    """
    Get an answer from the LLM based on the context documents
    
    Args:
        question: User's question
        context_documents: List of relevant document chunks
        model_config: Configuration for the model
        custom_prompt: Optional custom prompt to override default RAG prompt
        
    Returns:
        tuple: (answer, model_info)
    """
    # Log the entire question for better tracking
    logger.info(f"Getting answer for question: '{question}'")
    logger.info(f"Using {len(context_documents)} context documents")
    
    if model_config:
        logger.info(f"Using model config: {model_config}")
    
    if custom_prompt:
        logger.info("Using custom prompt instead of RAG prompt")
    
    # Check cache first (only for non-custom prompts)
    cache_key = None
    if not custom_prompt:
        cache_key = get_cache_key(question, context_documents, model_config)
        if cache_key in query_cache:
            logger.info("Using cached response")
            return query_cache[cache_key], "cached response"
    
    # Create a prompt with the context or use custom prompt
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = create_prompt_with_context(question, context_documents)
    
    # Determine which provider to use
    provider = "ollama"  # Default provider
    
    if model_config and 'provider' in model_config:
        provider = model_config.get('provider')
    
    answer = None
    model_info = None
    
    # Query the appropriate provider
    if provider == "openai":
        answer, model_info = query_openai_llm(prompt, model_config)
    elif provider == "gemini":
        answer, model_info = query_gemini_llm(prompt, model_config)
    elif provider == "huggingface":
        answer, model_info = query_huggingface_llm(prompt, model_config)
    else:  # Default to Ollama
        answer, model_info = query_ollama_llm(prompt, model_config)
    
    # Log the complete model response for better tracking
    logger.info(f"LLM response: '{answer[:200]}...' (truncated)")
    logger.info(f"Model info: {model_info}")
    
    # Cache the result (only for non-custom prompts)
    if answer and cache_key:
        query_cache[cache_key] = answer
    
    return answer, model_info

def get_llm_status() -> Dict[str, Any]:
    """
    Get the current status of the LLM service
    
    Returns:
        Dict: Status information
    """
    status_copy = llm_status.copy()
    
    # Add extra info
    status_copy["cache_size"] = len(query_cache)
    
    # Calculate time since last query if applicable
    if status_copy["last_query_time"]:
        status_copy["time_since_last_query"] = time.time() - status_copy["last_query_time"]
    
    return status_copy

async def get_streaming_response(question: str, context_documents: List[str], model_config: Dict[str, Any] = None) -> AsyncIterator[str]:
    raise NotImplementedError("Streaming is no longer supported.")

async def stream_from_ollama(prompt: str, model_config: Dict[str, Any] = None) -> AsyncIterator[str]:
    raise NotImplementedError("Streaming is no longer supported.")

async def stream_from_openai(prompt: str, model_config: Dict[str, Any] = None) -> AsyncIterator[str]:
    raise NotImplementedError("Streaming is no longer supported.")

async def stream_from_gemini(prompt: str, model_config: Dict[str, Any] = None) -> AsyncIterator[str]:
    raise NotImplementedError("Streaming is no longer supported.")
    
    # Update model used in status
    llm_status["last_model_used"] = f"gemini/{model}"
    
    logger.info(f"Streaming from Gemini with model: {model}")
    
    # Enhanced system instruction for Gemini
    system_instruction = """Answer questions based only on the provided reference passages. 
Include citation numbers [1], [2], etc. when referencing specific information.
If the information isn't in the passages, state that you don't have enough information.
Be accurate, clear, and concise in your responses."""
    
    try:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            yield "Google Generative AI package is not installed. Please install it with 'pip install google-generativeai'."
            return
        
        if not api_key:
            yield "Gemini API key is not configured. Please provide an API key to use Gemini models."
            return
        

        # Configure the Gemini API
        client = genai.Client(api_key=api_key)
        # Generate content with streaming (no duplicate system_instruction)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=60000,
                temperature=0.1
            ),
            stream=True
        )

        # Process the streaming response
        async for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f"Error streaming from Gemini: {str(e)}")
        yield f"Error streaming from Gemini: {str(e)}"

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available LLM models
    
    Returns:
        List[Dict]: List of model information
    """
    models = []
    
    # Try to get Ollama models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data and "models" in data:
                for model_info in data["models"]:
                    models.append({
                        "name": model_info["name"],
                        "provider": "ollama",
                        "description": f"Ollama - {model_info.get('size', 'Unknown size')}"
                    })
        else:
            # Add default Ollama model if we can't connect
            models.append({
                "name": DEFAULT_OLLAMA_MODEL,
                "provider": "ollama",
                "description": "Ollama - Default model"
            })
    except:
        # Add default Ollama model if we can't connect
        models.append({
            "name": DEFAULT_OLLAMA_MODEL,
            "provider": "ollama",
            "description": "Ollama - Default model"
        })
    
    # Add OpenAI models
    models.append({
        "name": "gpt-3.5-turbo",
        "provider": "openai",
        "description": "OpenAI - GPT-3.5 Turbo"
    })
    models.append({
        "name": "gpt-4",
        "provider": "openai",
        "description": "OpenAI - GPT-4"
    })
    
    # Add Gemini models
    models.append({
        "name": "gemini-1.5-flash",
        "provider": "gemini",
        "description": "Google - Gemini Pro"
    })
    
    # Add Hugging Face models
    models.append({
        "name": "PleIAs/Pleias-RAG-1B",
        "provider": "huggingface",
        "description": "Hugging Face - PleIAs RAG 1B (1.2B parameters Small Reasoning Model for RAG and source summarization)"
    })
    models.append({
        "name": "microsoft/DialoGPT-medium",
        "provider": "huggingface", 
        "description": "Hugging Face - DialoGPT Medium"
    })
    models.append({
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "provider": "huggingface",
        "description": "Hugging Face - Llama 2 7B Chat"
    })
    
    return models