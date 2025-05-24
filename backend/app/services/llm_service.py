import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import requests
import aiohttp
from functools import lru_cache

# Placeholder for LLM API key and endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")

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

def create_prompt_with_context(question: str, context_documents: List[str]) -> str:
    """
    Create a prompt for the LLM by combining the question and context documents
    
    Args:
        question: User's question
        context_documents: List of relevant document chunks
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Join context documents with separators
    context_text = "\n\n---\n\n".join(context_documents)
    
    # Create prompt template with context and question
    prompt = f"""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.
Be concise and focus on the information provided in the context.

Context:
{context_text}

Question: 
{question}

Answer:
"""
    return prompt

# Calculate a cache key based on question and context
def get_cache_key(question: str, context_documents: List[str]) -> str:
    """Generate a cache key from the question and a hash of the context"""
    context_hash = hash(tuple(sorted(context_documents)))
    return f"{question.strip().lower()}:{context_hash}"

# Simple LRU cache for the Ollama client to avoid repeated calls for the same query
@lru_cache(maxsize=10)
def get_cached_ollama_response(prompt_hash: str):
    """Cached version of Ollama query to avoid repeated calls"""
    # This is just a key for the cache, the actual query uses the real prompt
    return None

def query_ollama_llm(prompt: str, model: str = "llama3") -> str:
    """
    Query an Ollama LLM instance
    
    Args:
        prompt: The formatted prompt
        model: The Ollama model to use
        
    Returns:
        str: LLM response
    """
    global llm_status
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_model_used"] = model
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    # Calculate a hash for the cache
    prompt_hash = hash(prompt)
    
    # Check if we have a cached response
    cache_result = get_cached_ollama_response(prompt_hash)
    if cache_result is not None:
        llm_status["is_processing"] = False
        llm_status["successful_queries"] += 1
        return cache_result
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Add options to potentially speed up generation
        "options": {
            "temperature": 0.1,  # Lower temperature for more deterministic responses
            "top_p": 0.9,
            "num_predict": 1024  # Limit token generation for faster responses
        }
    }
    
    try:
        # Check if model is loaded first to avoid long loading times
        model_status_response = requests.get(f"http://localhost:11434/api/tags", timeout=2)
        
        if model_status_response.status_code == 200:
            model_list = model_status_response.json().get("models", [])
            model_loaded = any(m.get("name") == model for m in model_list)
            
            if not model_loaded:
                # If model not loaded, return informative message
                llm_status["is_processing"] = False
                return f"The model '{model}' is not currently loaded in Ollama. Please run 'ollama pull {model}' first."
        
        # Make the API call
        start_time = time.time()
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            
            # Cache successful responses
            get_cached_ollama_response.cache_clear()  # Clear old cache
            get_cached_ollama_response(prompt_hash)  # Set new cache entry
            
            # Update status
            llm_status["is_processing"] = False
            llm_status["successful_queries"] += 1
            
            return result
        else:
            # If we can't reach Ollama, provide a useful error message
            llm_status["is_processing"] = False
            return f"I couldn't get a response from the language model. Status code: {response.status_code}. Please check that the Ollama service is running with the {model} model loaded."
    except requests.exceptions.ConnectionError:
        # Handle connection errors specifically
        llm_status["is_processing"] = False
        return "I couldn't connect to the language model. Please check that the Ollama service is running."
    except requests.exceptions.Timeout:
        # Handle timeout errors
        llm_status["is_processing"] = False
        return "The request to the language model timed out. Please try again later."
    except Exception as e:
        # Generic error handling
        llm_status["is_processing"] = False
        return f"An error occurred while communicating with the language model: {str(e)}"

def query_openai_llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Query OpenAI's LLM API
    
    Args:
        prompt: The formatted prompt
        model: The OpenAI model to use
        
    Returns:
        str: LLM response
    """
    global llm_status
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_model_used"] = model
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    # Attempt to use OpenAI if configured
    try:
        import openai
        
        if not OPENAI_API_KEY:
            llm_status["is_processing"] = False
            return "OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment variables."
        
        openai.api_key = OPENAI_API_KEY
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for question-answering."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1  # Lower temperature for more deterministic responses
        )
        
        llm_status["is_processing"] = False
        llm_status["successful_queries"] += 1
        return response.choices[0].message.content
    except ImportError:
        llm_status["is_processing"] = False
        return "OpenAI Python package is not installed. Please install it with 'pip install openai'."
    except Exception as e:
        llm_status["is_processing"] = False
        return f"Error calling OpenAI API: {str(e)}"

def get_answer_from_llm(question: str, context_documents: List[str]) -> str:
    """
    Get an answer from the LLM using the question and context
    
    Args:
        question: User's question
        context_documents: List of relevant document chunks
        
    Returns:
        str: LLM's answer
    """
    # Check cache first
    cache_key = get_cache_key(question, context_documents)
    if cache_key in query_cache:
        return query_cache[cache_key]
    
    # Create prompt
    prompt = create_prompt_with_context(question, context_documents)
    
    # First try Ollama, then fallback to OpenAI if available
    ollama_response = query_ollama_llm(prompt)
    
    # Check if we got an error response from Ollama
    if "couldn't connect" in ollama_response or "error occurred" in ollama_response or "check that the Ollama service" in ollama_response:
        # Try OpenAI if Ollama failed and OpenAI is configured
        if OPENAI_API_KEY:
            result = query_openai_llm(prompt)
        else:
            # If both failed, return a specific error
            result = "No working LLM service is available. Please check that either Ollama is running or OpenAI API key is configured."
    else:
        result = ollama_response
    
    # Cache the result
    query_cache[cache_key] = result
    
    return result

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

# Async version of the LLM query function for background processing
async def get_answer_from_llm_async(question: str, context_documents: List[str]) -> str:
    """
    Async version of get_answer_from_llm
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_answer_from_llm, question, context_documents)

async def get_streaming_response(question: str, context_documents: List[str]) -> AsyncIterator[str]:
    """
    Stream tokens from the LLM as they're generated
    
    Args:
        question: User's question
        context_documents: List of relevant document chunks
        
    Yields:
        str: Individual tokens as they're generated
    """
    global llm_status
    
    # Update status
    llm_status["is_processing"] = True
    llm_status["last_query_time"] = time.time()
    llm_status["total_queries"] += 1
    
    # Try Ollama streaming first
    try:
        # Create prompt
        prompt = create_prompt_with_context(question, context_documents)
        
        # Set up the model to use
        model = "llama3.2:latest"
        llm_status["last_model_used"] = model
        
        # Check if model is loaded
        try:
            model_status_response = requests.get(f"http://localhost:11434/api/tags", timeout=2)
            
            if model_status_response.status_code == 200:
                model_list = model_status_response.json().get("models", [])
                model_loaded = any(m.get("name") == model for m in model_list)
                
                if not model_loaded:
                    yield f"The model '{model}' is not currently loaded in Ollama. Please run 'ollama pull {model}' first."
                    llm_status["is_processing"] = False
                    return
        except:
            # If we can't check, continue anyway
            print("Could not check model status, assuming it's loaded.")
            pass
            
        # Prepare the streaming request to Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024
            }
        }
          # Make streaming request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate", 
                json=payload, 
                timeout=60
            ) as response:
                if response.status == 200:
                    # Parse streaming response
                    buffer = ""
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line:
                            try:
                                json_line = json.loads(line)
                                if "response" in json_line:
                                    token = json_line["response"]
                                    buffer += token
                                    yield token
                            except json.JSONDecodeError:
                                continue
                                
                    # Update status after successful generation
                    llm_status["is_processing"] = False
                    llm_status["successful_queries"] += 1
                else:
                    # Fall back to OpenAI if Ollama fails
                    async for token in stream_from_openai(prompt):
                        yield token
        
    except Exception as e:
        # Attempt to use OpenAI streaming if available
        try:
            async for token in stream_from_openai(prompt):
                yield token
        except Exception as inner_e:
            # If everything fails, return an error message
            yield f"Error: Could not stream response from any available LLM service. {str(e)}"
            llm_status["is_processing"] = False

async def stream_from_openai(prompt: str) -> AsyncIterator[str]:
    """Stream tokens from OpenAI"""
    global llm_status
    
    try:
        # Check if OpenAI is properly configured
        if not OPENAI_API_KEY:
            yield "OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment variables."
            llm_status["is_processing"] = False
            return
            
        import openai
        openai.api_key = OPENAI_API_KEY
        
        # Create the streaming response
        stream = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for question-answering."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1,
            stream=True
        )
        
        # Process the streaming response
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
        # Update status after successful generation
        llm_status["is_processing"] = False
        llm_status["successful_queries"] += 1
        
    except ImportError:
        yield "OpenAI Python package is not installed. Please install it with 'pip install openai'."
        llm_status["is_processing"] = False
    except Exception as e:
        yield f"Error calling OpenAI API: {str(e)}"
        llm_status["is_processing"] = False