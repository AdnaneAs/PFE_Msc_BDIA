#!/usr/bin/env python3
"""
Test script to verify the Hugging Face optimizations are working correctly
"""

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_service import query_huggingface_llm, HUGGINGFACE_MODEL_CONFIGS

def test_huggingface_config():
    """Test that the HuggingFace model configurations are properly set up"""
    print("Testing Hugging Face Model Configurations:")
    print("-" * 50)
    
    # Test that configs exist
    print(f"Available model configs: {list(HUGGINGFACE_MODEL_CONFIGS.keys())}")
    
    # Test PleIAs specific config
    pleias_config = HUGGINGFACE_MODEL_CONFIGS.get("PleIAs/Pleias-RAG-1B")
    if pleias_config:
        print(f"PleIAs config: {pleias_config}")
        
        # Test prompt formatting
        test_prompt = "What is artificial intelligence?"
        formatted = pleias_config["prompt_template"].format(prompt=test_prompt)
        print(f"Formatted prompt example: {formatted}")
    
    # Test default config
    default_config = HUGGINGFACE_MODEL_CONFIGS.get("default")
    if default_config:
        print(f"Default config: {default_config}")
    
    print("✅ Configuration test completed")

def test_simple_query():
    """Test a simple query to verify the implementation works"""
    print("\nTesting Simple Query:")
    print("-" * 50)
    
    # Simple test prompt
    test_prompt = "Question: What is 2+2? Answer:"
    
    # Use local model with PleIAs configuration
    model_config = {
        'model': 'PleIAs/Pleias-RAG-1B',
        'provider': 'huggingface',
        'use_local': True
    }
    
    try:
        print("Attempting to query Hugging Face model...")
        print(f"Using model: {model_config['model']}")
        
        response, model_info = query_huggingface_llm(test_prompt, model_config)
        
        print(f"Model info: {model_info}")
        print(f"Response: {response}")
        
        # Check if response looks valid
        if response and len(response.strip()) > 10 and not any(bad in response for bad in ['__', '0/1/', '1/1/']):
            print("✅ Response looks valid")
        else:
            print("⚠️  Response may have quality issues")
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        print("This is expected if the model isn't downloaded yet")

if __name__ == "__main__":
    print("Hugging Face Optimization Test")
    print("=" * 50)
    
    test_huggingface_config()
    test_simple_query()
    
    print("\n" + "=" * 50)
    print("Test completed. If you see any errors about missing models,")
    print("they will be downloaded automatically on first use.")
