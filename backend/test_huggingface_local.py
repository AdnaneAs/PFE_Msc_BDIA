"""
Test script for Hugging Face local model integration
"""
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.llm_service import query_huggingface_llm

def test_huggingface_local():
    """Test Hugging Face local model without API key"""
    print("Testing Hugging Face local model...")
    
    # Test prompt
    prompt = """Answer the following question based on the context:

Question: What is machine learning?

Context: Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data.

Answer:"""
    
    # Model config for local usage (no API key)
    model_config = {
        'model': 'PleIAs/Pleias-RAG-1B',
        'use_local': True
    }
    
    try:
        response, model_info = query_huggingface_llm(prompt, model_config)
        print(f"✅ Response received: {response[:200]}...")
        print(f"✅ Model info: {model_info}")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_huggingface_local()
    if success:
        print("\n🎉 Hugging Face local model test passed!")
    else:
        print("\n❌ Hugging Face local model test failed!")
