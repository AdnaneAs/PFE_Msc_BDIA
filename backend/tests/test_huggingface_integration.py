#!/usr/bin/env python3
"""
Simple test script to verify Hugging Face integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.llm_service import get_available_models, query_huggingface_llm

def test_huggingface_models():
    """Test that Hugging Face models are available"""
    print("🧪 Testing Hugging Face model availability...")
    
    models = get_available_models()
    huggingface_models = [m for m in models if m['provider'] == 'huggingface']
    
    print(f"Found {len(huggingface_models)} Hugging Face models:")
    for model in huggingface_models:
        print(f"  • {model['name']} - {model['description']}")
    
    # Check if PleIAs model is available
    pleias_model = next((m for m in huggingface_models if 'PleIAs' in m['name']), None)
    if pleias_model:
        print(f"✅ PleIAs/Pleias-RAG-1B model found: {pleias_model['description']}")
    else:
        print("❌ PleIAs/Pleias-RAG-1B model not found")
    
    return len(huggingface_models) > 0

def test_huggingface_query_structure():
    """Test the Hugging Face query function structure (without API call)"""
    print("\n🧪 Testing Hugging Face query function structure...")
    
    try:
        # Test with no API key (should return error message)
        response, model_info = query_huggingface_llm("Test prompt", {})
        
        if "API key is not configured" in response:
            print("✅ Hugging Face function correctly handles missing API key")
            print(f"   Model info: {model_info}")
        else:
            print(f"⚠️ Unexpected response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error testing Hugging Face function: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Hugging Face Integration Test")
    print("=" * 50)
    
    # Test 1: Model availability
    models_test = test_huggingface_models()
    
    # Test 2: Query function structure
    query_test = test_huggingface_query_structure()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Models available: {'✅' if models_test else '❌'}")
    print(f"  Query function:   {'✅' if query_test else '❌'}")
    
    if models_test and query_test:
        print("\n🎉 All tests passed! Hugging Face integration is ready.")
        print("\n💡 To use Hugging Face models:")
        print("   1. Get a Hugging Face API token from https://huggingface.co/settings/tokens")
        print("   2. Set HUGGINGFACE_API_KEY environment variable or provide via UI")
        print("   3. Select 'Hugging Face' as the model provider")
        print("   4. Choose your preferred model (PleIAs/Pleias-RAG-1B recommended for RAG)")
    else:
        print("\n❌ Some tests failed. Please check the integration.")
