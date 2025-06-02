#!/usr/bin/env python3
"""
Test script to verify settings service improvements
"""
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.settings_service import (
    get_ollama_models,
    validate_model_for_provider,
    get_available_models_by_provider,
    update_model_with_validation,
    settings_manager
)

async def test_settings_improvements():
    """Test the improved settings service functionality"""
    
    print("🔧 Testing Settings Service Improvements")
    print("=" * 50)
    
    # Test 1: Get Ollama models
    print("\n1. Testing Ollama model detection:")
    ollama_models = get_ollama_models()
    if ollama_models:
        print(f"   ✅ Found {len(ollama_models)} Ollama models: {ollama_models}")
    else:
        print("   ⚠️  No Ollama models found (Ollama may not be running)")
    
    # Test 2: Test model validation
    print("\n2. Testing model validation:")
    test_cases = [
        ("ollama", "llama3.2:latest", True),
        ("ollama", "invalid-model", False),
        ("openai", "gpt-4o", True),
        ("openai", "llama3.2:latest", False),  # Wrong provider
        ("gemini", "gemini-2.0-flash", True),
        ("gemini", "gpt-4", False),  # Wrong provider
    ]
    
    for provider, model, expected in test_cases:
        result = validate_model_for_provider(provider, model)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {provider}/{model} -> {result} (expected: {expected})")
    
    # Test 3: Get available models by provider
    print("\n3. Testing available models by provider:")
    models_by_provider = get_available_models_by_provider()
    for provider, models in models_by_provider.items():
        count = len(models) if models else 0
        print(f"   📦 {provider}: {count} models")
        if provider == "ollama" and models:
            print(f"      Models: {models[:3]}{'...' if len(models) > 3 else ''}")
    
    # Test 4: Test settings loading and validation
    print("\n4. Testing settings loading with validation:")
    settings = settings_manager.load_settings()
    current_provider = settings.get("llm_provider", "unknown")
    current_model = settings.get("llm_model", "unknown")
    is_valid = validate_model_for_provider(current_provider, current_model)
    status = "✅" if is_valid else "❌"
    print(f"   {status} Current: {current_provider}/{current_model} (valid: {is_valid})")
    
    # Test 5: Test model update with validation
    print("\n5. Testing model update with validation:")
    # Try to update with a valid combination
    if ollama_models:
        test_model = ollama_models[0]
        success = update_model_with_validation("ollama", test_model)
        status = "✅" if success else "❌"
        print(f"   {status} Update to ollama/{test_model}: {success}")
    
    # Try to update with invalid combination
    success = update_model_with_validation("gemini", "llama3.2:latest")
    status = "✅" if not success else "❌"  # Should fail
    print(f"   {status} Invalid update gemini/llama3.2:latest: {not success} (correctly rejected)")
    
    print("\n" + "=" * 50)
    print("🎉 Settings service test completed!")

if __name__ == "__main__":
    asyncio.run(test_settings_improvements())
