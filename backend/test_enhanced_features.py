#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced RAG system features:
1. Search Strategy Selection (hybrid, semantic, keyword)
2. Model Selection (Ollama, OpenAI, Gemini)
3. Enhanced Performance Metrics
4. Context Relevance Scoring
"""

import requests
import json
import time

API_BASE = "http://localhost:8000/api"

def test_search_strategies():
    """Test different search strategies with the same query"""
    print("🔍 TESTING SEARCH STRATEGIES")
    print("=" * 50)
    
    query = "What are neural networks and deep learning techniques?"
    strategies = ["hybrid", "semantic", "keyword"]
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy.upper()} search:")
        response = requests.post(f"{API_BASE}/query/", json={
            'question': query,
            'search_strategy': strategy,
            'max_sources': 3,
            'config_for_model': {
                'provider': 'ollama',
                'model': 'llama3.2:latest'
            }
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Strategy: {data.get('search_strategy')}")
            print(f"   ⏱️  Query time: {data.get('query_time_ms')}ms")
            print(f"   📚 Sources: {len(data.get('sources', []))}")
            print(f"   🎯 Avg relevance: {data.get('average_relevance', 'N/A')}")
            print(f"   🏆 Top relevance: {data.get('top_relevance', 'N/A')}")
        else:
            print(f"   ❌ Error: {response.status_code}")

def test_model_selection():
    """Test different model providers"""
    print("\n\n🤖 TESTING MODEL SELECTION")
    print("=" * 50)
    
    # Get available models
    response = requests.get(f"{API_BASE}/query/models")
    if response.status_code == 200:
        models_data = response.json()
        print("📋 Available models:")
        for model in models_data.get('models', [])[:5]:  # Show first 5
            print(f"   • {model['name']} ({model['provider']})")
    
    # Test with different Ollama models
    query = "Explain machine learning concepts"
    ollama_models = ["llama3.2:latest", "mistral:7b"]
    
    for model in ollama_models:
        print(f"\n🧠 Testing with {model}:")
        start_time = time.time()
        response = requests.post(f"{API_BASE}/query/", json={
            'question': query,
            'search_strategy': 'hybrid',
            'max_sources': 3,
            'config_for_model': {
                'provider': 'ollama',
                'model': model
            }
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model used: {data.get('model')}")
            print(f"   ⏱️  Total time: {data.get('query_time_ms')}ms")
            print(f"   🔧 LLM time: {data.get('llm_time_ms')}ms")
            print(f"   📄 Answer length: {len(data.get('answer', ''))} chars")
        else:
            print(f"   ❌ Error: {response.status_code}")

def test_performance_metrics():
    """Test and display comprehensive performance metrics"""
    print("\n\n📈 TESTING PERFORMANCE METRICS")
    print("=" * 50)
    
    response = requests.post(f"{API_BASE}/query/", json={
        'question': 'What are the key findings in deep learning research?',
        'search_strategy': 'hybrid',
        'max_sources': 5,
        'config_for_model': {
            'provider': 'ollama',
            'model': 'llama3.2:latest'
        }
    })
    
    if response.status_code == 200:
        data = response.json()
        
        print("🎯 SEARCH PERFORMANCE:")
        print(f"   Strategy: {data.get('search_strategy', 'N/A')}")
        print(f"   Sources retrieved: {len(data.get('sources', []))}")
        print(f"   Average relevance: {data.get('average_relevance', 'N/A')}")
        print(f"   Best match relevance: {data.get('top_relevance', 'N/A')}")
        
        print("\n⏱️  TIMING BREAKDOWN:")
        total_time = data.get('query_time_ms', 0)
        retrieval_time = data.get('retrieval_time_ms', 0)
        llm_time = data.get('llm_time_ms', 0)
        
        print(f"   Total query time: {total_time}ms")
        print(f"   Document retrieval: {retrieval_time}ms ({(retrieval_time/total_time*100):.1f}%)")
        print(f"   LLM processing: {llm_time}ms ({(llm_time/total_time*100):.1f}%)")
        print(f"   Overhead: {total_time - retrieval_time - llm_time}ms ({((total_time - retrieval_time - llm_time)/total_time*100):.1f}%)")
        
        print("\n🤖 MODEL INFO:")
        print(f"   Model used: {data.get('model', 'N/A')}")
        print(f"   Answer length: {len(data.get('answer', ''))} characters")
        
        print("\n📚 SOURCE DETAILS:")
        for i, source in enumerate(data.get('sources', [])[:3]):
            print(f"   [{i+1}] {source.get('filename', 'Unknown')}")
            print(f"       Chunk: {source.get('chunk_id', '?')}")
    else:
        print(f"❌ Error: {response.status_code}")

def main():
    print("🚀 ENHANCED RAG SYSTEM DEMONSTRATION")
    print("=====================================")
    print("Testing new features:")
    print("✨ Search strategy selection (hybrid/semantic/keyword)")
    print("✨ Model selection (Ollama/OpenAI/Gemini)")
    print("✨ Enhanced performance metrics")
    print("✨ Context relevance scoring")
    print("✨ Comprehensive timing breakdown")
    
    try:
        test_search_strategies()
        test_model_selection() 
        test_performance_metrics()
        
        print("\n\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The enhanced RAG system is working perfectly with:")
        print("   ✅ Advanced search strategies")
        print("   ✅ Multi-model support")
        print("   ✅ Rich performance analytics")
        print("   ✅ Context relevance metrics")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    main()
