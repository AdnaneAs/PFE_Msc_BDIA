#!/usr/bin/env python3
"""
Test script for BGE Reranker integration
"""
import sys
import os
import requests
import json

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_bge_reranker_api():
    """Test the BGE reranker through the API endpoints"""
    
    base_url = "http://localhost:8000/api/query"
    
    # Test query without reranking
    print("🔍 Testing query WITHOUT BGE reranking...")
    query_without_reranking = {
        "question": "What is machine learning?",
        "max_sources": 5,
        "search_strategy": "semantic",
        "use_reranking": False
    }
    
    try:
        response = requests.post(f"{base_url}/", json=query_without_reranking)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query without reranking successful")
            print(f"   - Sources found: {result.get('num_sources', 0)}")
            print(f"   - Search strategy: {result.get('search_strategy', 'N/A')}")
            print(f"   - Reranking used: {result.get('reranking_used', False)}")
        else:
            print(f"❌ Query without reranking failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error in query without reranking: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test query with BGE reranking
    print("🚀 Testing query WITH BGE reranking...")
    query_with_reranking = {
        "question": "What is machine learning?",
        "max_sources": 5,
        "search_strategy": "semantic",
        "use_reranking": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
    
    try:
        response = requests.post(f"{base_url}/", json=query_with_reranking)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query with reranking successful")
            print(f"   - Sources found: {result.get('num_sources', 0)}")
            print(f"   - Search strategy: {result.get('search_strategy', 'N/A')}")
            print(f"   - Reranking used: {result.get('reranking_used', False)}")
            print(f"   - Reranker model: {result.get('reranker_model', 'N/A')}")
            
            # Show source relevance if available
            sources = result.get('sources', [])
            if sources:
                print(f"   - Top source relevance: {sources[0].get('relevance_score', 'N/A')}")
        else:
            print(f"❌ Query with reranking failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error in query with reranking: {e}")

    print("\n" + "="*60 + "\n")
    
    # Test BGE reranker configuration endpoint
    print("⚙️  Testing BGE reranker configuration endpoint...")
    try:
        response = requests.get(f"{base_url}/reranker/models")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ BGE reranker config endpoint successful")
            print(f"   - Available models: {len(result.get('available_models', []))}")
            print(f"   - Default model: {result.get('default_model', 'N/A')}")
            print(f"   - Reranker available: {result.get('reranker_available', False)}")
        else:
            print(f"❌ BGE reranker config failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error in BGE reranker config: {e}")

def test_bge_reranker_direct():
    """Test the BGE reranker service directly"""
    
    print("🧪 Testing BGE Reranker service directly...")
    
    try:
        from app.services.rerank_service import BGEReranker, is_reranker_available
          # Check if reranker is available
        print(f"   - Reranker available: {is_reranker_available()}")
        
        if is_reranker_available():
            # Test reranker initialization
            reranker = BGEReranker("BAAI/bge-reranker-base")
            print(f"   - Reranker initialized: {reranker.model_name}")
            print(f"   - Device: {reranker.device}")
            
            # Test with sample query and documents
            query = "What is artificial intelligence?"
            documents = [
                "Artificial intelligence (AI) is the simulation of human intelligence in machines.",
                "Machine learning is a subset of artificial intelligence.",
                "The weather is nice today.",
                "Deep learning uses neural networks with multiple layers."
            ]
            
            reranked_results = reranker.rerank(query, documents)
            print(f"   - Reranking successful, got {len(reranked_results)} results")
            if reranked_results:
                scores = [score for _, score in reranked_results]
                print(f"   - Top score: {max(scores):.3f}")
                print(f"   - Score range: {min(scores):.3f} - {max(scores):.3f}")
                print(f"   - Top document: {reranked_results[0][0][:50]}...")
        
    except Exception as e:
        print(f"❌ Error in direct BGE reranker test: {e}")

if __name__ == "__main__":
    print("🎯 BGE Reranker Integration Test")
    print("=" * 60)
    
    # Test API endpoints
    test_bge_reranker_api()
    
    print("\n" + "=" * 60 + "\n")
    
    # Test direct service
    test_bge_reranker_direct()
    
    print("\n🏁 BGE Reranker test completed!")
