#!/usr/bin/env python3
"""
Test script for multimodal endpoint to verify relevance scores are properly returned
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1/query"

def test_frontend_multimodal_endpoint():
    """Test the multimodal endpoint that frontend calls"""
    
    url = f"{BASE_URL}/multimodal"
    
    # Test payload - mimics what frontend sends
    payload = {
        "question": "What are the audit compliance requirements?",
        "config_for_model": None,
        "max_sources": 10,
        "text_weight": 0.7,
        "image_weight": 0.3,
        "use_decomposition": False,
        "use_reranking": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
    
    print("🧪 Testing Frontend Multimodal Endpoint")
    print("==================================================")
    print(f"URL: {url}")
    print(f"Query: {payload['question']}")
    print("==================================================")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract performance metrics
            avg_relevance = result.get('average_relevance')
            top_relevance = result.get('top_relevance')
            text_sources = result.get('text_sources', [])
            image_sources = result.get('image_sources', [])
            
            print(f"✅ Multimodal API Response successful")
            print(f"📊 Average Relevance: {avg_relevance}%")
            print(f"📊 Top Relevance: {top_relevance}%")
            print(f"📄 Text sources: {len(text_sources)}")
            print(f"🖼️ Image sources: {len(image_sources)}")
            print(f"🔮 Decomposed: {result.get('is_decomposed', False)}")
            
            # Check if this matches what we expect
            print(f"\n📋 Detailed Response:")
            print(f"- Model: {result.get('model', 'N/A')}")
            print(f"- Query time: {result.get('query_time_ms', 'N/A')}ms")
            print(f"- Retrieval time: {result.get('retrieval_time_ms', 'N/A')}ms")
            print(f"- LLM time: {result.get('llm_time_ms', 'N/A')}ms")
            print(f"- Reranking used: {result.get('reranking_used', False)}")
            
            # Check first few text sources for relevance scores
            if text_sources:
                print(f"\n📄 First 3 text sources:")
                for i, source in enumerate(text_sources[:3]):
                    relevance = source.get('relevance_score', 'N/A')
                    filename = source.get('original_filename', 'unknown')
                    print(f"   {i+1}. {filename} - {relevance}%")
            
            # Check image sources
            if image_sources:
                print(f"\n🖼️ First 3 image sources:")
                for i, source in enumerate(image_sources[:3]):
                    relevance = source.relevance_score if hasattr(source, 'relevance_score') else 'N/A'
                    filename = source.original_filename if hasattr(source, 'original_filename') else 'unknown'
                    print(f"   {i+1}. {filename} - {relevance}%")
            
            # Analysis
            print(f"\n🔧 Analysis:")
            if avg_relevance is not None:
                if avg_relevance >= 40:
                    print(f"✅ Good average relevance ({avg_relevance}%)")
                elif avg_relevance >= 5:
                    print(f"⚠️ Moderate average relevance ({avg_relevance}%)")
                else:
                    print(f"❌ Low average relevance ({avg_relevance}%) - needs improvement")
            else:
                print("❌ No relevance score available")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    test_frontend_multimodal_endpoint()
