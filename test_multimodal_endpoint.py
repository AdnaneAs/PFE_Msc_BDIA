#!/usr/bin/env python3
"""
Test script to verify multimodal endpoint functionality
"""
import requests
import json

def test_multimodal_endpoint():
    """Test the multimodal query endpoint"""
    
    # Test data
    test_payload = {
        "question": "What is shown in the documents?",
        "max_sources": 3,
        "text_weight": 0.7,
        "image_weight": 0.3,
        "config_for_model": {},
        "search_strategy": "multimodal",
        "include_images": True,
        "use_reranking": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
    
    print("Testing multimodal endpoint...")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        # Test endpoint availability
        response = requests.post(
            "http://localhost:8000/api/v1/query/multimodal",
            json=test_payload,
            timeout=30
        )
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multimodal endpoint is working!")
            print(f"Answer length: {len(result.get('answer', ''))}")
            print(f"Text sources: {result.get('num_text_sources', 0)}")
            print(f"Image sources: {result.get('num_image_sources', 0)}")
            print(f"Reranking used: {result.get('reranking_used', False)}")
            print(f"Query time: {result.get('query_time_ms', 0)}ms")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_hello_endpoint():
    """Test the hello endpoint to verify server is running"""
    try:
        response = requests.get("http://localhost:8000/api/hello", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except:
        print("❌ Backend server is not responding")
        return False

if __name__ == "__main__":
    print("=== Multimodal Endpoint Test ===")
    
    # First check if server is running
    if test_hello_endpoint():
        print()
        test_multimodal_endpoint()
    else:
        print("\nPlease start the backend server first:")
        print("cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
