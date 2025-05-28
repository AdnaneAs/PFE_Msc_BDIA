#!/usr/bin/env python3
"""
Quick test to check if decomposed endpoint is accessible
"""

import requests
import json

def test_api_structure():
    """Test if the API endpoint accepts our request structure"""
    url = "http://localhost:8000/api/query/decomposed"
    
    # Simple test data
    test_data = {
        "question": "What is the revenue?",
        "config_for_model": {
            "provider": "ollama",
            "model": "llama3.2:latest"
        },
        "search_strategy": "hybrid",
        "max_sources": 3,
        "use_decomposition": True
    }
    
    print("Testing decomposed query API structure...")
    print(f"URL: {url}")
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=10  # Shorter timeout
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response structure:")
            print(f"- Original query: {result.get('original_query', 'N/A')}")
            print(f"- Is decomposed: {result.get('is_decomposed', 'N/A')}")
            print(f"- Average relevance: {result.get('average_relevance', 'N/A')}")
            print(f"- Top relevance: {result.get('top_relevance', 'N/A')}")
            print(f"- Total sources: {result.get('total_sources', 'N/A')}")
            return True
        else:
            print(f"HTTP Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out - API is processing but taking time")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    test_api_structure()
