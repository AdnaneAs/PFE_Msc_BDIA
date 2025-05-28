#!/usr/bin/env python3
"""
Simple test script for the decomposed query endpoint
"""

import requests
import json
import sys

def test_decomposed_endpoint():
    """Test the decomposed query endpoint"""
    url = "http://localhost:8000/api/query/decomposed"
    
    # Test data
    test_data = {
        "question": "What are the main financial risks mentioned in the audit report and how do they affect the company's performance?",
        "config_for_model": {
            "provider": "ollama",
            "model": "llama3.2:latest"
        },
        "search_strategy": "hybrid",
        "max_sources": 5,
        "use_decomposition": True
    }
    
    print("Testing decomposed query endpoint...")
    print(f"URL: {url}")
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    print("-" * 50)
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response received:")
            print(f"- Original query: {result.get('original_query', 'N/A')}")
            print(f"- Is decomposed: {result.get('is_decomposed', 'N/A')}")
            print(f"- Sub-queries count: {len(result.get('sub_queries', []))}")
            print(f"- Final answer length: {len(result.get('final_answer', ''))}")
            print(f"- Total time: {result.get('total_query_time_ms', 'N/A')}ms")
            
            if result.get('is_decomposed'):
                print("\nSub-queries:")
                for i, sub_query in enumerate(result.get('sub_queries', []), 1):
                    print(f"  {i}. {sub_query}")
                    
                print(f"\nDecomposition time: {result.get('decomposition_time_ms', 'N/A')}ms")
                print(f"Synthesis time: {result.get('synthesis_time_ms', 'N/A')}ms")
            
            return True
        else:
            print(f"ERROR: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_decomposed_endpoint()
    sys.exit(0 if success else 1)
