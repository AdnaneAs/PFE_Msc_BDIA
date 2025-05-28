#!/usr/bin/env python3
"""
Simple API Test - Fix query parameter issue
"""

import requests
import json

def test_query_with_correct_format():
    """Test query with the correct parameter name"""
    print("Testing query with 'question' parameter...")
    
    try:
        # Use 'question' instead of 'query'
        query_data = {
            "question": "What types of documents are available in the system?",
            "config_for_model": {
                "max_tokens": 100,
                "model": "llama3.2:3b"
            }
        }
        
        response = requests.post(
            "http://localhost:8000/api/query/",
            json=query_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"Answer length: {len(result.get('answer', ''))}")
            return True
        else:
            print(f"❌ Query failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_documents_endpoint():
    """Test documents endpoint"""
    print("\nTesting documents endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/api/documents", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Documents retrieved: {len(result) if isinstance(result, list) else 'Object type'}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_query_with_correct_format()
    test_documents_endpoint()
