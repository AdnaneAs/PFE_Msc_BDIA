#!/usr/bin/env python3
"""
Quick test to investigate source chunk structure
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def investigate_source_structure():
    """Investigate the actual structure of source chunks"""
    payload = {
        "question": "What is machine learning?",
        "search_strategy": "semantic",
        "max_sources": 1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            sources = result.get('sources', [])
            
            if sources:
                source = sources[0]
                print("📄 Source chunk structure:")
                print(json.dumps(source, indent=2))
            else:
                print("No sources found")
        else:
            print(f"Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    investigate_source_structure()
