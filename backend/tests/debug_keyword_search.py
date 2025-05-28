#!/usr/bin/env python3
"""
Debug script to check keyword search functionality and fix the issue
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_keyword_search_fix():
    """Test and fix keyword search functionality"""
    print("🔧 Testing and fixing keyword search...")
    
    # Test if we have documents in the database
    print("\n1. Checking documents in database...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents")
        if response.status_code == 200:
            docs = response.json().get('documents', [])
            print(f"   ✅ Found {len(docs)} documents in database")
            for i, doc in enumerate(docs[:3]):
                print(f"      {i+1}. {doc.get('filename', 'unknown')} - {doc.get('status', 'unknown')}")
        else:
            print(f"   ❌ Error fetching documents: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test simple queries to understand the issue
    print("\n2. Testing simple semantic query...")
    try:
        payload = {
            "question": "audit",
            "config_for_model": {
                "provider": "ollama",
                "model": "llama3.2:latest"
            },
            "search_strategy": "semantic",
            "max_sources": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/api/query/", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Semantic search found {result.get('num_sources', 0)} sources")
        else:
            print(f"   ❌ Semantic search failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test keyword search with simpler query
    print("\n3. Testing keyword search with simple terms...")
    simple_keywords = ["audit", "report", "finding", "compliance"]
    
    for keyword in simple_keywords:
        print(f"\n   Testing keyword: '{keyword}'")
        try:
            payload = {
                "question": keyword,
                "config_for_model": {
                    "provider": "ollama", 
                    "model": "llama3.2:latest"
                },
                "search_strategy": "keyword",
                "max_sources": 5
            }
            
            response = requests.post(f"{API_BASE_URL}/api/query/", json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                sources = result.get('num_sources', 0)
                strategy = result.get('search_strategy', 'unknown')
                print(f"      ✅ Found {sources} sources with strategy '{strategy}'")
                
                if sources == 0:
                    print(f"      ⚠️ No sources found - keyword search may need adjustment")
                    
            else:
                print(f"      ❌ Failed: {response.status_code}")
                print(f"      Error: {response.text}")
        except Exception as e:
            print(f"      ❌ Error: {e}")

def check_vector_db_directly():
    """Check vector database directly"""
    print("\n4. Direct vector database check...")
    
    try:
        # This will help us understand what's in the vector DB
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
        
        from services.vector_db_service import get_collection
        
        collection = get_collection()
        count = collection.count()
        print(f"   📊 Vector database has {count} document chunks")
        
        if count > 0:
            # Get a sample of documents
            sample = collection.get(limit=3)
            docs = sample.get('documents', [])
            metadatas = sample.get('metadatas', [])
            
            print(f"   📝 Sample documents:")
            for i, (doc, meta) in enumerate(zip(docs, metadatas)):
                filename = meta.get('filename', 'unknown')
                print(f"      {i+1}. {filename}: {doc[:100]}...")
                
    except Exception as e:
        print(f"   ❌ Error accessing vector DB: {e}")

if __name__ == "__main__":
    print("🔍 Keyword Search Debug & Fix")
    print("=" * 50)
    
    test_keyword_search_fix()
    check_vector_db_directly()
    
    print("\n" + "=" * 50)
    print("✅ Debug complete!")
