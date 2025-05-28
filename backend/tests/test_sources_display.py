#!/usr/bin/env python3
"""
Test script to verify that sources are properly displayed for both regular and decomposed queries.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_regular_query():
    """Test regular query and check sources display"""
    print("🔍 Testing Regular Query Sources...")
    
    payload = {
        "question": "What is machine learning?",
        "search_strategy": "semantic",
        "max_sources": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Regular query successful")
            print(f"   Sources count: {result.get('num_sources', 'N/A')}")
            sources = result.get('sources', [])
            print(f"   Sources array length: {len(sources)}")
            if sources:
                print(f"   Sample source filenames:")
                for i, source in enumerate(sources[:3]):
                    print(f"     {i+1}. {source.get('filename', 'Unknown')}")
            return result
        else:
            print(f"❌ Regular query failed: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"❌ Regular query error: {e}")
        return None

def test_decomposed_query():
    """Test decomposed query and check sources aggregation"""
    print("\n🧩 Testing Decomposed Query Sources...")
    
    payload = {
        "question": "What are the benefits and challenges of artificial intelligence in healthcare?",
        "search_strategy": "semantic", 
        "max_sources": 3,
        "use_decomposition": True,
        "max_sub_queries": 2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query/decomposed", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Decomposed query successful")
            print(f"   Is decomposed: {result.get('is_decomposed', False)}")
            print(f"   Total sources: {result.get('total_sources', 'N/A')}")
            
            sub_results = result.get('sub_results', [])
            print(f"   Sub-queries count: {len(sub_results)}")
            
            all_sources = []
            for i, sub_result in enumerate(sub_results):
                sources = sub_result.get('sources', [])
                print(f"   Sub-query {i+1} sources: {len(sources)}")
                all_sources.extend(sources)
            
            # Collect unique filenames
            unique_files = set()
            for source in all_sources:
                unique_files.add(source.get('filename', 'Unknown'))
            
            print(f"   Total unique source files: {len(unique_files)}")
            print(f"   Source filenames:")
            for filename in sorted(unique_files):
                print(f"     - {filename}")
                
            return result
        else:
            print(f"❌ Decomposed query failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error details: {error_detail}")
            except:
                print(f"   Response text: {response.text[:200]}...")
            return None
    except requests.RequestException as e:
        print(f"❌ Decomposed query error: {e}")
        return None

def main():
    print("🧪 Testing Sources Display for PFE System")
    print("=" * 50)
    
    # Test regular query
    regular_result = test_regular_query()
    
    # Test decomposed query  
    decomposed_result = test_decomposed_query()
    
    print("\n📊 Summary:")
    print("-" * 30)
    
    if regular_result:
        print(f"✅ Regular query: {regular_result.get('num_sources', 0)} sources")
    else:
        print("❌ Regular query: Failed")
        
    if decomposed_result:
        print(f"✅ Decomposed query: {decomposed_result.get('total_sources', 0)} total sources")
        if decomposed_result.get('is_decomposed'):
            print(f"   📝 Sub-queries: {len(decomposed_result.get('sub_results', []))}")
    else:
        print("❌ Decomposed query: Failed")
    
    print("\n💡 Frontend should now display:")
    print("   • Source Documents section with actual filenames")
    print("   • Performance Analytics showing correct source counts") 
    print("   • Aggregated sources from all sub-queries for decomposed queries")

if __name__ == "__main__":
    main()
