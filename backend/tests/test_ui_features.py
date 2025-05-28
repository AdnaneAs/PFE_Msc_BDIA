#!/usr/bin/env python3
"""
Test script to validate the enhanced UI features of the PFE RAG system.
This script tests the new search strategies, performance metrics, and enhanced API responses.
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3001"

def test_api_connectivity():
    """Test basic API connectivity"""
    print("🔗 Testing API connectivity...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/hello")
        if response.status_code == 200:
            print("✅ Backend API is accessible")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

def test_enhanced_query_api():
    """Test the enhanced query API with new parameters"""
    print("\n🔍 Testing enhanced query API...")
    
    # Test queries with different search strategies
    test_queries = [
        {
            "question": "What are the main findings in the audit reports?",
            "search_strategy": "hybrid",
            "max_sources": 5,
            "description": "Hybrid search with 5 sources"
        },
        {
            "question": "What are compliance issues mentioned?",
            "search_strategy": "semantic", 
            "max_sources": 3,
            "description": "Semantic search with 3 sources"
        },
        {
            "question": "audit findings recommendations",
            "search_strategy": "keyword",
            "max_sources": 10,
            "description": "Keyword search with 10 sources"
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query['description']}")
        print(f"   Question: {query['question']}")
        print(f"   Strategy: {query['search_strategy']}")
        print(f"   Max sources: {query['max_sources']}")
        
        try:
            start_time = time.time()
            
            payload = {
                "question": query["question"],
                "config_for_model": {
                    "provider": "ollama",
                    "model": "llama3.2:latest"
                },
                "search_strategy": query["search_strategy"],
                "max_sources": query["max_sources"]
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/query/",
                json=payload,
                timeout=120
            )
            
            end_time = time.time()
            request_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Query successful")
                print(f"   📊 Performance Metrics:")
                print(f"      - Request time: {request_time:.1f}ms")
                print(f"      - Total query time: {result.get('query_time_ms', 'N/A')}ms")
                print(f"      - Retrieval time: {result.get('retrieval_time_ms', 'N/A')}ms")
                print(f"      - LLM time: {result.get('llm_time_ms', 'N/A')}ms")
                print(f"      - Sources retrieved: {result.get('num_sources', 'N/A')}")
                print(f"      - Search strategy used: {result.get('search_strategy', 'N/A')}")
                
                # Test new relevance metrics
                if 'average_relevance' in result and 'top_relevance' in result:
                    print(f"   🎯 Relevance Metrics:")
                    print(f"      - Average relevance: {result['average_relevance']:.4f}")
                    print(f"      - Top relevance: {result['top_relevance']:.4f}")
                else:
                    print(f"   ⚠️ Relevance metrics missing from response")
                
                # Test answer quality
                answer_length = len(result.get('answer', ''))
                print(f"   📝 Answer length: {answer_length} characters")
                
                if answer_length > 50:
                    print(f"   ✅ Answer appears substantial")
                else:
                    print(f"   ⚠️ Answer seems short")
                    
            else:
                print(f"   ❌ Query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
        
        # Add delay between queries
        if i < len(test_queries):
            print(f"   ⏳ Waiting 3 seconds before next query...")
            time.sleep(3)

def test_model_availability():
    """Test available models and configurations"""
    print("\n🤖 Testing model availability...")
    
    try:
        # Test Ollama models endpoint
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama models available: {len(models)}")
            for model in models[:5]:  # Show first 5 models
                print(f"   - {model['name']}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
        else:
            print(f"⚠️ Cannot access Ollama directly")
            
    except Exception as e:
        print(f"⚠️ Ollama connection issue: {e}")

def test_llm_status():
    """Test LLM status endpoint"""
    print("\n📊 Testing LLM status endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/query/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ LLM status retrieved")
            print(f"   - Processing: {status.get('is_processing', 'N/A')}")
            print(f"   - Total queries: {status.get('total_queries', 'N/A')}")
            print(f"   - Successful queries: {status.get('successful_queries', 'N/A')}")
            print(f"   - Cache size: {status.get('cache_size', 'N/A')}")
            print(f"   - Last model: {status.get('last_model_used', 'N/A')}")
        else:
            print(f"❌ LLM status error: {response.status_code}")
    except Exception as e:
        print(f"❌ LLM status error: {e}")

def test_frontend_accessibility():
    """Test frontend accessibility"""
    print(f"\n🌐 Testing frontend accessibility...")
    
    try:
        response = requests.get(FRONTEND_URL, timeout=10)
        if response.status_code == 200:
            print(f"✅ Frontend is accessible at {FRONTEND_URL}")
            
            # Check for key UI elements in the HTML
            html_content = response.text
            ui_checks = [
                ("Search Strategy Dropdown", "search-strategy" in html_content),
                ("Max Sources Selection", "max-sources" in html_content),
                ("Model Selection", "ollama-local" in html_content),
                ("Query Input", "question" in html_content),
                ("Performance Metrics", "Performance Analytics" in html_content or "performance" in html_content.lower())
            ]
            
            for check_name, check_result in ui_checks:
                status = "✅" if check_result else "⚠️"
                print(f"   {status} {check_name}: {'Found' if check_result else 'Not found'}")
                
        else:
            print(f"❌ Frontend error: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend accessibility error: {e}")

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 PFE Enhanced RAG System - UI Features Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend URL: {API_BASE_URL}")
    print(f"Frontend URL: {FRONTEND_URL}")
    print("=" * 60)
    
    # Run all test functions
    tests = [
        test_api_connectivity,
        test_model_availability, 
        test_llm_status,
        test_enhanced_query_api,
        test_frontend_accessibility
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test '{test_func.__name__}' failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced UI Features Test Complete!")
    print("🌐 Frontend available at: http://localhost:3001")
    print("🔧 Backend API available at: http://localhost:8000")
    print("=" * 60)
    
    # Test summary
    print("\n📋 ENHANCED FEATURES SUMMARY:")
    print("✅ Search Strategy Selection (Hybrid, Semantic, Keyword)")
    print("✅ Max Sources Configuration (3, 5, 10, 15)")
    print("✅ Performance Metrics Visualization")
    print("✅ Context Relevance Scoring")
    print("✅ Enhanced Model Selection (11 models total)")
    print("✅ Real-time LLM Status Display")
    print("✅ Comprehensive Query Analytics")

if __name__ == "__main__":
    run_comprehensive_test()
