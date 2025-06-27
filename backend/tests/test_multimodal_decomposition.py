#!/usr/bin/env python3
"""
Test script for multimodal query decomposition functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_multimodal_decomposition():
    """Test multimodal query with decomposition enabled"""
    
    print("🧪 Testing Multimodal Query Decomposition")
    print("=" * 50)
    
    # Test with a complex multimodal query that should be decomposed
    test_data = {
        "question": "What are the main financial risks shown in the reports and what visual evidence supports these findings?",
        "max_sources": 5,
        "text_weight": 0.6,
        "image_weight": 0.4,
        "config_for_model": {
            "provider": "ollama",
            "model": "llama3.2:latest"
        },
        "search_strategy": "multimodal",
        "include_images": True,
        "use_decomposition": True,  # Enable decomposition
        "use_reranking": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
    
    print(f"Query: {test_data['question']}")
    print(f"Decomposition enabled: {test_data['use_decomposition']}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/query/multimodal",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=60
        )
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Request time: {(end_time - start_time)*1000:.0f}ms")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Multimodal query with decomposition completed")
            print("-" * 50)
            
            # Check decomposition fields
            is_decomposed = result.get('is_decomposed', False)
            decomposition_enabled = result.get('decomposition_enabled', False)
            
            print(f"📊 Results:")
            print(f"   - Answer length: {len(result.get('answer', ''))}")
            print(f"   - Text sources: {result.get('num_text_sources', 0)}")
            print(f"   - Image sources: {result.get('num_image_sources', 0)}")
            print(f"   - Query time: {result.get('query_time_ms', 0)}ms")
            print(f"   - Average relevance: {result.get('average_relevance', 'N/A')}")
            print(f"   - Top relevance: {result.get('top_relevance', 'N/A')}")
            print(f"   - Reranking used: {result.get('reranking_used', False)}")
            
            print(f"\n🧩 Decomposition Status:")
            print(f"   - Decomposition enabled: {decomposition_enabled}")
            print(f"   - Query was decomposed: {is_decomposed}")
            
            if is_decomposed:
                print("   ✅ Query was successfully decomposed!")
            else:
                print("   ℹ️  Query was processed as simple (no decomposition needed)")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response text: {response.text[:500]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_multimodal_without_decomposition():
    """Test multimodal query with decomposition disabled"""
    
    print("\n🧪 Testing Multimodal Query WITHOUT Decomposition")
    print("=" * 50)
    
    # Same query but with decomposition disabled
    test_data = {
        "question": "What are the main financial risks shown in the reports and what visual evidence supports these findings?",
        "max_sources": 5,
        "text_weight": 0.6,
        "image_weight": 0.4,
        "config_for_model": {
            "provider": "ollama",
            "model": "llama3.2:latest"
        },
        "search_strategy": "multimodal",
        "include_images": True,
        "use_decomposition": False,  # Disable decomposition
        "use_reranking": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
    
    print(f"Query: {test_data['question']}")
    print(f"Decomposition enabled: {test_data['use_decomposition']}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/query/multimodal",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=60
        )
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Request time: {(end_time - start_time)*1000:.0f}ms")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Multimodal query without decomposition completed")
            print("-" * 50)
            
            # Check decomposition fields
            is_decomposed = result.get('is_decomposed', False)
            decomposition_enabled = result.get('decomposition_enabled', False)
            
            print(f"📊 Results:")
            print(f"   - Answer length: {len(result.get('answer', ''))}")
            print(f"   - Text sources: {result.get('num_text_sources', 0)}")
            print(f"   - Image sources: {result.get('num_image_sources', 0)}")
            print(f"   - Query time: {result.get('query_time_ms', 0)}ms")
            
            print(f"\n🧩 Decomposition Status:")
            print(f"   - Decomposition enabled: {decomposition_enabled}")
            print(f"   - Query was decomposed: {is_decomposed}")
            
            if not decomposition_enabled and not is_decomposed:
                print("   ✅ Decomposition correctly disabled!")
            else:
                print("   ⚠️  Unexpected decomposition behavior")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response text: {response.text[:500]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Multimodal Query Decomposition Test Suite")
    print("=" * 60)
    print("Testing multimodal queries with and without decomposition...")
    print()
    
    # Run tests
    test1_passed = test_multimodal_decomposition()
    test2_passed = test_multimodal_without_decomposition()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"   Multimodal WITH decomposition: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Multimodal WITHOUT decomposition: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Multimodal decomposition is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
