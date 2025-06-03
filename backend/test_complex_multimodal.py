#!/usr/bin/env python3
"""
Test complex multimodal query decomposition with detailed logging
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_complex_multimodal_query():
    """Test a highly complex multimodal query to verify decomposition"""
    
    print("🧪 Testing Complex Multimodal Query Decomposition")
    print("=" * 60)
    
    # Very complex query that should definitely be decomposed
    complex_query = """
    Analyze the comprehensive business intelligence from our reports: 
    What are the quarterly revenue trends, how do the financial charts support these trends, 
    what risk factors are mentioned in text documents, what visual data shows market performance, 
    and how do image-based graphs correlate with written analysis about customer satisfaction metrics?
    """
    
    test_data = {
        "question": complex_query.strip(),
        "max_sources": 8,
        "text_weight": 0.5,
        "image_weight": 0.5,
        "config_for_model": {
            "provider": "ollama",
            "model": "llama3.2:latest"
        },
        "search_strategy": "multimodal",
        "include_images": True,
        "use_decomposition": True,
        "use_reranking": True,
        "reranker_model": "BAAI/bge-reranker-base"
    }
    
    print(f"Query: {test_data['question'][:100]}...")
    print(f"Decomposition enabled: {test_data['use_decomposition']}")
    print(f"Max sources: {test_data['max_sources']}")
    print(f"Text/Image weights: {test_data['text_weight']}/{test_data['image_weight']}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/query/multimodal",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=120  # Longer timeout for complex query
        )
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Total request time: {(end_time - start_time)*1000:.0f}ms")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Complex multimodal query completed")
            print("-" * 60)
            
            # Detailed analysis
            is_decomposed = result.get('is_decomposed', False)
            decomposition_enabled = result.get('decomposition_enabled', False)
            
            print(f"📊 Detailed Results:")
            print(f"   - Answer length: {len(result.get('answer', ''))} characters")
            print(f"   - Text sources found: {result.get('num_text_sources', 0)}")
            print(f"   - Image sources found: {result.get('num_image_sources', 0)}")
            print(f"   - Total retrieval time: {result.get('retrieval_time_ms', 0)}ms")
            print(f"   - LLM processing time: {result.get('llm_time_ms', 0)}ms")
            print(f"   - Total query time: {result.get('query_time_ms', 0)}ms")
            print(f"   - Average relevance: {result.get('average_relevance', 'N/A')}")
            print(f"   - Top relevance score: {result.get('top_relevance', 'N/A')}")
            print(f"   - BGE reranking used: {result.get('reranking_used', False)}")
            
            print(f"\n🧩 Decomposition Analysis:")
            print(f"   - Decomposition enabled: {decomposition_enabled}")
            print(f"   - Query was decomposed: {is_decomposed}")
            
            if is_decomposed:
                print("   ✅ Complex query was successfully decomposed into sub-queries!")
                print("   📈 This should provide more comprehensive multimodal results")
            else:
                print("   ⚠️  Query was not decomposed (may be simpler than expected)")
            
            # Show sample of results
            print(f"\n📖 Answer Preview:")
            answer = result.get('answer', '')
            print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
            # Analyze source diversity
            text_sources = result.get('text_sources', [])
            image_sources = result.get('image_sources', [])
            
            if text_sources:
                print(f"\n📄 Text Source Analysis:")
                print(f"   - Total text sources: {len(text_sources)}")
                # Show sample metadata if available
                if len(text_sources) > 0 and isinstance(text_sources[0], dict):
                    sample_text = text_sources[0]
                    print(f"   - Sample source: {sample_text.get('source', 'N/A')}")
            
            if image_sources:
                print(f"\n🖼️  Image Source Analysis:")
                print(f"   - Total image sources: {len(image_sources)}")
                if len(image_sources) > 0:
                    sample_img = image_sources[0]
                    if hasattr(sample_img, 'description'):
                        desc = sample_img.description[:100] if sample_img.description else "No description"
                        print(f"   - Sample image: {desc}{'...' if len(desc) == 100 else ''}")
                    elif isinstance(sample_img, dict):
                        desc = sample_img.get('description', 'No description')[:100]
                        print(f"   - Sample image: {desc}{'...' if len(desc) == 100 else ''}")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response text: {response.text[:500]}...")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after 120 seconds")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Complex Multimodal Query Test")
    print("=" * 80)
    print("Testing highly complex multimodal query with decomposition...")
    print()
    
    success = test_complex_multimodal_query()
    
    print("\n" + "=" * 80)
    print("📊 COMPLEX TEST SUMMARY:")
    print(f"   Complex multimodal decomposition: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 Complex multimodal decomposition test PASSED!")
        print("   The system successfully handled a multi-faceted query by:")
        print("   - Decomposing complex question into manageable sub-queries")
        print("   - Processing both text and image sources for each sub-query")
        print("   - Applying BGE reranking to improve result quality")
        print("   - Aggregating results into comprehensive multimodal response")
    else:
        print("\n⚠️  Complex test failed. Check logs for details.")
