#!/usr/bin/env python3
"""
Test the actual multimodal endpoint that frontend uses
"""
import requests
import json

def test_frontend_multimodal_endpoint():
    """Test the exact endpoint the frontend calls"""
    
    url = "http://localhost:8000/api/v1/query/multimodal"
    
    test_data = {
        "question": "What are the audit compliance requirements?",
        "model": "llama3.2:latest",
        "use_reranking": False,
        "search_strategy": "multimodal",
        "text_weight": 0.7,
        "image_weight": 0.3
    }
    
    print("🧪 Testing Frontend Multimodal Endpoint")
    print("=" * 50)
    print(f"URL: {url}")
    print(f"Query: {test_data['question']}")
    print("=" * 50)
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check relevance scores            avg_relevance = result.get('average_relevance')
            top_relevance = result.get('top_relevance')
            sources = result.get('sources', [])
            text_sources = result.get('text_sources', [])
            image_sources = result.get('image_sources', [])
            
            print(f"✅ Multimodal API Response successful")
            print(f"📊 Average Relevance: {avg_relevance}%")
            print(f"📊 Top Relevance: {top_relevance}%")
            print(f"📚 Total sources: {len(sources)}")
            print(f"📄 Text sources: {len(text_sources)}")
            print(f"🖼️ Image sources: {len(image_sources)}")
            print(f"🔮 Multimodal flag: {result.get('multimodal', False)}")
            
            # Check if this matches what we expect
            print(f"\n📋 Detailed Response:")
            print(f"- Model: {result.get('model', 'N/A')}")
            print(f"- Query time: {result.get('query_time_ms', 'N/A')}ms")
            print(f"- Search strategy: {result.get('search_strategy', 'N/A')}")
            print(f"- Text weight: {result.get('text_weight', 'N/A')}")
            print(f"- Image weight: {result.get('image_weight', 'N/A')}")
            
            # Check first few sources for relevance scores
            if sources:
                print(f"\n📄 First 3 sources:")
                for i, source in enumerate(sources[:3]):
                    chunks = source.get('chunks', [])
                    for j, chunk in enumerate(chunks[:1]):  # Just first chunk
                        metadata = chunk.get('metadata', {})
                        relevance = metadata.get('relevance_score', 'N/A')
                        filename = metadata.get('original_filename', 'unknown')
                        print(f"   {i+1}.{j+1} {filename} - {relevance}%")
            
            # Diagnosis
            print(f"\n🔧 Analysis:")
            if avg_relevance and avg_relevance > 40:
                print(f"✅ Average relevance ({avg_relevance}%) looks good!")
            elif avg_relevance and avg_relevance < 5:
                print(f"❌ Average relevance ({avg_relevance}%) is very low - this is the problem!")
            else:
                print(f"⚠️ Average relevance ({avg_relevance}%) is moderate")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    test_frontend_multimodal_endpoint()
