import requests
import json
import time

print("=== Testing Current RAG Query System ===")

# Test the query endpoint
def test_query():
    query_data = {
        "question": "What is the main topic of the documents?",
        "config_for_model": {
            "provider": "ollama",
            "model": "llama2"
        }
    }
    
    try:
        print("1. Testing basic query...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/api/query/", 
            json=query_data,
            timeout=30
        )
        
        end_time = time.time()
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Sources found: {result.get('num_sources', 0)}")
            print(f"   Query time: {result.get('query_time_ms', 0)}ms")
            print(f"   Model used: {result.get('model', 'Unknown')}")
            
            # Show answer preview
            answer = result.get('answer', '')
            if answer:
                print(f"   Answer preview: {answer[:200]}...")
            
            # Show sources
            sources = result.get('sources', [])
            if sources:
                print(f"   Top sources:")
                for i, source in enumerate(sources[:3]):
                    filename = source.get('filename', 'Unknown')
                    chunk_idx = source.get('chunk_index', 0)
                    print(f"     [{i+1}] {filename} (chunk {chunk_idx})")
                    
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

def test_models():
    try:
        print("\n2. Testing available models...")
        response = requests.get("http://localhost:8000/api/query/models")
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {len(models.get('models', []))}")
            for model in models.get('models', [])[:3]:
                print(f"   - {model.get('name', 'Unknown')} ({model.get('provider', 'Unknown')})")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Models test failed: {e}")

def test_status():
    try:
        print("\n3. Testing LLM status...")
        response = requests.get("http://localhost:8000/api/query/status")
        
        if response.status_code == 200:
            status = response.json()
            print("✅ LLM status retrieved:")
            print(f"   Processing: {status.get('is_processing', False)}")
            print(f"   Total queries: {status.get('total_queries', 0)}")
            print(f"   Successful queries: {status.get('successful_queries', 0)}")
            print(f"   Last model: {status.get('last_model_used', 'None')}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Status test failed: {e}")

if __name__ == "__main__":
    test_models()
    test_status()
    test_query()
