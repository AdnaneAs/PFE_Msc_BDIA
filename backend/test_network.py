import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Test basic HTTP connectivity
print("=== Testing Network Connectivity ===")

try:
    # Test basic HTTPS connection
    response = requests.get("https://api.cloud.llamaindex.ai", timeout=10)
    print(f"✅ Basic HTTPS connection: {response.status_code}")
except Exception as e:
    print(f"❌ Basic HTTPS connection failed: {e}")

# Test with LlamaParse API key
api_key = os.getenv('LLAMAPARSE_API_KEY')
if api_key:
    print(f"✅ API Key loaded: {api_key[:10]}...")
    
    # Test API endpoint with auth
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        response = requests.get("https://api.cloud.llamaindex.ai/api/parsing/health", 
                              headers=headers, timeout=10)
        print(f"✅ API health check: {response.status_code}")
    except Exception as e:
        print(f"❌ API health check failed: {e}")
else:
    print("❌ No API key found")

# Test the actual LlamaParse library
print("\n=== Testing LlamaParse Library ===")
try:
    from llama_parse import LlamaParse
    print("✅ LlamaParse imported successfully")
    
    # Try to create a parser instance
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )
    print("✅ LlamaParse parser created")
    
    # Test a simple API call (without actually parsing a file)
    # This will help us see if the library can connect to the API
    print("Testing API connectivity through LlamaParse...")
    
except Exception as e:
    print(f"❌ LlamaParse test failed: {e}")
    import traceback
    traceback.print_exc()
