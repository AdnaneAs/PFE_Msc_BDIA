import os
from dotenv import load_dotenv

load_dotenv()

# Check LlamaParse library defaults
try:
    from llama_parse import LlamaParse
    
    # Create a parser to see what URL it uses
    api_key = os.getenv('LLAMAPARSE_API_KEY')
    parser = LlamaParse(api_key=api_key)
    
    # Check if the parser has any URL attributes
    print("LlamaParse parser attributes:")
    for attr in dir(parser):
        if 'url' in attr.lower() or 'api' in attr.lower() or 'endpoint' in attr.lower():
            print(f"  {attr}: {getattr(parser, attr, 'N/A')}")
            
    # Check the actual implementation
    print(f"\nParser type: {type(parser)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test the correct API URL
import requests
print("\n=== Testing correct API URL ===")

try:
    # Test the cloud.llamaindex.ai endpoint
    api_key = os.getenv('LLAMAPARSE_API_KEY')
    headers = {'Authorization': f'Bearer {api_key}'}
    
    # Try different possible endpoints
    urls_to_test = [
        "https://api.cloud.llamaindex.ai/api/v1/parsing/upload",
        "https://api.cloud.llamaindex.ai/api/parsing/upload", 
        "https://api.cloud.llamaindex.ai/v1/parse",
        "https://api.cloud.llamaindex.ai/parse"
    ]
    
    for url in urls_to_test:
        try:
            response = requests.post(url, headers=headers, timeout=5)
            print(f"✅ {url}: {response.status_code}")
            if response.status_code != 404:
                print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"❌ {url}: {e}")
            
except Exception as e:
    print(f"Error testing URLs: {e}")
