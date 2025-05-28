import os
import asyncio
import sys
sys.path.append('.')

# Test LlamaParse connection
async def test_llamaparse():
    try:
        print("1. Testing basic import...")
        from llama_parse import LlamaParse
        print("✅ LlamaParse imported successfully")
        
        print("2. Testing API key...")
        from app.config import LLAMAPARSE_API_KEY
        if LLAMAPARSE_API_KEY:
            print(f"✅ API key loaded: {LLAMAPARSE_API_KEY[:10]}...")
        else:
            print("❌ API key not loaded")
            return
        
        print("3. Testing parser creation...")
        parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
            num_workers=1,  # Reduce workers for testing
            verbose=True,   # Enable verbose for debugging
            language="en"
        )
        print("✅ Parser created successfully")
        
        print("4. Testing basic functionality...")
        # Test with a small PDF file
        test_pdf_path = "data/uploads/32efb456-8097-466e-8946-3a17c7a07840.pdf"
        if os.path.exists(test_pdf_path):
            print(f"✅ Test PDF file found: {test_pdf_path}")
            
            # Read file content
            with open(test_pdf_path, "rb") as f:
                file_content = f.read()
            print(f"✅ File read: {len(file_content)} bytes")
            
            print("5. Testing LlamaParse API call...")
            try:
                result = parser.parse(file_content, extra_info={"file_name": "test.pdf"})
                print("✅ LlamaParse API call successful")
                print(f"Result type: {type(result)}")
                
                # Try to get content
                if hasattr(result, 'get_markdown_documents'):
                    docs = result.get_markdown_documents()
                    print(f"✅ Got {len(docs)} markdown documents")
                    if docs:
                        print(f"First doc preview: {docs[0].text[:100]}...")
                else:
                    print(f"Result methods: {[m for m in dir(result) if not m.startswith('_')]}")
                    
            except Exception as api_error:
                print(f"❌ LlamaParse API call failed: {api_error}")
                print(f"Error type: {type(api_error).__name__}")
                import traceback
                traceback.print_exc()
                return
        else:
            print(f"❌ Test PDF file not found: {test_pdf_path}")
            return
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llamaparse())
