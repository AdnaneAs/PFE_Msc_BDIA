import os
import asyncio
import sys
sys.path.append('.')

async def test_application_flow():
    """Test the exact flow that happens in the application"""
    try:
        print("Testing application PDF processing flow...")
        
        # Import the function that's actually called
        from app.services.llamaparse_service import parse_pdf_document, processing_status
        
        # Create a mock UploadFile object
        class MockUploadFile:
            def __init__(self, filename, content):
                self.filename = filename
                self.content = content
            
            async def read(self):
                return self.content
        
        # Read test PDF
        test_pdf_path = "data/uploads/32efb456-8097-466e-8946-3a17c7a07840.pdf"
        with open(test_pdf_path, "rb") as f:
            pdf_content = f.read()
        
        mock_file = MockUploadFile("test.pdf", pdf_content)
        doc_id = "test_123"
        
        print(f"1. Starting PDF processing for doc_id: {doc_id}")
        
        # Call the actual function used in the application
        result = await parse_pdf_document(mock_file, doc_id)
        
        print("✅ PDF processing completed successfully!")
        print(f"Result length: {len(result)} characters")
        print(f"Result preview: {result[:200]}...")
        
        # Check final status
        final_status = processing_status.get(doc_id, {})
        print(f"Final status: {final_status}")
        
    except Exception as e:
        print(f"❌ Error in application flow: {e}")
        import traceback
        traceback.print_exc()
        
        # Check error status
        error_status = processing_status.get(doc_id, {})
        print(f"Error status: {error_status}")

if __name__ == "__main__":
    asyncio.run(test_application_flow())
