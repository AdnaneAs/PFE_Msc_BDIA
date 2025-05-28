#!/usr/bin/env python3
"""
Test script to verify the nested async fix for LlamaParse image extraction.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.llamaparse_service import parse_document_with_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_image_extraction_fix():
    """Test the nested async fix with a sample PDF."""
    
    print("🧪 Testing nested async fix for LlamaParse image extraction...")
    
    # Check if we have any PDF files in uploads directory
    uploads_dir = Path("data/uploads")
    if not uploads_dir.exists():
        print("❌ No uploads directory found. Please upload a PDF file first.")
        return False
    
    pdf_files = list(uploads_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in uploads directory.")
        print("💡 Please upload a PDF file through the frontend first.")
        return False
    
    # Use the first PDF file found
    test_pdf = pdf_files[0]
    document_id = f"test_{test_pdf.stem}"
    
    print(f"📄 Testing with PDF: {test_pdf.name}")
    print(f"🆔 Document ID: {document_id}")
    
    try:
        # Test the enhanced parsing with image extraction
        text_content, image_paths = await parse_document_with_images(
            str(test_pdf), 
            "pdf", 
            document_id
        )
        
        print(f"✅ Successfully parsed document!")
        print(f"📝 Text content length: {len(text_content) if text_content else 0} characters")
        print(f"🖼️  Images extracted: {len(image_paths)}")
        
        if image_paths:
            print("📂 Image files saved:")
            for img_path in image_paths:
                if os.path.exists(img_path):
                    file_size = os.path.getsize(img_path)
                    print(f"   ✓ {os.path.basename(img_path)} ({file_size} bytes)")
                else:
                    print(f"   ❌ {os.path.basename(img_path)} (file not found)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        logger.exception("Detailed error:")
        return False

async def test_basic_functionality():
    """Test basic LlamaParse functionality without images."""
    
    print("\n🔧 Testing basic LlamaParse functionality...")
    
    try:
        from app.services.llamaparse_service import parse_document
        
        # Create a simple test text file
        test_content = "This is a test document for basic functionality verification."
        test_file = "test_basic.txt"
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        result = await parse_document(test_file, "txt")
        
        os.remove(test_file)  # Clean up
        
        if result and test_content in result:
            print("✅ Basic document parsing works correctly")
            return True
        else:
            print("❌ Basic document parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 Starting LlamaParse nested async fix verification...")
    print("=" * 60)
    
    try:
        # Test basic functionality first
        basic_result = asyncio.run(test_basic_functionality())
        
        # Test image extraction fix
        image_result = asyncio.run(test_image_extraction_fix())
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print(f"   Basic functionality: {'✅ PASS' if basic_result else '❌ FAIL'}")
        print(f"   Image extraction fix: {'✅ PASS' if image_result else '❌ FAIL'}")
        
        if basic_result and image_result:
            print("\n🎉 All tests passed! The nested async fix is working correctly.")
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the logs above.")
            return False
            
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        logger.exception("Detailed error:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
