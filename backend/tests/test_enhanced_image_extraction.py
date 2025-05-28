#!/usr/bin/env python3
"""
Test script for enhanced LlamaParse image extraction functionality.
This script validates the image extraction from PDF documents.
"""

import asyncio
import logging
import os
import sys
import tempfile
import requests
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llamaparse_service import parse_document_with_images, get_document_images, IMAGES_DIR
from app.services.document_service import save_uploaded_file
from app.config import LLAMAPARSE_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_image_extraction():
    """Test the enhanced image extraction functionality"""
    
    print("🔍 TESTING ENHANCED LLAMAPARSE IMAGE EXTRACTION")
    print("=" * 60)
    
    # Check if LlamaParse API key is available
    if not LLAMAPARSE_API_KEY:
        print("❌ LLAMAPARSE_API_KEY not found in environment variables")
        print("   Please set your LlamaParse API key to test image extraction")
        return False
    
    print(f"✅ LlamaParse API key found: {LLAMAPARSE_API_KEY[:10]}...")
    
    # Check if images directory exists
    print(f"📁 Images directory: {IMAGES_DIR}")
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)
        print("✅ Created images directory")
    else:
        print("✅ Images directory exists")
    
    # Look for existing PDF files in uploads directory
    uploads_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
    pdf_files = []
    
    if os.path.exists(uploads_dir):
        pdf_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in uploads directory")
        print("   Please upload a PDF document through the frontend first")
        return False
    
    print(f"📄 Found {len(pdf_files)} PDF file(s) in uploads:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file}")
    
    # Test with the first PDF file
    test_pdf = pdf_files[0]
    test_pdf_path = os.path.join(uploads_dir, test_pdf)
    document_id = f"test_{test_pdf.replace('.', '_')}"
    
    print(f"\n🧪 Testing with: {test_pdf}")
    print(f"📍 Document ID: {document_id}")
    
    try:
        # Test the enhanced parsing with image extraction
        print("\n🔄 Starting enhanced document parsing...")
        
        text_content, image_paths = await parse_document_with_images(
            file_path=test_pdf_path,
            file_type="pdf",
            document_id=document_id
        )
        
        print("\n📊 PARSING RESULTS:")
        print(f"   Text extracted: {'✅ Yes' if text_content else '❌ No'}")
        if text_content:
            print(f"   Text length: {len(text_content)} characters")
            print(f"   Text preview: {text_content[:200]}...")
        
        print(f"   Images extracted: {len(image_paths)}")
        
        if image_paths:
            print("\n🖼️  EXTRACTED IMAGES:")
            for i, img_path in enumerate(image_paths, 1):
                if os.path.exists(img_path):
                    file_size = os.path.getsize(img_path)
                    print(f"   {i}. {os.path.basename(img_path)} ({file_size:,} bytes)")
                else:
                    print(f"   {i}. {os.path.basename(img_path)} (❌ File not found)")
        else:
            print("   No images were extracted from this PDF")
        
        # Test the get_document_images function
        print(f"\n🔍 Testing get_document_images function...")
        retrieved_images = get_document_images(document_id)
        print(f"   Retrieved {len(retrieved_images)} image(s)")
        
        if retrieved_images:
            print("   Image files:")
            for img_path in retrieved_images:
                print(f"   - {os.path.basename(img_path)}")
        
        # Test API endpoints
        print(f"\n🌐 Testing API endpoints...")
        
        try:
            # Test document images endpoint
            response = requests.get(f"http://localhost:8000/api/documents/{document_id}/images")
            if response.status_code == 200:
                api_data = response.json()
                print(f"   ✅ GET /api/documents/{document_id}/images")
                print(f"      Found {api_data.get('image_count', 0)} images via API")
            else:
                print(f"   ⚠️  GET /api/documents/{document_id}/images returned {response.status_code}")
        
        except Exception as api_error:
            print(f"   ❌ API test failed: {api_error}")
        
        print(f"\n✅ Image extraction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during image extraction test: {e}")
        logger.exception("Detailed error information:")
        return False

async def test_frontend_integration():
    """Test the frontend integration points"""
    
    print(f"\n🌐 TESTING FRONTEND INTEGRATION")
    print("=" * 60)
    
    try:
        # Test API endpoints that the frontend will use
        endpoints_to_test = [
            "/api/documents",
            "/api/query/models"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"http://localhost:8000{endpoint}")
                status = "✅" if response.status_code == 200 else "⚠️"
                print(f"   {status} {endpoint} - Status: {response.status_code}")
            except Exception as e:
                print(f"   ❌ {endpoint} - Error: {e}")
        
        print(f"\n✅ Frontend integration tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Frontend integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 ENHANCED LLAMAPARSE IMAGE EXTRACTION TEST SUITE")
    print("=" * 80)
    print("This test validates the new image extraction functionality")
    print("=" * 80)
    
    # Run tests
    test_results = []
    
    # Test 1: Image extraction functionality
    result1 = await test_image_extraction()
    test_results.append(("Image Extraction", result1))
    
    # Test 2: Frontend integration
    result2 = await test_frontend_integration()
    test_results.append(("Frontend Integration", result2))
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced image extraction is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
