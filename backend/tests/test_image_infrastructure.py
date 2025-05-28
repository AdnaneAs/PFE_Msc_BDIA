#!/usr/bin/env python3
"""
Test script to find and test with a PDF that contains images.
This will help us validate the image extraction functionality.
"""

import asyncio
import logging
import os
import sys
import requests
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llamaparse_service import parse_document_with_images, get_document_images, IMAGES_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_with_image_rich_pdf():
    """Try to find and test a PDF with images"""
    
    print("🔍 SEARCHING FOR PDF WITH IMAGES")
    print("=" * 50)
    
    uploads_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
    pdf_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')]
    
    # Test multiple PDFs to find one with images
    for i, pdf_file in enumerate(pdf_files[:5]):  # Test first 5 PDFs
        print(f"\n📄 Testing PDF {i+1}/5: {pdf_file}")
        
        test_pdf_path = os.path.join(uploads_dir, pdf_file)
        document_id = f"image_test_{pdf_file.replace('.', '_').replace('-', '_')}"
        
        try:
            # Quick test with this PDF
            text_content, image_paths = await parse_document_with_images(
                file_path=test_pdf_path,
                file_type="pdf",
                document_id=document_id
            )
            
            if image_paths:
                print(f"   🎉 FOUND {len(image_paths)} IMAGES!")
                
                for j, img_path in enumerate(image_paths, 1):
                    if os.path.exists(img_path):
                        file_size = os.path.getsize(img_path)
                        print(f"   📸 Image {j}: {os.path.basename(img_path)} ({file_size:,} bytes)")
                
                # Test the API endpoints
                print(f"\n🌐 Testing API with document containing images...")
                
                try:
                    response = requests.get(f"http://localhost:8000/api/documents/{document_id}/images")
                    if response.status_code == 200:
                        api_data = response.json()
                        print(f"   ✅ API Response: {api_data.get('image_count', 0)} images found")
                        
                        # Test individual image endpoints
                        for img_info in api_data.get('images', []):
                            img_url = f"http://localhost:8000{img_info['url']}"
                            img_response = requests.head(img_url)
                            status = "✅" if img_response.status_code == 200 else "❌"
                            print(f"   {status} Image URL: {img_info['filename']}")
                        
                    else:
                        print(f"   ⚠️  API returned status {response.status_code}")
                
                except Exception as api_error:
                    print(f"   ❌ API test error: {api_error}")
                
                return True  # Found a PDF with images
            else:
                print(f"   📝 Text extracted: {len(text_content) if text_content else 0} chars, no images")
                
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file}: {e}")
    
    print(f"\n⚠️  No PDFs with images found in the first 5 tested")
    return False

async def create_test_image_document():
    """Create a simple test to validate the image infrastructure"""
    
    print(f"\n🔧 TESTING IMAGE INFRASTRUCTURE")
    print("=" * 50)
    
    # Test image directory creation
    test_doc_id = "infrastructure_test"
    test_dir = os.path.join(IMAGES_DIR, test_doc_id)
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ Created test image directory: {test_dir}")
        
        # Create a dummy image file to test the infrastructure
        dummy_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        dummy_image_path = os.path.join(test_dir, "test_image.png")
        with open(dummy_image_path, 'wb') as f:
            f.write(dummy_image_content)
        
        print(f"✅ Created dummy image: {dummy_image_path}")
        
        # Test get_document_images function
        images = get_document_images(test_doc_id)
        print(f"✅ get_document_images found: {len(images)} images")
        
        # Test API endpoint
        try:
            response = requests.get(f"http://localhost:8000/api/documents/{test_doc_id}/images")
            if response.status_code == 200:
                api_data = response.json()
                print(f"✅ API endpoint working: {api_data.get('image_count', 0)} images")
            else:
                print(f"⚠️  API returned status {response.status_code}")
        except Exception as api_error:
            print(f"❌ API test error: {api_error}")
        
        # Cleanup
        os.remove(dummy_image_path)
        os.rmdir(test_dir)
        print(f"✅ Cleaned up test files")
        
        return True
        
    except Exception as e:
        print(f"❌ Infrastructure test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🖼️  ENHANCED IMAGE EXTRACTION VALIDATION")
    print("=" * 60)
    
    # Test 1: Try to find a PDF with images
    found_images = await test_with_image_rich_pdf()
    
    # Test 2: Validate infrastructure
    infrastructure_ok = await create_test_image_document()
    
    print(f"\n📋 SUMMARY")
    print("=" * 30)
    print(f"Image extraction tested: {'✅' if found_images else '⚠️  (no image-rich PDFs found)'}")
    print(f"Infrastructure working: {'✅' if infrastructure_ok else '❌'}")
    
    if infrastructure_ok:
        print(f"\n🎉 Image extraction infrastructure is ready!")
        print(f"💡 To test with images, upload a PDF containing charts/diagrams via the frontend")
    else:
        print(f"\n❌ Infrastructure issues detected")

if __name__ == "__main__":
    asyncio.run(main())
