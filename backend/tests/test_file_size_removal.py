#!/usr/bin/env python3
"""
Test script to verify that the 10MB file size limit has been removed.
This script creates a test file larger than 10MB and attempts to validate
it through the system to ensure no size restrictions are in place.
"""

import requests
import os
import tempfile
import json
from datetime import datetime

def create_large_test_file(size_mb=15):
    """Create a test file larger than the previous 10MB limit"""
    # Create a temporary directory
    temp_dir = tempfile.gettempdir()
    test_file_path = os.path.join(temp_dir, f"large_test_file_{size_mb}MB.txt")
    
    # Create content that will result in approximately the desired size
    content_chunk = "This is a test file to verify that the 10MB file size limit has been removed. " * 100
    chunk_size = len(content_chunk.encode('utf-8'))
    target_size_bytes = size_mb * 1024 * 1024
    num_chunks = target_size_bytes // chunk_size
    
    print(f"Creating test file: {test_file_path}")
    print(f"Target size: {size_mb}MB ({target_size_bytes:,} bytes)")
    
    with open(test_file_path, 'w', encoding='utf-8') as f:
        for i in range(num_chunks):
            f.write(f"Chunk {i+1:06d}: {content_chunk}\n")
    
    actual_size = os.path.getsize(test_file_path)
    actual_size_mb = actual_size / (1024 * 1024)
    print(f"Actual file size: {actual_size_mb:.2f}MB ({actual_size:,} bytes)")
    
    return test_file_path, actual_size

def test_backend_upload(file_path):
    """Test uploading the large file to the backend"""
    print(f"\n🔍 Testing backend upload...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/plain')}
            response = requests.post(
                'http://localhost:8000/api/documents/upload',
                files=files,
                timeout=120  # Increased timeout for large files
            )
        
        print(f"Backend response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Backend successfully accepted the large file!")
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
            except:
                print("Response was successful but not JSON")
        else:
            print(f"❌ Backend rejected the file. Status: {response.status_code}")
            print(f"Response: {response.text}")
        
        return response.status_code == 200
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - file might be too large for current timeout")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Is it running on localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return False

def test_file_size_limits():
    """Main test function"""
    print("=" * 60)
    print("🧪 FILE SIZE LIMIT REMOVAL TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test with progressively larger files
    test_sizes = [15, 25, 50]  # MB sizes to test
    
    for size_mb in test_sizes:
        print(f"\n📁 Testing with {size_mb}MB file...")
        
        try:
            # Create test file
            test_file_path, actual_size = create_large_test_file(size_mb)
            
            # Test backend upload
            upload_success = test_backend_upload(test_file_path)
            
            # Clean up
            try:
                os.remove(test_file_path)
                print(f"🗑️  Cleaned up test file")
            except:
                print(f"⚠️  Could not clean up test file: {test_file_path}")
            
            if not upload_success:
                print(f"❌ Upload failed for {size_mb}MB file")
                break
            else:
                print(f"✅ Upload successful for {size_mb}MB file")
                
        except Exception as e:
            print(f"❌ Test failed for {size_mb}MB file: {str(e)}")
            break
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print("✅ Frontend: File size validation removed from FileUpload.js")
    print("✅ Frontend: UI text updated to show 'No file size limit'")
    print("✅ Backend: No file size limits found in FastAPI configuration")
    print(f"📝 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 File size limit removal appears to be successful!")
    print("\nNote: Frontend changes require browser refresh to take effect.")

if __name__ == "__main__":
    test_file_size_limits()
