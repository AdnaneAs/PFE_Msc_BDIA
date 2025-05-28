#!/usr/bin/env python3
"""
Document Upload Test
Tests the document upload functionality with the fixed API routing
"""

import requests
import os
import tempfile
import json

def test_document_upload():
    """Test document upload functionality"""
    print("=" * 60)
    print("  Document Upload Test")
    print("=" * 60)
    
    # Create a test document
    test_content = """
# Test Document for Upload

This is a test document to verify that the document upload functionality is working correctly with the fixed API routing.

## Key Points
1. This document tests file upload
2. It verifies the /api/documents endpoint
3. It checks document processing and storage

## Testing Notes
- Upload should return a document ID
- Document should be processed and stored
- Content should be available for retrieval
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    try:
        # Test upload
        print("🔄 Testing document upload...")
        
        with open(temp_file_path, 'rb') as file:
            files = {'file': ('test_document.md', file, 'text/markdown')}
            response = requests.post(
                "http://localhost:8000/api/documents/upload",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Document ID: {result.get('document_id', 'N/A')}")
            print(f"   Filename: {result.get('filename', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            return True
        else:
            print(f"❌ Upload failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink(temp_file_path)
        except:
            pass

def test_document_retrieval():
    """Test document retrieval"""
    print("\n🔄 Testing document retrieval...")
    
    try:
        response = requests.get("http://localhost:8000/api/documents", timeout=10)
        
        if response.status_code == 200:
            documents = response.json()
            print(f"✅ Retrieved {len(documents)} documents")
            
            # Show first few documents
            for i, doc in enumerate(documents[:3]):
                print(f"   {i+1}. {doc.get('filename', 'Unknown')} ({doc.get('id', 'No ID')})")
            
            if len(documents) > 3:
                print(f"   ... and {len(documents) - 3} more documents")
            
            return True
        else:
            print(f"❌ Retrieval failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Retrieval error: {str(e)}")
        return False

def test_query_functionality():
    """Test query functionality"""
    print("\n🔄 Testing query functionality...")
    
    try:
        query_data = {
            "query": "What types of documents are available in the system?",
            "max_tokens": 100,
            "model": "llama3.2:3b"
        }
        
        response = requests.post(
            "http://localhost:8000/api/query/",
            json=query_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"   Response length: {len(result.get('response', ''))}")
            print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
            print(f"   Sources found: {len(result.get('sources', []))}")
            return True
        else:
            print(f"❌ Query failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return False

def main():
    """Run document functionality tests"""
    print("Testing Document Management Functionality\n")
    
    # Test document retrieval (should work)
    retrieval_success = test_document_retrieval()
    
    # Test document upload
    upload_success = test_document_upload()
    
    # Test query functionality
    query_success = test_query_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    total_tests = 3
    passed_tests = sum([retrieval_success, upload_success, query_success])
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests:
        print("🎉 All document management features are working perfectly!")
    elif passed_tests >= 2:
        print("⚠️  Most document management features are working.")
        print("   Some minor issues may need attention.")
    else:
        print("❌ Document management system needs attention.")
        print("   Multiple issues detected.")

if __name__ == "__main__":
    main()
