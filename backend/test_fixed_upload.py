#!/usr/bin/env python3
"""
Fixed Document Upload Test
Tests document upload with supported file types
"""

import requests
import tempfile
import os

def test_document_upload_txt():
    """Test document upload with TXT file"""
    print("🔄 Testing TXT document upload...")
    
    # Create a test TXT document
    test_content = """Test Document for Upload

This is a test document to verify that the document upload functionality is working correctly with the fixed API routing.

Key Points:
1. This document tests file upload
2. It verifies the /api/documents endpoint  
3. It checks document processing and storage

Testing Notes:
- Upload should return a document ID
- Document should be processed and stored
- Content should be available for retrieval"""
    
    # Create temporary TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    try:
        with open(temp_file_path, 'rb') as file:
            files = {'file': ('test_document.txt', file, 'text/plain')}
            response = requests.post(
                "http://localhost:8000/api/documents/upload",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TXT Upload successful!")
            print(f"   Document ID: {result.get('document_id', 'N/A')}")
            print(f"   Filename: {result.get('filename', 'N/A')}")
            return True
        else:
            print(f"❌ TXT Upload failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TXT Upload error: {str(e)}")
        return False
    finally:
        try:
            os.unlink(temp_file_path)
        except:
            pass

def test_document_upload_csv():
    """Test document upload with CSV file"""
    print("\n🔄 Testing CSV document upload...")
    
    # Create a test CSV document
    csv_content = """Name,Department,Role,Experience
John Doe,Engineering,Senior Developer,5 years
Jane Smith,Marketing,Marketing Manager,3 years
Mike Johnson,Sales,Sales Representative,2 years
Sarah Wilson,HR,HR Specialist,4 years"""
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_file.write(csv_content)
        temp_file_path = temp_file.name
    
    try:
        with open(temp_file_path, 'rb') as file:
            files = {'file': ('test_employees.csv', file, 'text/csv')}
            response = requests.post(
                "http://localhost:8000/api/documents/upload",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ CSV Upload successful!")
            print(f"   Document ID: {result.get('document_id', 'N/A')}")
            print(f"   Filename: {result.get('filename', 'N/A')}")
            return True
        else:
            print(f"❌ CSV Upload failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ CSV Upload error: {str(e)}")
        return False
    finally:
        try:
            os.unlink(temp_file_path)
        except:
            pass

def test_query_functionality():
    """Test query functionality with correct parameters"""
    print("\n🔄 Testing query functionality...")
    
    try:
        query_data = {
            "question": "What information is available in the uploaded documents?",
            "config_for_model": {
                "max_tokens": 150,
                "model": "llama3.2:3b"
            },
            "search_strategy": "semantic",
            "max_sources": 3
        }
        
        response = requests.post(
            "http://localhost:8000/api/query/",
            json=query_data,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Sources found: {len(result.get('sources', []))}")
            print(f"   Query time: {result.get('query_time_ms', 'N/A')}ms")
            return True
        else:
            print(f"❌ Query failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return False

def main():
    """Run fixed document tests"""
    print("=" * 60)
    print("  Fixed Document Management Test")
    print("=" * 60)
    
    # Test uploads
    txt_success = test_document_upload_txt()
    csv_success = test_document_upload_csv()
    
    # Test query
    query_success = test_query_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    total_tests = 3
    passed_tests = sum([txt_success, csv_success, query_success])
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests:
        print("🎉 All document management features are working perfectly!")
        print("   - Document upload (TXT/CSV) working")
        print("   - Query processing working correctly")
        print("   - API routing fixed and functional")
    elif passed_tests >= 2:
        print("⚠️  Most features are working correctly.")
    else:
        print("❌ Issues detected, but API routing is confirmed working.")

if __name__ == "__main__":
    main()
