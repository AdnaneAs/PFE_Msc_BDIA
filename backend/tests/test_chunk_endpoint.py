#!/usr/bin/env python3
"""
Test script for the new chunk content endpoint
Tests the /api/documents/{doc_id}/chunks endpoint
"""

import requests
import json

def test_chunk_endpoint():
    """Test the chunk content endpoint"""
    print("=" * 60)
    print("  Testing Document Chunks Endpoint")
    print("=" * 60)
    
    try:
        # First, get list of documents to find a valid doc_id
        print("🔄 Getting list of documents...")
        docs_response = requests.get("http://localhost:8000/api/documents/", timeout=10)
        
        if docs_response.status_code != 200:
            print(f"❌ Failed to get documents: {docs_response.status_code}")
            return False
        
        documents_data = docs_response.json()
        documents = documents_data.get('documents', [])
        
        if not documents:
            print("❌ No documents found in the system")
            return False
        
        # Find a document with chunks
        test_doc = None
        for doc in documents:
            if doc.get('status') == 'completed' and doc.get('chunk_count', 0) > 0:
                test_doc = doc
                break
        
        if not test_doc:
            print("❌ No completed documents with chunks found")
            return False
        
        doc_id = test_doc['id']
        print(f"✅ Found test document: {test_doc['filename']} (ID: {doc_id})")
        print(f"   Status: {test_doc['status']}, Chunks: {test_doc.get('chunk_count', 0)}")
        
        # Test the chunks endpoint
        print(f"\n🔄 Testing chunks endpoint for document {doc_id}...")
        chunks_response = requests.get(f"http://localhost:8000/api/documents/{doc_id}/chunks", timeout=10)
        
        if chunks_response.status_code == 200:
            chunks_data = chunks_response.json()
            print(f"✅ Chunks endpoint successful!")
            print(f"   Document ID: {chunks_data.get('doc_id')}")
            print(f"   Total chunks: {chunks_data.get('chunk_count', 0)}")
            
            chunks = chunks_data.get('chunks', [])
            if chunks:
                print(f"   First chunk preview:")
                first_chunk = chunks[0]
                content_preview = first_chunk.get('content', '')[:100]
                print(f"     Content: {content_preview}...")
                print(f"     Metadata: {first_chunk.get('metadata', {})}")
                
                # Test specific chunk endpoint
                chunk_index = first_chunk.get('chunk_index', 0)
                print(f"\n🔄 Testing specific chunk endpoint (index {chunk_index})...")
                chunk_response = requests.get(
                    f"http://localhost:8000/api/documents/{doc_id}/chunks/{chunk_index}", 
                    timeout=10
                )
                
                if chunk_response.status_code == 200:
                    chunk_data = chunk_response.json()
                    print(f"✅ Specific chunk endpoint successful!")
                    print(f"   Chunk ID: {chunk_data.get('chunk_id')}")
                    print(f"   Content length: {len(chunk_data.get('content', ''))}")
                    return True
                else:
                    print(f"❌ Specific chunk endpoint failed: {chunk_response.status_code}")
                    print(f"   Response: {chunk_response.text}")
                    return False
            else:
                print("⚠️  No chunks found in response")
                return False
        else:
            print(f"❌ Chunks endpoint failed: {chunks_response.status_code}")
            print(f"   Response: {chunks_response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Exception during testing: {str(e)}")
        return False

def test_nonexistent_document():
    """Test with non-existent document ID"""
    print(f"\n🔄 Testing with non-existent document ID...")
    
    try:
        response = requests.get("http://localhost:8000/api/documents/99999/chunks", timeout=10)
        
        if response.status_code == 404:
            print(f"✅ Correctly returned 404 for non-existent document")
            return True
        else:
            print(f"❌ Expected 404, got {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("Testing new chunk content endpoints...")
    
    success1 = test_chunk_endpoint()
    success2 = test_nonexistent_document()
    
    print(f"\n" + "=" * 60)
    if success1 and success2:
        print("✅ All chunk endpoint tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
