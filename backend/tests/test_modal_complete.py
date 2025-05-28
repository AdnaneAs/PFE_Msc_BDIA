#!/usr/bin/env python3
"""
Complete test for the source modal functionality
Tests the entire flow from query to chunk content retrieval
"""

import requests
import json
import time

def test_complete_modal_flow():
    """Test the complete flow for source modal functionality"""
    print("=" * 70)
    print("  Complete Source Modal Flow Test")
    print("=" * 70)
    
    try:
        # Step 1: Submit a query to get sources
        print("🔄 Step 1: Submitting a query to get sources...")
        query_data = {
            "question": "What are the main topics covered in the documents?",
            "config_for_model": {
                "max_tokens": 150,
                "model": "llama3.2:3b"
            },
            "search_strategy": "semantic",
            "max_sources": 3
        }
        
        query_response = requests.post(
            "http://localhost:8000/api/query/",
            json=query_data,
            timeout=30
        )
        
        if query_response.status_code != 200:
            print(f"❌ Query failed: {query_response.status_code}")
            print(f"   Response: {query_response.text}")
            return False
        
        query_result = query_response.json()
        sources = query_result.get('sources', [])
        
        if not sources:
            print("❌ No sources returned from query")
            return False
        
        print(f"✅ Query successful! Found {len(sources)} sources")
        for i, source in enumerate(sources):
            print(f"   Source {i+1}: {source.get('filename')} (doc_id: {source.get('doc_id')}, chunk: {source.get('chunk_index')})")
        
        # Step 2: Test chunk retrieval for each source
        print(f"\n🔄 Step 2: Testing chunk retrieval for sources...")
        successful_retrievals = 0
        
        for i, source in enumerate(sources):
            doc_id = source.get('doc_id')
            if not doc_id:
                print(f"   ⚠️  Source {i+1}: Missing doc_id")
                continue
            
            print(f"   🔄 Testing source {i+1}: {source.get('filename')} (doc_id: {doc_id})")
            
            # Test chunks endpoint
            chunks_response = requests.get(
                f"http://localhost:8000/api/documents/{doc_id}/chunks",
                timeout=10
            )
            
            if chunks_response.status_code == 200:
                chunks_data = chunks_response.json()
                chunk_count = chunks_data.get('chunk_count', 0)
                chunks = chunks_data.get('chunks', [])
                
                print(f"      ✅ Retrieved {chunk_count} chunks")
                if chunks:
                    # Show preview of first chunk
                    first_chunk = chunks[0]
                    content_preview = first_chunk.get('content', '')[:100]
                    print(f"      📄 First chunk preview: {content_preview}...")
                    
                    # Test specific chunk retrieval
                    chunk_index = first_chunk.get('chunk_index', 0)
                    chunk_response = requests.get(
                        f"http://localhost:8000/api/documents/{doc_id}/chunks/{chunk_index}",
                        timeout=10
                    )
                    
                    if chunk_response.status_code == 200:
                        chunk_data = chunk_response.json()
                        content_length = len(chunk_data.get('content', ''))
                        print(f"      ✅ Specific chunk retrieval successful (content: {content_length} chars)")
                        successful_retrievals += 1
                    else:
                        print(f"      ❌ Specific chunk retrieval failed: {chunk_response.status_code}")
                else:
                    print(f"      ⚠️  No chunks found in response")
            else:
                print(f"      ❌ Chunks retrieval failed: {chunks_response.status_code}")
                print(f"         Response: {chunks_response.text}")
        
        # Step 3: Summary
        print(f"\n📊 Step 3: Summary")
        print(f"   Total sources: {len(sources)}")
        print(f"   Successful chunk retrievals: {successful_retrievals}")
        
        if successful_retrievals > 0:
            print(f"✅ Modal functionality test PASSED!")
            print(f"   Frontend can now display actual chunk content in the modal")
            return True
        else:
            print(f"❌ Modal functionality test FAILED!")
            print(f"   No chunks could be retrieved for modal display")
            return False
    
    except Exception as e:
        print(f"❌ Exception during testing: {str(e)}")
        return False

def test_modal_api_integration():
    """Test API functions that the frontend modal will use"""
    print(f"\n🔄 Testing Frontend API Integration...")
    
    # Test the exact API call pattern the frontend will use
    try:
        # Get documents first
        docs_response = requests.get("http://localhost:8000/api/documents/")
        if docs_response.status_code != 200:
            print("❌ Cannot get documents list")
            return False
        
        docs_data = docs_response.json()
        documents = docs_data.get('documents', [])
        
        # Find any document with chunks
        test_doc = None
        for doc in documents:
            if doc.get('chunk_count', 0) > 0:
                test_doc = doc
                break
        
        if not test_doc:
            print("❌ No documents with chunks found")
            return False
        
        doc_id = test_doc['id']
        filename = test_doc['filename']
        
        print(f"✅ Testing with document: {filename} (ID: {doc_id})")
        
        # Test the chunks API call exactly as frontend will use it
        api_url = f"http://localhost:8000/api/documents/{doc_id}/chunks"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify response structure matches frontend expectations
            required_fields = ['doc_id', 'chunk_count', 'chunks']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"❌ Response missing fields: {missing_fields}")
                return False
            
            chunks = data['chunks']
            if chunks:
                # Verify chunk structure
                first_chunk = chunks[0]
                chunk_required_fields = ['chunk_id', 'content', 'metadata', 'chunk_index']
                chunk_missing_fields = [field for field in chunk_required_fields if field not in first_chunk]
                
                if chunk_missing_fields:
                    print(f"❌ Chunk missing fields: {chunk_missing_fields}")
                    return False
                
                content_length = len(first_chunk.get('content', ''))
                print(f"✅ API response structure is correct")
                print(f"   Doc ID: {data['doc_id']}")
                print(f"   Chunk count: {data['chunk_count']}")
                print(f"   First chunk content length: {content_length}")
                print(f"   First chunk metadata: {first_chunk.get('metadata', {})}")
                
                return True
            else:
                print("❌ No chunks in response")
                return False
        else:
            print(f"❌ API call failed: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Run complete modal functionality tests"""
    print("🧪 Testing Complete Source Modal Functionality")
    print("=" * 70)
    
    # Run tests
    test1_success = test_complete_modal_flow()
    test2_success = test_modal_api_integration()
    
    print(f"\n" + "=" * 70)
    print("📋 FINAL RESULTS:")
    print(f"   Query → Chunks Flow: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   API Integration: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   ✅ Backend endpoints are working correctly")
        print(f"   ✅ Frontend can now fetch and display actual chunk content")
        print(f"   ✅ Source modal functionality is ready for use")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"   Please fix the issues before proceeding")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
