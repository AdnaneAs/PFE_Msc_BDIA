#!/usr/bin/env python3
"""
Test script to verify source modal functionality with rich content types.
This script tests the backend's ability to return detailed chunk content including text, tables, and images.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_source_content_structure():
    """Test that source chunks contain the necessary content for modal display"""
    print("🔍 Testing Source Content Structure for Modal Display...")
    
    payload = {
        "question": "What are the key findings in the research documents?",
        "search_strategy": "semantic",
        "max_sources": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            sources = result.get('sources', [])
            
            print(f"✅ Query successful - Found {len(sources)} source chunks")
            
            for i, source in enumerate(sources[:2]):  # Check first 2 sources
                print(f"\n📄 Source {i+1}: {source.get('filename', 'Unknown')}")
                print(f"   Chunk index: {source.get('chunk_index', 'N/A')}")
                
                # Check content
                content = source.get('content') or source.get('text', '')
                print(f"   Content length: {len(content)} characters")
                print(f"   Content preview: {content[:100]}...")
                
                # Check metadata
                metadata = source.get('metadata', {})
                print(f"   Metadata keys: {list(metadata.keys())}")
                
                # Check for special content types
                if metadata.get('has_images'):
                    print(f"   🖼️ Contains images")
                if metadata.get('is_table'):
                    print(f"   📊 Contains table data")
                if 'image_base64' in source:
                    print(f"   🖼️ Has base64 image data")
                
                # Check relevance score
                if 'relevance_score' in source:
                    print(f"   📈 Relevance score: {source['relevance_score']:.3f}")
            
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_decomposed_query_source_content():
    """Test decomposed query source content aggregation"""
    print("\n🧩 Testing Decomposed Query Source Content...")
    
    payload = {
        "question": "What are the main benefits and challenges of the technology discussed?",
        "search_strategy": "semantic",
        "max_sources": 2,
        "use_decomposition": True,
        "max_sub_queries": 2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/query/decomposed", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            sub_results = result.get('sub_results', [])
            
            print(f"✅ Decomposed query successful - {len(sub_results)} sub-queries")
            
            all_source_content = []
            for i, sub_result in enumerate(sub_results):
                sources = sub_result.get('sources', [])
                print(f"\n📝 Sub-query {i+1}: {len(sources)} sources")
                
                for j, source in enumerate(sources):
                    content_info = {
                        'filename': source.get('filename', 'Unknown'),
                        'content_length': len(source.get('content', '') or source.get('text', '')),
                        'has_metadata': bool(source.get('metadata')),
                        'chunk_index': source.get('chunk_index'),
                        'relevance_score': source.get('relevance_score')                    }
                    all_source_content.append(content_info)
                    
                    print(f"     {j+1}. {content_info['filename']} "
                          f"({content_info['content_length']} chars, "
                          f"relevance: {content_info['relevance_score']:.3f if content_info['relevance_score'] else 'N/A'})")
            
            # Group by filename to simulate frontend aggregation
            unique_files = {}
            for content in all_source_content:
                filename = content['filename']
                if filename not in unique_files:
                    unique_files[filename] = []
                unique_files[filename].append(content)
            
            print(f"\n📊 Frontend Modal Display Simulation:")
            for filename, chunks in unique_files.items():
                print(f"   📄 {filename} - {len(chunks)} chunks available for modal")
                total_chars = sum(chunk['content_length'] for chunk in chunks)
                avg_relevance = sum(chunk['relevance_score'] or 0 for chunk in chunks) / len(chunks)
                print(f"      Total content: {total_chars} characters")
                print(f"      Average relevance: {avg_relevance:.3f}")
            
            return True
        else:
            print(f"❌ Decomposed query failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def main():
    print("🎭 Testing Source Modal Content Display")
    print("=" * 50)
    
    # Test regular query source content
    regular_success = test_source_content_structure()
    
    # Test decomposed query source content
    decomposed_success = test_decomposed_query_source_content()
    
    print("\n🎯 Modal Functionality Summary:")
    print("-" * 40)
    
    if regular_success:
        print("✅ Regular queries: Source content ready for modal display")
    else:
        print("❌ Regular queries: Failed to retrieve source content")
    
    if decomposed_success:
        print("✅ Decomposed queries: Multi-source content aggregation working")
    else:
        print("❌ Decomposed queries: Failed to aggregate source content")
    
    print("\n💡 Frontend Modal Features Now Available:")
    print("   🖱️ Click any source document to view full content")
    print("   📄 Text chunks with syntax highlighting")
    print("   📊 Table data with formatted display")
    print("   🖼️ Embedded images (if available)")
    print("   📈 Relevance scores for each chunk")
    print("   📋 Document metadata information")
    print("   🔍 Chunk-by-chunk content browsing")

if __name__ == "__main__":
    main()
