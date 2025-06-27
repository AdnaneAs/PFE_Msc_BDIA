#!/usr/bin/env python3
"""
Real-world test of improved relevance calculation
"""
import sys
import os
import asyncio
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_query():
    """Test with a real query to see relevance score improvements"""
    try:
        from app.services.vector_db_service import query_documents_advanced
        from app.services.embedding_service import generate_embedding
        
        # Test query
        test_query = "audit compliance requirements"
        
        print(f"🔍 Testing query: '{test_query}'")
        print("=" * 60)
        
        # Get embedding for the query
        query_embedding = generate_embedding(test_query)
        
        # Test with the improved relevance calculation
        results = query_documents_advanced(
            query_embedding=query_embedding,
            query_text=test_query,
            n_results=5,
            search_strategy="semantic"
        )
        
        documents = results.get("documents", [])
        relevance_scores = results.get("relevance_scores", [])
        metadatas = results.get("metadatas", [])
        
        print(f"📊 Found {len(documents)} relevant documents:")
        print("-" * 60)
        
        for i, (doc, score, metadata) in enumerate(zip(documents, relevance_scores, metadatas)):
            filename = metadata.get('original_filename', 'unknown')
            chunk_idx = metadata.get('chunk_index', 0)
            print(f"{i+1}. Score: {score:.1f}% | {filename}:{chunk_idx}")
            print(f"   Preview: {doc[:100]}...")
            print()
        
        if relevance_scores:
            avg_score = sum(relevance_scores) / len(relevance_scores)
            max_score = max(relevance_scores)
            min_score = min(relevance_scores)
            
            print(f"📈 Relevance Statistics:")
            print(f"   Average: {avg_score:.1f}%")
            print(f"   Maximum: {max_score:.1f}%")
            print(f"   Minimum: {min_score:.1f}%")
            print(f"   Range: {max_score - min_score:.1f}%")
            
            # Check if we're getting realistic scores (> 30% for top results)
            realistic_scores = [s for s in relevance_scores if s > 30]
            print(f"   Scores > 30%: {len(realistic_scores)}/{len(relevance_scores)}")
            
            if avg_score > 40:
                print("✅ Relevance scores look good - academic improvement working!")
            elif avg_score > 25:
                print("⚠️  Moderate relevance scores - some improvement visible")
            else:
                print("❌ Low relevance scores - may need further tuning")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Real-World Relevance Score Test\n")
    
    # Run the test (no async needed now)
    success = test_real_query()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("💡 The improved relevance calculation is now active in your system")
    else:
        print("\n❌ Test failed - check your vector database setup")
