#!/usr/bin/env python3

from app.services.vector_db_service import query_documents_advanced
from app.services.embedding_service import generate_embedding

query_text = "What is the main topic of the document?"
query_embedding = generate_embedding(query_text)

print("🧪 Testing 17% boost directly...")

results = query_documents_advanced(
    query_embedding=query_embedding,
    query_text=query_text,
    n_results=5,
    use_reranking=True
)

print(f"📊 Results received:")
print(f"  - Documents: {len(results['documents'])}")
print(f"  - Relevance scores: {results.get('relevance_scores', [])}")

for i, metadata in enumerate(results['metadatas'][:3]):
    score = metadata.get('relevance_score', 'N/A')
    print(f"  - Source {i+1}: {score}")
