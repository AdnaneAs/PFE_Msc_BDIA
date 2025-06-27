#!/usr/bin/env python3

from app.services.vector_db_service import query_multimodal_documents
from app.services.embedding_service import generate_embedding

query_text = "What is the main topic of the document?"
query_embedding = generate_embedding(query_text)

print("🧪 Testing 17% boost in multimodal...")

results = query_multimodal_documents(
    query_embedding=query_embedding,
    query_text=query_text,
    n_results=5,
    use_reranking=True
)

print(f"📊 Multimodal Results:")
print(f"  - Documents: {len(results['documents'])}")

for i, metadata in enumerate(results['metadatas'][:3]):
    score = metadata.get('relevance_score', 'N/A')
    content_type = metadata.get('content_type', 'unknown')
    print(f"  - Source {i+1} ({content_type}): {score}")
