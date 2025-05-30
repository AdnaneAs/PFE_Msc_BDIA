import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb

print("🚀 Starting Simple Vector Database Benchmark...")

# Initialize clients
qdrant_client = QdrantClient("localhost", port=6333)
chroma_client = chromadb.Client()

# Simple benchmark parameters
vector_size = 128
data_size = 1000
k = 5

print(f"📊 Testing with {data_size} vectors of size {vector_size}")

# Generate test data
vectors = np.random.random((data_size, vector_size)).astype(np.float32)
qdrant_ids = list(range(data_size))
chroma_ids = [f"id_{i}" for i in range(data_size)]
metadata = [{"category": f"cat_{i%10}"} for i in range(data_size)]
query_vectors = np.random.random((10, vector_size)).astype(np.float32)

# Benchmark Qdrant
print("🔍 Testing Qdrant...")
collection_name = "benchmark_test"

# Setup
try:
    qdrant_client.delete_collection(collection_name)
except:
    pass

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Insert
start_time = time.time()
points = [
    PointStruct(
        id=qdrant_ids[i],
        vector=vectors[i].tolist(),
        payload=metadata[i]
    )
    for i in range(len(vectors))
]
qdrant_client.upsert(collection_name=collection_name, points=points)
qdrant_insert_time = time.time() - start_time

# Search
start_time = time.time()
for query_vector in query_vectors:
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=k
    )
qdrant_search_time = time.time() - start_time

print(f"✅ Qdrant - Insert: {qdrant_insert_time:.3f}s, Search: {qdrant_search_time:.3f}s")

# Benchmark ChromaDB
print("🔍 Testing ChromaDB...")
try:
    chroma_client.delete_collection("benchmark_test")
except:
    pass

chroma_collection = chroma_client.create_collection("benchmark_test")

# Insert
start_time = time.time()
chroma_collection.add(
    embeddings=vectors.tolist(),
    ids=chroma_ids,
    metadatas=metadata
)
chroma_insert_time = time.time() - start_time

# Search
start_time = time.time()
search_result = chroma_collection.query(
    query_embeddings=query_vectors.tolist(),
    n_results=k
)
chroma_search_time = time.time() - start_time

print(f"✅ ChromaDB - Insert: {chroma_insert_time:.3f}s, Search: {chroma_search_time:.3f}s")

# Results
print("\n📊 Results Summary:")
print(f"Data Size: {data_size} vectors")
print(f"Vector Size: {vector_size} dimensions")
print(f"Query Count: {len(query_vectors)}")
print(f"Top-K: {k}")
print()
print("Insert Performance:")
print(f"  Qdrant:  {qdrant_insert_time:.3f}s ({data_size/qdrant_insert_time:.1f} vectors/sec)")
print(f"  ChromaDB: {chroma_insert_time:.3f}s ({data_size/chroma_insert_time:.1f} vectors/sec)")
print()
print("Search Performance:")
print(f"  Qdrant:  {qdrant_search_time:.3f}s ({len(query_vectors)/qdrant_search_time:.1f} queries/sec)")
print(f"  ChromaDB: {chroma_search_time:.3f}s ({len(query_vectors)/chroma_search_time:.1f} queries/sec)")

# Simple comparison
if qdrant_insert_time < chroma_insert_time:
    insert_winner = "Qdrant"
    insert_ratio = chroma_insert_time / qdrant_insert_time
else:
    insert_winner = "ChromaDB"
    insert_ratio = qdrant_insert_time / chroma_insert_time

if qdrant_search_time < chroma_search_time:
    search_winner = "Qdrant" 
    search_ratio = chroma_search_time / qdrant_search_time
else:
    search_winner = "ChromaDB"
    search_ratio = qdrant_search_time / chroma_search_time

print()
print("🏆 Performance Winners:")
print(f"  Insert: {insert_winner} is {insert_ratio:.2f}x faster")
print(f"  Search: {search_winner} is {search_ratio:.2f}x faster")

print("\n✅ Simple benchmark completed!")
