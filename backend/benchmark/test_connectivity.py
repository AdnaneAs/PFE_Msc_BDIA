import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb

print("Testing Qdrant connection...")
try:
    qdrant_client = QdrantClient("localhost", port=6333)
    # Test basic operation
    collection_name = "test_collection"
    try:
        qdrant_client.delete_collection(collection_name)
    except:
        pass
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )
    print("✅ Qdrant connection successful!")
    
    # Test insert
    vectors = np.random.random((10, 128)).astype(np.float32)
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"test": f"value_{i}"}
        )
        for i in range(10)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print("✅ Qdrant insert successful!")
    
    # Test search
    query_vector = np.random.random(128).astype(np.float32)
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=5
    )
    print(f"✅ Qdrant search successful! Found {len(search_result.points)} results")
    
except Exception as e:
    print(f"❌ Qdrant error: {e}")

print("\nTesting ChromaDB connection...")
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("test_chroma")
    print("✅ ChromaDB connection successful!")
    
    # Test insert
    vectors = np.random.random((10, 128)).astype(np.float32)
    collection.add(
        embeddings=vectors.tolist(),
        ids=[f"id_{i}" for i in range(10)],
        metadatas=[{"test": f"value_{i}"} for i in range(10)]
    )
    print("✅ ChromaDB insert successful!")
    
    # Test search
    query_vector = np.random.random((1, 128)).astype(np.float32)
    search_result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=5
    )
    print(f"✅ ChromaDB search successful! Found {len(search_result['ids'][0])} results")
    
except Exception as e:
    print(f"❌ ChromaDB error: {e}")

print("\n🎉 Basic connectivity test completed!")
