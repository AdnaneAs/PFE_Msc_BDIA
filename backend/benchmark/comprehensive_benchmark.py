import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class VectorDBBenchmarkComplete:
    def __init__(self):
        print("🔧 Initializing benchmark clients...")
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.chroma_client = chromadb.Client()
        
    def run_single_benchmark(self, vector_size: int, data_size: int, k: int):
        """Run a single benchmark configuration"""
        print(f"📊 Benchmarking: {data_size} vectors, {vector_size}D, top-{k}")
        
        # Generate test data
        vectors = np.random.random((data_size, vector_size)).astype(np.float32)
        qdrant_ids = list(range(data_size))
        chroma_ids = [f"id_{i}" for i in range(data_size)]
        metadata = [{"category": f"cat_{i%10}"} for i in range(data_size)]
        query_vectors = np.random.random((50, vector_size)).astype(np.float32)  # 50 queries
        
        results = {}
        
        # Benchmark Qdrant
        print("  🔍 Testing Qdrant...")
        collection_name = f"bench_{vector_size}_{data_size}"
        
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
            
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # Qdrant Insert
        start_time = time.time()
        batch_size = 1000
        for i in range(0, data_size, batch_size):
            end_idx = min(i + batch_size, data_size)
            points = [
                PointStruct(
                    id=qdrant_ids[j],
                    vector=vectors[j].tolist(),
                    payload=metadata[j]
                )
                for j in range(i, end_idx)
            ]
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
        qdrant_insert_time = time.time() - start_time
        
        # Qdrant Search
        start_time = time.time()
        qdrant_results = []
        for query_vector in query_vectors:
            search_result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=k
            )
            qdrant_results.append([(hit.id, hit.score) for hit in search_result.points])
        qdrant_search_time = time.time() - start_time
        
        # Benchmark ChromaDB
        print("  🔍 Testing ChromaDB...")
        try:
            self.chroma_client.delete_collection(f"chroma_{vector_size}_{data_size}")
        except:
            pass
            
        chroma_collection = self.chroma_client.create_collection(f"chroma_{vector_size}_{data_size}")
        
        # ChromaDB Insert
        start_time = time.time()
        batch_size = 1000
        for i in range(0, data_size, batch_size):
            end_idx = min(i + batch_size, data_size)
            chroma_collection.add(
                embeddings=vectors[i:end_idx].tolist(),
                ids=chroma_ids[i:end_idx],
                metadatas=metadata[i:end_idx]
            )
        chroma_insert_time = time.time() - start_time
        
        # ChromaDB Search
        start_time = time.time()
        chroma_search_result = chroma_collection.query(
            query_embeddings=query_vectors.tolist(),
            n_results=k
        )
        chroma_search_time = time.time() - start_time
        
        # Calculate simple relevance metrics (synthetic)
        # In real scenarios, you'd have ground truth relevance data
        qdrant_precision = self.calculate_synthetic_precision(qdrant_results[0], k)
        chroma_precision = self.calculate_synthetic_precision(
            [(chroma_search_result['ids'][0][i], chroma_search_result['distances'][0][i]) 
             for i in range(len(chroma_search_result['ids'][0]))], k
        )
        
        return {
            'vector_size': vector_size,
            'data_size': data_size,
            'k': k,
            'qdrant_insert_time': qdrant_insert_time,
            'qdrant_search_time': qdrant_search_time,
            'qdrant_search_per_query': qdrant_search_time / len(query_vectors),
            'qdrant_precision': qdrant_precision,
            'chroma_insert_time': chroma_insert_time,
            'chroma_search_time': chroma_search_time,
            'chroma_search_per_query': chroma_search_time / len(query_vectors),
            'chroma_precision': chroma_precision,
            'query_count': len(query_vectors)
        }
    
    def calculate_synthetic_precision(self, results, k):
        """Calculate synthetic precision for demonstration"""
        # Simple synthetic precision based on score distribution
        if not results:
            return 0.0
        
        scores = [score for _, score in results[:k]]
        if not scores:
            return 0.0
            
        # Assume higher scores are better, calculate relative precision
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return 1.0
            
        # Normalize scores and use as precision proxy
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        return sum(normalized_scores) / len(normalized_scores)
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across different configurations"""
        print("🚀 Starting Comprehensive Vector Database Benchmark")
        print("=" * 60)
        
        # Benchmark configurations
        configs = [
            (128, 1000, 5),
            (128, 5000, 5),
            (256, 1000, 5),
            (256, 5000, 5),
            (512, 1000, 5),
            (512, 1000, 10),
            (128, 1000, 10),
            (256, 1000, 10),
        ]
        
        results = []
        for i, (vector_size, data_size, k) in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Configuration: {vector_size}D vectors, {data_size} documents, top-{k}")
            try:
                result = self.run_single_benchmark(vector_size, data_size, k)
                results.append(result)
                print(f"  ✅ Completed in {result['qdrant_insert_time'] + result['qdrant_search_time']:.2f}s")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def create_summary_report(self, df: pd.DataFrame):
        """Create a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("📊 VECTOR DATABASE BENCHMARK RESULTS")
        print("=" * 60)
        
        # Overall performance summary
        print("\n🏆 OVERALL PERFORMANCE SUMMARY")
        print("-" * 40)
        
        # Average performance across all tests
        avg_qdrant_insert = df['qdrant_insert_time'].mean()
        avg_chroma_insert = df['chroma_insert_time'].mean()
        avg_qdrant_search = df['qdrant_search_per_query'].mean()
        avg_chroma_search = df['chroma_search_per_query'].mean()
        avg_qdrant_precision = df['qdrant_precision'].mean()
        avg_chroma_precision = df['chroma_precision'].mean()
        
        print(f"Average Insert Time:")
        print(f"  Qdrant:  {avg_qdrant_insert:.4f}s")
        print(f"  ChromaDB: {avg_chroma_insert:.4f}s")
        
        if avg_qdrant_insert < avg_chroma_insert:
            insert_winner = "Qdrant"
            insert_ratio = avg_chroma_insert / avg_qdrant_insert
        else:
            insert_winner = "ChromaDB"
            insert_ratio = avg_qdrant_insert / avg_chroma_insert
        
        print(f"  → {insert_winner} is {insert_ratio:.2f}x faster for insertions")
        
        print(f"\nAverage Search Time per Query:")
        print(f"  Qdrant:  {avg_qdrant_search:.6f}s")
        print(f"  ChromaDB: {avg_chroma_search:.6f}s")
        
        if avg_qdrant_search < avg_chroma_search:
            search_winner = "Qdrant"
            search_ratio = avg_chroma_search / avg_qdrant_search
        else:
            search_winner = "ChromaDB"
            search_ratio = avg_qdrant_search / avg_chroma_search
            
        print(f"  → {search_winner} is {search_ratio:.2f}x faster for searches")
        
        print(f"\nAverage Precision:")
        print(f"  Qdrant:  {avg_qdrant_precision:.4f}")
        print(f"  ChromaDB: {avg_chroma_precision:.4f}")
        
        # Performance by data size
        print("\n📈 PERFORMANCE BY DATA SIZE")
        print("-" * 40)
        
        size_groups = df.groupby('data_size').agg({
            'qdrant_search_per_query': 'mean',
            'chroma_search_per_query': 'mean',
            'qdrant_insert_time': 'mean',
            'chroma_insert_time': 'mean'
        }).round(6)
        
        print(size_groups.to_string())
        
        # Performance by vector dimension
        print("\n📏 PERFORMANCE BY VECTOR DIMENSION")
        print("-" * 40)
        
        dim_groups = df.groupby('vector_size').agg({
            'qdrant_search_per_query': 'mean',
            'chroma_search_per_query': 'mean',
            'qdrant_insert_time': 'mean',
            'chroma_insert_time': 'mean'
        }).round(6)
        
        print(dim_groups.to_string())
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        if search_winner == "Qdrant":
            print(f"✅ For search-heavy workloads: Consider Qdrant ({search_ratio:.1f}x faster searches)")
        else:
            print(f"✅ For search-heavy workloads: Consider ChromaDB ({search_ratio:.1f}x faster searches)")
            
        if insert_winner == "Qdrant":
            print(f"✅ For insert-heavy workloads: Consider Qdrant ({insert_ratio:.1f}x faster insertions)")
        else:
            print(f"✅ For insert-heavy workloads: Consider ChromaDB ({insert_ratio:.1f}x faster insertions)")
        
        # Scalability insights
        large_data = df[df['data_size'] >= 5000]
        if not large_data.empty:
            large_qdrant_search = large_data['qdrant_search_per_query'].mean()
            large_chroma_search = large_data['chroma_search_per_query'].mean()
            
            if large_qdrant_search < large_chroma_search:
                print(f"✅ For large datasets (5K+ vectors): Qdrant shows better search performance")
            else:
                print(f"✅ For large datasets (5K+ vectors): ChromaDB shows better search performance")
        
        print(f"\n🔗 Integration Notes:")
        print(f"   • Qdrant: REST API, gRPC, supports filtering, cloud-native")
        print(f"   • ChromaDB: Python-native, simple API, good for prototyping")
        print(f"   • Consider your specific use case, data size, and query patterns")
        
        return {
            'insert_winner': insert_winner,
            'search_winner': search_winner,
            'insert_ratio': insert_ratio,
            'search_ratio': search_ratio,
            'avg_qdrant_search': avg_qdrant_search,
            'avg_chroma_search': avg_chroma_search
        }
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create performance visualization charts"""
        print("\n📊 Creating performance visualizations...")
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Search time comparison
        search_data = []
        for _, row in df.iterrows():
            search_data.append({'DB': 'Qdrant', 'Search Time (s)': row['qdrant_search_per_query'], 
                              'Data Size': row['data_size'], 'Vector Size': row['vector_size']})
            search_data.append({'DB': 'ChromaDB', 'Search Time (s)': row['chroma_search_per_query'],
                              'Data Size': row['data_size'], 'Vector Size': row['vector_size']})
        
        search_df = pd.DataFrame(search_data)
        
        sns.boxplot(data=search_df, x='DB', y='Search Time (s)', ax=axes[0,0])
        axes[0,0].set_title('Search Time Comparison')
        axes[0,0].set_ylabel('Time per Query (seconds)')
        
        # Insert time comparison
        insert_data = []
        for _, row in df.iterrows():
            insert_data.append({'DB': 'Qdrant', 'Insert Time (s)': row['qdrant_insert_time'],
                              'Data Size': row['data_size'], 'Vector Size': row['vector_size']})
            insert_data.append({'DB': 'ChromaDB', 'Insert Time (s)': row['chroma_insert_time'],
                              'Data Size': row['data_size'], 'Vector Size': row['vector_size']})
        
        insert_df = pd.DataFrame(insert_data)
        
        sns.boxplot(data=insert_df, x='DB', y='Insert Time (s)', ax=axes[0,1])
        axes[0,1].set_title('Insert Time Comparison')
        axes[0,1].set_ylabel('Total Insert Time (seconds)')
        
        # Search performance by data size
        perf_by_size = []
        for size in df['data_size'].unique():
            size_data = df[df['data_size'] == size]
            perf_by_size.append({
                'Data Size': size,
                'DB': 'Qdrant',
                'Avg Search Time': size_data['qdrant_search_per_query'].mean()
            })
            perf_by_size.append({
                'Data Size': size,
                'DB': 'ChromaDB', 
                'Avg Search Time': size_data['chroma_search_per_query'].mean()
            })
        
        perf_df = pd.DataFrame(perf_by_size)
        sns.lineplot(data=perf_df, x='Data Size', y='Avg Search Time', hue='DB', ax=axes[1,0])
        axes[1,0].set_title('Search Performance vs Data Size')
        axes[1,0].set_ylabel('Avg Search Time per Query (s)')
        
        # Search performance by vector size
        perf_by_vec = []
        for vec_size in df['vector_size'].unique():
            vec_data = df[df['vector_size'] == vec_size]
            perf_by_vec.append({
                'Vector Size': vec_size,
                'DB': 'Qdrant',
                'Avg Search Time': vec_data['qdrant_search_per_query'].mean()
            })
            perf_by_vec.append({
                'Vector Size': vec_size,
                'DB': 'ChromaDB',
                'Avg Search Time': vec_data['chroma_search_per_query'].mean()
            })
        
        vec_perf_df = pd.DataFrame(perf_by_vec)
        sns.lineplot(data=vec_perf_df, x='Vector Size', y='Avg Search Time', hue='DB', ax=axes[1,1])
        axes[1,1].set_title('Search Performance vs Vector Dimension')
        axes[1,1].set_ylabel('Avg Search Time per Query (s)')
        
        plt.tight_layout()
        plt.savefig('vector_db_comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved to 'vector_db_comprehensive_benchmark.png'")

if __name__ == "__main__":
    benchmark = VectorDBBenchmarkComplete()
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comprehensive_benchmark()
    
    if not results_df.empty:
        # Create summary report
        summary = benchmark.create_summary_report(results_df)
        
        # Create visualizations
        benchmark.create_visualizations(results_df)
        
        # Save detailed results
        results_df.to_csv('detailed_benchmark_results.csv', index=False)
        print(f"\n💾 Detailed results saved to 'detailed_benchmark_results.csv'")
        print(f"📈 Visualizations saved to 'vector_db_comprehensive_benchmark.png'")
        
        print("\n✅ Comprehensive benchmark completed successfully!")
    else:
        print("❌ No benchmark results generated")
