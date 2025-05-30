import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score
import random

class ComprehensiveVectorDBBenchmark:
    def __init__(self):
        print("🔧 Initializing comprehensive benchmark clients...")
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.chroma_client = chromadb.Client()
        
    def generate_ground_truth(self, data_size: int, query_count: int, k: int) -> Dict[int, Set[int]]:
        """Generate synthetic ground truth for relevance metrics"""
        ground_truth = {}
        for query_idx in range(query_count):
            # Generate realistic relevance: some docs are more relevant than others
            relevant_docs = set()
            
            # High relevance docs (top 20% of score range)
            high_rel_count = max(1, k // 2)
            high_rel_docs = random.sample(range(data_size), min(high_rel_count, data_size))
            relevant_docs.update(high_rel_docs)
            
            # Medium relevance docs 
            med_rel_count = max(1, k // 3)
            remaining_docs = [i for i in range(data_size) if i not in relevant_docs]
            if remaining_docs:
                med_rel_docs = random.sample(remaining_docs, min(med_rel_count, len(remaining_docs)))
                relevant_docs.update(med_rel_docs)
            
            ground_truth[query_idx] = relevant_docs
            
        return ground_truth
    
    def calculate_precision_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Precision@k"""
        if k == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Recall@k"""
        if len(relevant) == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_retrieved / len(relevant)
    
    def calculate_hit_rate_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Hit Rate@k (whether at least one relevant doc is in top-k)"""
        if len(relevant) == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return 1.0 if len(retrieved_k.intersection(relevant)) > 0 else 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate NDCG@k"""
        if len(relevant) == 0 or k == 0:
            return 0.0
        
        # Create relevance scores (1 for relevant, 0 for non-relevant)
        y_true = [1 if doc in relevant else 0 for doc in retrieved[:k]]
        
        # Pad or truncate to exactly k items
        if len(y_true) < k:
            y_true.extend([0] * (k - len(y_true)))
        else:
            y_true = y_true[:k]
        
        # Create predicted scores (decreasing order based on position)
        y_score = [1.0 - (i / k) for i in range(k)]
        
        try:
            # sklearn expects (n_samples, n_outputs) format, so we reshape
            y_true_array = np.array([y_true])
            y_score_array = np.array([y_score])
            return ndcg_score(y_true_array, y_score_array, k=k)
        except:
            # Fallback to manual calculation if sklearn fails
            return self._manual_ndcg(y_true, y_score, k)
    
    def _manual_ndcg(self, y_true: List[int], y_score: List[float], k: int) -> float:
        """Manual NDCG calculation"""
        def dcg_at_k(scores, k):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores[:k]))
        
        # Sort by predicted scores
        sorted_pairs = sorted(zip(y_score, y_true), reverse=True)
        sorted_relevance = [rel for _, rel in sorted_pairs]
        
        dcg = dcg_at_k(sorted_relevance, k)
        idcg = dcg_at_k(sorted(y_true, reverse=True), k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def benchmark_single_config(self, vector_size: int, data_size: int, k_values: List[int], 
                               query_count: int = 100) -> Dict:
        """Run benchmark for a single configuration"""
        print(f"📊 Config: {data_size} docs, {vector_size}D vectors, k={k_values}")
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducible results
        random.seed(42)
        
        vectors = np.random.random((data_size, vector_size)).astype(np.float32)
        qdrant_ids = list(range(data_size))
        chroma_ids = [f"doc_{i}" for i in range(data_size)]
        metadata = [{"category": f"cat_{i%10}", "score": random.random()} for i in range(data_size)]
        
        # Generate query vectors
        query_vectors = np.random.random((query_count, vector_size)).astype(np.float32)
        
        # Generate ground truth for largest k value
        max_k = max(k_values)
        ground_truth = self.generate_ground_truth(data_size, query_count, max_k * 2)
        
        results = {
            'vector_size': vector_size,
            'data_size': data_size,
            'query_count': query_count
        }
        
        # Benchmark Qdrant
        print("  🔍 Testing Qdrant...")
        collection_name = f"bench_qdrant_{vector_size}_{data_size}"
        
        # Setup Qdrant collection
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
            
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # Qdrant Insert Performance
        insert_start = time.time()
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
        
        qdrant_insert_time = time.time() - insert_start
        
        # Qdrant Search Performance
        search_start = time.time()
        qdrant_all_results = []
        
        for query_idx, query_vector in enumerate(query_vectors):
            search_result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=max_k
            )
            retrieved_ids = [hit.id for hit in search_result.points]
            qdrant_all_results.append(retrieved_ids)
        
        qdrant_search_time = time.time() - search_start
        qdrant_search_per_query = qdrant_search_time / query_count
        qdrant_qps = query_count / qdrant_search_time
        
        # Calculate Qdrant relevance metrics for each k
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            
            for query_idx, retrieved_ids in enumerate(qdrant_all_results):
                relevant_docs = ground_truth[query_idx]
                
                precision = self.calculate_precision_at_k(retrieved_ids, relevant_docs, k)
                recall = self.calculate_recall_at_k(retrieved_ids, relevant_docs, k)
                ndcg = self.calculate_ndcg_at_k(retrieved_ids, relevant_docs, k)
                hit_rate = self.calculate_hit_rate_at_k(retrieved_ids, relevant_docs, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                hit_rate_scores.append(hit_rate)
            
            results[f'qdrant_precision_at_{k}'] = np.mean(precision_scores)
            results[f'qdrant_recall_at_{k}'] = np.mean(recall_scores)
            results[f'qdrant_ndcg_at_{k}'] = np.mean(ndcg_scores)
            results[f'qdrant_hit_rate_at_{k}'] = np.mean(hit_rate_scores)
        
        # Store Qdrant performance metrics
        results.update({
            'qdrant_insert_time': qdrant_insert_time,
            'qdrant_search_time': qdrant_search_time,
            'qdrant_search_per_query': qdrant_search_per_query,
            'qdrant_qps': qdrant_qps,
            'qdrant_insert_throughput': data_size / qdrant_insert_time
        })
        
        # Benchmark ChromaDB
        print("  🔍 Testing ChromaDB...")
        chroma_collection_name = f"bench_chroma_{vector_size}_{data_size}"
        
        # Setup ChromaDB collection
        try:
            self.chroma_client.delete_collection(chroma_collection_name)
        except:
            pass
            
        chroma_collection = self.chroma_client.create_collection(chroma_collection_name)
        
        # ChromaDB Insert Performance
        insert_start = time.time()
        batch_size = 1000
        for i in range(0, data_size, batch_size):
            end_idx = min(i + batch_size, data_size)
            chroma_collection.add(
                embeddings=vectors[i:end_idx].tolist(),
                ids=chroma_ids[i:end_idx],
                metadatas=metadata[i:end_idx]
            )
        
        chroma_insert_time = time.time() - insert_start
        
        # ChromaDB Search Performance
        search_start = time.time()
        chroma_search_result = chroma_collection.query(
            query_embeddings=query_vectors.tolist(),
            n_results=max_k
        )
        chroma_search_time = time.time() - search_start
        chroma_search_per_query = chroma_search_time / query_count
        chroma_qps = query_count / chroma_search_time
        
        # Process ChromaDB results
        chroma_all_results = []
        for query_idx in range(query_count):
            retrieved_ids = []
            for doc_id in chroma_search_result['ids'][query_idx]:
                # Convert doc_0 -> 0
                doc_num = int(doc_id.split('_')[1])
                retrieved_ids.append(doc_num)
            chroma_all_results.append(retrieved_ids)
        
        # Calculate ChromaDB relevance metrics for each k
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            
            for query_idx, retrieved_ids in enumerate(chroma_all_results):
                relevant_docs = ground_truth[query_idx]
                
                precision = self.calculate_precision_at_k(retrieved_ids, relevant_docs, k)
                recall = self.calculate_recall_at_k(retrieved_ids, relevant_docs, k)
                ndcg = self.calculate_ndcg_at_k(retrieved_ids, relevant_docs, k)
                hit_rate = self.calculate_hit_rate_at_k(retrieved_ids, relevant_docs, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                hit_rate_scores.append(hit_rate)
            
            results[f'chroma_precision_at_{k}'] = np.mean(precision_scores)
            results[f'chroma_recall_at_{k}'] = np.mean(recall_scores)
            results[f'chroma_ndcg_at_{k}'] = np.mean(ndcg_scores)
            results[f'chroma_hit_rate_at_{k}'] = np.mean(hit_rate_scores)
        
        # Store ChromaDB performance metrics
        results.update({
            'chroma_insert_time': chroma_insert_time,
            'chroma_search_time': chroma_search_time,
            'chroma_search_per_query': chroma_search_per_query,
            'chroma_qps': chroma_qps,
            'chroma_insert_throughput': data_size / chroma_insert_time
        })
        
        print(f"    ✅ Qdrant: {qdrant_insert_time:.2f}s insert, {qdrant_qps:.1f} QPS")
        print(f"    ✅ ChromaDB: {chroma_insert_time:.2f}s insert, {chroma_qps:.1f} QPS")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark with all requested configurations"""
        print("🚀 Starting Comprehensive Vector Database Benchmark")
        print("=" * 80)
        print("📊 Performance Metrics: Insert time, Search time, Throughput (QPS)")
        print("📈 Relevance Metrics: Precision@k, Recall@k, NDCG@k, Hit Rate@k")
        print("📏 Scalability Tests: Multiple vector sizes, data sizes, k values")
        print("=" * 80)
        
        # Test configurations
        vector_sizes = [128, 256, 512, 1024]
        data_sizes = [1000, 5000, 10000, 50000]  # 1K, 5K, 10K, 50K
        k_values = [1, 5, 10, 20]
        
        all_results = []
        total_configs = len(vector_sizes) * len(data_sizes)
        current_config = 0
        
        for vector_size in vector_sizes:
            for data_size in data_sizes:
                current_config += 1
                print(f"\n[{current_config}/{total_configs}] Vector Size: {vector_size}D, Data Size: {data_size:,}")
                
                try:
                    result = self.benchmark_single_config(
                        vector_size=vector_size,
                        data_size=data_size,
                        k_values=k_values,
                        query_count=100  # 100 queries per configuration
                    )
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    continue
        
        return pd.DataFrame(all_results)
    
    def create_comprehensive_analysis(self, df: pd.DataFrame):
        """Create comprehensive analysis and visualizations"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE VECTOR DATABASE BENCHMARK ANALYSIS")
        print("=" * 80)
        
        # Performance Summary
        print("\n🚀 PERFORMANCE METRICS SUMMARY")
        print("-" * 50)
        
        perf_summary = df.groupby('data_size').agg({
            'qdrant_insert_time': 'mean',
            'chroma_insert_time': 'mean',
            'qdrant_search_per_query': 'mean',
            'chroma_search_per_query': 'mean',
            'qdrant_qps': 'mean',
            'chroma_qps': 'mean',
            'qdrant_insert_throughput': 'mean',
            'chroma_insert_throughput': 'mean'
        }).round(6)
        
        print("Average Performance by Data Size:")
        print(perf_summary.to_string())
        
        # Relevance Metrics Summary
        print("\n📈 RELEVANCE METRICS SUMMARY")
        print("-" * 50)
        
        k_values = [1, 5, 10, 20]
        
        for k in k_values:
            print(f"\n--- Top-{k} Relevance Metrics ---")
            relevance_cols = [
                f'qdrant_precision_at_{k}', f'chroma_precision_at_{k}',
                f'qdrant_recall_at_{k}', f'chroma_recall_at_{k}',
                f'qdrant_ndcg_at_{k}', f'chroma_ndcg_at_{k}',
                f'qdrant_hit_rate_at_{k}', f'chroma_hit_rate_at_{k}'
            ]
            
            relevance_summary = df[relevance_cols].mean().round(4)
            
            print(f"Precision@{k}:")
            print(f"  Qdrant:  {relevance_summary[f'qdrant_precision_at_{k}']:.4f}")
            print(f"  ChromaDB: {relevance_summary[f'chroma_precision_at_{k}']:.4f}")
            
            print(f"Recall@{k}:")
            print(f"  Qdrant:  {relevance_summary[f'qdrant_recall_at_{k}']:.4f}")
            print(f"  ChromaDB: {relevance_summary[f'chroma_recall_at_{k}']:.4f}")
            
            print(f"NDCG@{k}:")
            print(f"  Qdrant:  {relevance_summary[f'qdrant_ndcg_at_{k}']:.4f}")
            print(f"  ChromaDB: {relevance_summary[f'chroma_ndcg_at_{k}']:.4f}")
            
            print(f"Hit Rate@{k}:")
            print(f"  Qdrant:  {relevance_summary[f'qdrant_hit_rate_at_{k}']:.4f}")
            print(f"  ChromaDB: {relevance_summary[f'chroma_hit_rate_at_{k}']:.4f}")
        
        # Scalability Analysis
        print(f"\n📏 SCALABILITY ANALYSIS")
        print("-" * 50)
        
        print("Search Performance vs Data Size (avg search time per query):")
        scale_analysis = df.groupby('data_size').agg({
            'qdrant_search_per_query': 'mean',
            'chroma_search_per_query': 'mean'
        }).round(6)
        print(scale_analysis.to_string())
        
        print("\nThroughput vs Data Size (queries per second):")
        throughput_analysis = df.groupby('data_size').agg({
            'qdrant_qps': 'mean',
            'chroma_qps': 'mean'
        }).round(1)
        print(throughput_analysis.to_string())
        
        # Vector Size Impact
        print("\nPerformance vs Vector Dimension:")
        vector_analysis = df.groupby('vector_size').agg({
            'qdrant_search_per_query': 'mean',
            'chroma_search_per_query': 'mean',
            'qdrant_qps': 'mean',
            'chroma_qps': 'mean'
        }).round(6)
        print(vector_analysis.to_string())
        
        # Overall Winners
        print(f"\n🏆 OVERALL PERFORMANCE WINNERS")
        print("-" * 50)
        
        avg_qdrant_search = df['qdrant_search_per_query'].mean()
        avg_chroma_search = df['chroma_search_per_query'].mean()
        avg_qdrant_insert = df['qdrant_insert_time'].mean()
        avg_chroma_insert = df['chroma_insert_time'].mean()
        avg_qdrant_qps = df['qdrant_qps'].mean()
        avg_chroma_qps = df['chroma_qps'].mean()
        
        if avg_qdrant_search < avg_chroma_search:
            search_winner = "Qdrant"
            search_ratio = avg_chroma_search / avg_qdrant_search
        else:
            search_winner = "ChromaDB"
            search_ratio = avg_qdrant_search / avg_chroma_search
            
        if avg_qdrant_insert < avg_chroma_insert:
            insert_winner = "Qdrant"
            insert_ratio = avg_chroma_insert / avg_qdrant_insert
        else:
            insert_winner = "ChromaDB"
            insert_ratio = avg_qdrant_insert / avg_chroma_insert
            
        if avg_qdrant_qps > avg_chroma_qps:
            qps_winner = "Qdrant"
            qps_ratio = avg_qdrant_qps / avg_chroma_qps
        else:
            qps_winner = "ChromaDB"
            qps_ratio = avg_chroma_qps / avg_qdrant_qps
        
        print(f"🔍 Search Latency: {search_winner} is {search_ratio:.2f}x faster")
        print(f"💾 Insert Performance: {insert_winner} is {insert_ratio:.2f}x faster")
        print(f"⚡ Query Throughput: {qps_winner} achieves {qps_ratio:.2f}x higher QPS")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR RAG SYSTEMS")
        print("-" * 50)
        print("Based on comprehensive analysis:")
        
        if search_winner == "ChromaDB":
            print(f"✅ For query-heavy RAG workloads: ChromaDB ({search_ratio:.1f}x faster searches)")
        else:
            print(f"✅ For query-heavy RAG workloads: Qdrant ({search_ratio:.1f}x faster searches)")
            
        if insert_winner == "ChromaDB":
            print(f"✅ For document ingestion: ChromaDB ({insert_ratio:.1f}x faster inserts)")
        else:
            print(f"✅ For document ingestion: Qdrant ({insert_ratio:.1f}x faster inserts)")
        
        print(f"✅ For large-scale deployment: Consider {qps_winner} for higher throughput")
        
        return {
            'search_winner': search_winner,
            'insert_winner': insert_winner, 
            'qps_winner': qps_winner,
            'search_ratio': search_ratio,
            'insert_ratio': insert_ratio,
            'qps_ratio': qps_ratio
        }
    
    def create_advanced_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations"""
        print("\n📊 Creating advanced performance visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Search Performance Comparison
        plt.subplot(3, 3, 1)
        search_data = []
        for _, row in df.iterrows():
            search_data.extend([
                {'DB': 'Qdrant', 'Search Time (ms)': row['qdrant_search_per_query'] * 1000},
                {'DB': 'ChromaDB', 'Search Time (ms)': row['chroma_search_per_query'] * 1000}
            ])
        
        search_df = pd.DataFrame(search_data)
        sns.boxplot(data=search_df, x='DB', y='Search Time (ms)')
        plt.title('Search Latency Comparison')
        plt.ylabel('Time per Query (milliseconds)')
        
        # 2. Throughput Comparison
        plt.subplot(3, 3, 2)
        qps_data = []
        for _, row in df.iterrows():
            qps_data.extend([
                {'DB': 'Qdrant', 'QPS': row['qdrant_qps']},
                {'DB': 'ChromaDB', 'QPS': row['chroma_qps']}
            ])
        
        qps_df = pd.DataFrame(qps_data)
        sns.boxplot(data=qps_df, x='DB', y='QPS')
        plt.title('Query Throughput Comparison')
        plt.ylabel('Queries Per Second')
        
        # 3. Insert Performance
        plt.subplot(3, 3, 3)
        insert_data = []
        for _, row in df.iterrows():
            insert_data.extend([
                {'DB': 'Qdrant', 'Insert Throughput': row['qdrant_insert_throughput']},
                {'DB': 'ChromaDB', 'Insert Throughput': row['chroma_insert_throughput']}
            ])
        
        insert_df = pd.DataFrame(insert_data)
        sns.boxplot(data=insert_df, x='DB', y='Insert Throughput')
        plt.title('Insert Throughput Comparison')
        plt.ylabel('Documents Per Second')
        
        # 4. Scalability: Search Time vs Data Size
        plt.subplot(3, 3, 4)
        scale_data = []
        for _, row in df.iterrows():
            scale_data.extend([
                {'Data Size': row['data_size'], 'DB': 'Qdrant', 
                 'Search Time (ms)': row['qdrant_search_per_query'] * 1000},
                {'Data Size': row['data_size'], 'DB': 'ChromaDB', 
                 'Search Time (ms)': row['chroma_search_per_query'] * 1000}
            ])
        
        scale_df = pd.DataFrame(scale_data)
        sns.lineplot(data=scale_df, x='Data Size', y='Search Time (ms)', hue='DB')
        plt.title('Search Performance vs Data Size')
        plt.xlabel('Number of Documents')
        plt.ylabel('Search Time (ms)')
        plt.xscale('log')
        
        # 5. Vector Dimension Impact
        plt.subplot(3, 3, 5)
        dim_data = []
        for _, row in df.iterrows():
            dim_data.extend([
                {'Vector Size': row['vector_size'], 'DB': 'Qdrant', 
                 'Search Time (ms)': row['qdrant_search_per_query'] * 1000},
                {'Vector Size': row['vector_size'], 'DB': 'ChromaDB', 
                 'Search Time (ms)': row['chroma_search_per_query'] * 1000}
            ])
        
        dim_df = pd.DataFrame(dim_data)
        sns.lineplot(data=dim_df, x='Vector Size', y='Search Time (ms)', hue='DB')
        plt.title('Search Performance vs Vector Dimension')
        plt.xlabel('Vector Dimensions')
        plt.ylabel('Search Time (ms)')
        
        # 6. Precision@5 Comparison
        plt.subplot(3, 3, 6)
        precision_data = []
        for _, row in df.iterrows():
            precision_data.extend([
                {'DB': 'Qdrant', 'Precision@5': row['qdrant_precision_at_5']},
                {'DB': 'ChromaDB', 'Precision@5': row['chroma_precision_at_5']}
            ])
        
        precision_df = pd.DataFrame(precision_data)
        sns.boxplot(data=precision_df, x='DB', y='Precision@5')
        plt.title('Precision@5 Comparison')
        plt.ylabel('Precision@5')
        
        # 7. NDCG@10 Comparison
        plt.subplot(3, 3, 7)
        ndcg_data = []
        for _, row in df.iterrows():
            ndcg_data.extend([
                {'DB': 'Qdrant', 'NDCG@10': row['qdrant_ndcg_at_10']},
                {'DB': 'ChromaDB', 'NDCG@10': row['chroma_ndcg_at_10']}
            ])
        
        ndcg_df = pd.DataFrame(ndcg_data)
        sns.boxplot(data=ndcg_df, x='DB', y='NDCG@10')
        plt.title('NDCG@10 Comparison')
        plt.ylabel('NDCG@10')
        
        # 8. Hit Rate@5 vs Data Size
        plt.subplot(3, 3, 8)
        hit_rate_data = []
        for _, row in df.iterrows():
            hit_rate_data.extend([
                {'Data Size': row['data_size'], 'DB': 'Qdrant', 'Hit Rate@5': row['qdrant_hit_rate_at_5']},
                {'Data Size': row['data_size'], 'DB': 'ChromaDB', 'Hit Rate@5': row['chroma_hit_rate_at_5']}
            ])
        
        hit_rate_df = pd.DataFrame(hit_rate_data)
        sns.lineplot(data=hit_rate_df, x='Data Size', y='Hit Rate@5', hue='DB')
        plt.title('Hit Rate@5 vs Data Size')
        plt.xlabel('Number of Documents')
        plt.ylabel('Hit Rate@5')
        plt.xscale('log')
        
        # 9. Overall Performance Radar Chart Data
        plt.subplot(3, 3, 9)
        
        # Calculate normalized performance metrics (0-1 scale)
        metrics = ['Search Speed', 'Insert Speed', 'Precision@5', 'NDCG@10', 'Hit Rate@5']
        
        qdrant_scores = [
            1 - df['qdrant_search_per_query'].mean() / max(df['qdrant_search_per_query'].max(), df['chroma_search_per_query'].max()),
            1 - df['qdrant_insert_time'].mean() / max(df['qdrant_insert_time'].max(), df['chroma_insert_time'].max()),
            df['qdrant_precision_at_5'].mean(),
            df['qdrant_ndcg_at_10'].mean(),
            df['qdrant_hit_rate_at_5'].mean()
        ]
        
        chroma_scores = [
            1 - df['chroma_search_per_query'].mean() / max(df['qdrant_search_per_query'].max(), df['chroma_search_per_query'].max()),
            1 - df['chroma_insert_time'].mean() / max(df['qdrant_insert_time'].max(), df['chroma_insert_time'].max()),
            df['chroma_precision_at_5'].mean(),
            df['chroma_ndcg_at_10'].mean(),
            df['chroma_hit_rate_at_5'].mean()
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, qdrant_scores, width, label='Qdrant', alpha=0.8)
        plt.bar(x + width/2, chroma_scores, width, label='ChromaDB', alpha=0.8)
        
        plt.xlabel('Performance Metrics')
        plt.ylabel('Normalized Score (0-1)')
        plt.title('Overall Performance Comparison')
        plt.xticks(x, metrics, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('vector_db_comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
        print("✅ Comprehensive visualizations saved!")

if __name__ == "__main__":
    benchmark = ComprehensiveVectorDBBenchmark()
    
    # Run comprehensive benchmark
    print("⚠️  Warning: This comprehensive benchmark will take significant time!")
    print("📊 Testing configurations: 4 vector sizes × 4 data sizes = 16 total configurations")
    print("🔍 Each configuration tests multiple k values and runs 100 queries")
    
    results_df = benchmark.run_comprehensive_benchmark()
    
    if not results_df.empty:
        # Create comprehensive analysis
        analysis = benchmark.create_comprehensive_analysis(results_df)
        
        # Create advanced visualizations
        benchmark.create_advanced_visualizations(results_df)
        
        # Save detailed results
        results_df.to_csv('detailed_benchmark_results.csv', index=False)
        print(f"\n💾 Detailed results saved to 'detailed_benchmark_results.csv'")
        print(f"📊 Comprehensive visualizations saved to 'vector_db_comprehensive_benchmark.png'")
        
        print("\n✅ Comprehensive benchmark with all metrics completed successfully!")
        print("\n📋 Summary:")
        print(f"   🔍 Search Winner: {analysis['search_winner']} ({analysis['search_ratio']:.1f}x faster)")
        print(f"   💾 Insert Winner: {analysis['insert_winner']} ({analysis['insert_ratio']:.1f}x faster)")
        print(f"   ⚡ Throughput Winner: {analysis['qps_winner']} ({analysis['qps_ratio']:.1f}x higher QPS)")
        
    else:
        print("❌ No benchmark results generated")
