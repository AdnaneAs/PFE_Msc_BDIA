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
from sentence_transformers import SentenceTransformer
import torch

class RealisticBGEBenchmark:
    def __init__(self):
        print("🔧 Initializing BGE-M3 benchmark...")
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.chroma_client = chromadb.Client()
        
        # Initialize BGE-M3 model
        print("📚 Loading BGE-M3 model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.vector_size = 1024  # BGE-M3 produces 1024-dimensional embeddings
        print(f"✅ BGE-M3 loaded (embedding dimension: {self.vector_size})")
        
    def generate_realistic_documents(self, data_size: int) -> List[str]:
        """Generate realistic document corpus for different domains"""
        domains = {
            'technology': [
                "artificial intelligence machine learning algorithms",
                "cloud computing infrastructure scalability",
                "cybersecurity data protection encryption",
                "software development programming languages",
                "database management systems optimization",
                "neural networks deep learning models",
                "distributed systems microservices architecture",
                "web development frameworks APIs",
                "mobile applications user interface design",
                "data science analytics visualization"
            ],
            'business': [
                "market analysis competitive strategy",
                "financial planning investment portfolio",
                "project management agile methodologies",
                "customer relationship management sales",
                "supply chain logistics operations",
                "human resources talent acquisition",
                "digital transformation business processes",
                "risk management compliance regulations",
                "entrepreneurship startup funding",
                "marketing campaigns brand strategy"
            ],
            'science': [
                "quantum physics particle mechanics",
                "climate change environmental impact",
                "biotechnology genetic engineering",
                "space exploration planetary science",
                "renewable energy sustainable technology",
                "medical research pharmaceutical development",
                "chemical reactions molecular structure",
                "materials science nanotechnology",
                "biological systems ecosystem analysis",
                "mathematical modeling statistical analysis"
            ],
            'health': [
                "nutrition dietary supplements wellness",
                "mental health psychological therapy",
                "fitness exercise physical training",
                "medical diagnostics treatment protocols",
                "preventive care health screening",
                "pharmaceutical drugs side effects",
                "surgical procedures recovery methods",
                "chronic disease management",
                "immunology vaccine development",
                "public health epidemiology"
            ]
        }
        
        documents = []
        domain_keys = list(domains.keys())
        
        for i in range(data_size):
            domain = domain_keys[i % len(domain_keys)]
            base_topics = domains[domain]
            topic = base_topics[i % len(base_topics)]
            
            # Create variations of the topic
            variations = [
                f"Introduction to {topic} and its applications in modern industry",
                f"Advanced techniques in {topic} for professional development", 
                f"Comprehensive guide to {topic} with practical examples",
                f"Latest research developments in {topic} and future trends",
                f"Best practices for implementing {topic} in enterprise environments",
                f"Case study analysis of {topic} in real-world scenarios",
                f"Comparative study of {topic} methodologies and approaches",
                f"Technical documentation for {topic} with detailed specifications"
            ]
            
            doc_text = variations[i % len(variations)]
            documents.append(doc_text)
            
        return documents
    
    def generate_realistic_queries(self, documents: List[str], query_count: int) -> Tuple[List[str], Dict[int, Set[int]]]:
        """Generate realistic queries with semantic relevance to documents"""
        queries = []
        ground_truth = {}
        
        # Create query templates that will have semantic similarity to documents
        query_patterns = [
            "How to implement {}?",
            "What are the benefits of {}?", 
            "Best practices for {}",
            "Introduction to {}",
            "Advanced {} techniques",
            "Recent developments in {}",
            "Comparison of {} methods",
            "Guide to {} implementation"
        ]
        
        # Extract key terms from documents
        doc_keywords = []
        for doc in documents:
            words = doc.lower().split()
            # Extract meaningful terms (longer than 3 characters)
            keywords = [w for w in words if len(w) > 3 and w not in ['with', 'and', 'the', 'for', 'in', 'of', 'to']]
            doc_keywords.append(keywords)
        
        for query_idx in range(query_count):
            # Select a random document as basis for query
            base_doc_idx = random.randint(0, len(documents) - 1)
            base_keywords = doc_keywords[base_doc_idx]
            
            if base_keywords:
                # Pick 1-2 keywords from the base document
                num_keywords = min(2, len(base_keywords))
                selected_keywords = random.sample(base_keywords, num_keywords)
                keyword_phrase = " ".join(selected_keywords)
                
                # Create query using template
                template = random.choice(query_patterns)
                query = template.format(keyword_phrase)
                queries.append(query)
                
                # Create ground truth: documents with semantic similarity
                relevant_docs = set()
                
                # Base document is always relevant
                relevant_docs.add(base_doc_idx)
                
                # Find other semantically similar documents
                for doc_idx, doc_keys in enumerate(doc_keywords):
                    if doc_idx != base_doc_idx:
                        # Check for keyword overlap
                        overlap = len(set(selected_keywords) & set(doc_keys))
                        if overlap > 0:
                            relevant_docs.add(doc_idx)
                
                # Ensure we have some relevant documents
                if len(relevant_docs) < 3:
                    # Add some random documents from same domain
                    domain_docs = [i for i in range(len(documents)) if i % 4 == base_doc_idx % 4]
                    additional = random.sample([d for d in domain_docs if d not in relevant_docs], 
                                             min(2, len([d for d in domain_docs if d not in relevant_docs])))
                    relevant_docs.update(additional)
                    
                ground_truth[query_idx] = relevant_docs
            else:
                # Fallback query
                queries.append("information technology systems")
                ground_truth[query_idx] = {base_doc_idx}
        
        return queries, ground_truth
    
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
        """Calculate Hit Rate@k"""
        if len(relevant) == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return 1.0 if len(retrieved_k.intersection(relevant)) > 0 else 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate NDCG@k with proper relevance scoring"""
        if len(relevant) == 0 or k == 0:
            return 0.0
        
        # Create relevance scores (1 for relevant, 0 for non-relevant)
        y_true = [1 if doc in relevant else 0 for doc in retrieved[:k]]
        
        # Pad to exactly k items if needed
        if len(y_true) < k:
            y_true.extend([0] * (k - len(y_true)))
        else:
            y_true = y_true[:k]
        
        # Create predicted scores (decreasing by position)
        y_score = [1.0 - (i / k) for i in range(k)]
        
        try:
            y_true_array = np.array([y_true])
            y_score_array = np.array([y_score])
            return ndcg_score(y_true_array, y_score_array, k=k)
        except:
            return self._manual_ndcg(y_true, y_score, k)
    
    def _manual_ndcg(self, y_true: List[int], y_score: List[float], k: int) -> float:
        """Manual NDCG calculation"""
        def dcg_at_k(scores, k):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores[:k]))
        
        dcg = dcg_at_k(y_true, k)
        idcg = dcg_at_k(sorted(y_true, reverse=True), k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def benchmark_configuration(self, data_size: int, query_count: int = 100) -> Dict:
        """Benchmark single configuration with BGE-M3 embeddings"""
        print(f"📊 Benchmarking: {data_size} documents, {query_count} queries (BGE-M3)")
        
        # Generate realistic documents and queries
        print("  📝 Generating realistic documents...")
        documents = self.generate_realistic_documents(data_size)
        
        print("  🔍 Generating semantic queries...")
        queries, ground_truth = self.generate_realistic_queries(documents, query_count)
        
        # Generate embeddings
        print("  🧠 Computing BGE-M3 embeddings...")
        doc_embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        query_embeddings = self.embedding_model.encode(queries, convert_to_numpy=True)
        
        print(f"  ✅ Generated {len(doc_embeddings)} doc embeddings, {len(query_embeddings)} query embeddings")
        
        # Test configurations
        k_values = [1, 5, 10, 20]
        max_k = max(k_values)
        
        results = {
            'vector_size': self.vector_size,
            'data_size': data_size,
            'query_count': query_count
        }
        
        # Benchmark Qdrant
        print("  🔧 Testing Qdrant...")
        collection_name = f"bge_qdrant_{data_size}"
        
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
            
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        
        # Qdrant Insert
        insert_start = time.time()
        batch_size = 100  # Smaller batches for embeddings
        for i in range(0, data_size, batch_size):
            end_idx = min(i + batch_size, data_size)
            points = [
                PointStruct(
                    id=j,
                    vector=doc_embeddings[j].tolist(),
                    payload={"text": documents[j], "doc_id": j}
                )
                for j in range(i, end_idx)
            ]
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
        
        qdrant_insert_time = time.time() - insert_start
        qdrant_insert_throughput = data_size / qdrant_insert_time
        
        # Qdrant Search
        search_start = time.time()
        qdrant_all_results = []
        
        for query_embedding in query_embeddings:
            search_result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding.tolist(),
                limit=max_k
            )
            retrieved_ids = [hit.id for hit in search_result.points]
            qdrant_all_results.append(retrieved_ids)
        
        qdrant_search_time = time.time() - search_start
        qdrant_search_per_query = qdrant_search_time / query_count
        qdrant_qps = query_count / qdrant_search_time
        
        # Calculate Qdrant metrics
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
        
        results['qdrant_insert_time'] = qdrant_insert_time
        results['qdrant_search_time'] = qdrant_search_time
        results['qdrant_search_per_query'] = qdrant_search_per_query
        results['qdrant_qps'] = qdrant_qps
        results['qdrant_insert_throughput'] = qdrant_insert_throughput
        
        # Benchmark ChromaDB
        print("  🔧 Testing ChromaDB...")
        try:
            self.chroma_client.delete_collection(f"bge_chroma_{data_size}")
        except:
            pass
            
        chroma_collection = self.chroma_client.create_collection(f"bge_chroma_{data_size}")
        
        # ChromaDB Insert
        insert_start = time.time()
        batch_size = 100
        for i in range(0, data_size, batch_size):
            end_idx = min(i + batch_size, data_size)
            chroma_collection.add(
                embeddings=doc_embeddings[i:end_idx].tolist(),
                ids=[f"doc_{j}" for j in range(i, end_idx)],
                metadatas=[{"text": documents[j]} for j in range(i, end_idx)]
            )
        
        chroma_insert_time = time.time() - insert_start
        chroma_insert_throughput = data_size / chroma_insert_time
        
        # ChromaDB Search
        search_start = time.time()
        chroma_search_result = chroma_collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=max_k
        )
        chroma_search_time = time.time() - search_start
        chroma_search_per_query = chroma_search_time / query_count
        chroma_qps = query_count / chroma_search_time
        
        # Process ChromaDB results
        chroma_all_results = []
        for query_idx in range(query_count):
            retrieved_ids = [int(doc_id.split('_')[1]) for doc_id in chroma_search_result['ids'][query_idx]]
            chroma_all_results.append(retrieved_ids)
        
        # Calculate ChromaDB metrics
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
        
        results['chroma_insert_time'] = chroma_insert_time
        results['chroma_search_time'] = chroma_search_time
        results['chroma_search_per_query'] = chroma_search_per_query
        results['chroma_qps'] = chroma_qps
        results['chroma_insert_throughput'] = chroma_insert_throughput
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark with different data sizes"""
        print("🚀 Starting Comprehensive BGE-M3 Vector Database Benchmark")
        print("=" * 70)
        
        # Test configurations: smaller sizes for BGE-M3 due to computational cost
        data_sizes = [500, 1000, 2500, 5000]
        
        all_results = []
        
        for i, data_size in enumerate(data_sizes, 1):
            print(f"\n[{i}/{len(data_sizes)}] Testing {data_size} documents...")
            try:
                result = self.benchmark_configuration(data_size)
                all_results.append(result)
                print(f"  ✅ Completed {data_size} documents")
            except Exception as e:
                print(f"  ❌ Failed {data_size} documents: {e}")
                continue
        
        return pd.DataFrame(all_results)
    
    def create_analysis_report(self, df: pd.DataFrame):
        """Create comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("📊 BGE-M3 VECTOR DATABASE BENCHMARK RESULTS")
        print("=" * 70)
        
        if df.empty:
            print("❌ No results to analyze")
            return
        
        # Performance Summary
        print("\n🏆 PERFORMANCE SUMMARY")
        print("-" * 50)
        
        # Average metrics
        metrics = ['precision_at_10', 'recall_at_10', 'ndcg_at_10', 'hit_rate_at_10', 
                  'search_per_query', 'insert_time', 'qps']
        
        for metric in metrics:
            qdrant_col = f'qdrant_{metric}'
            chroma_col = f'chroma_{metric}'
            
            if qdrant_col in df.columns and chroma_col in df.columns:
                qdrant_avg = df[qdrant_col].mean()
                chroma_avg = df[chroma_col].mean()
                
                if 'time' in metric:
                    winner = "Qdrant" if qdrant_avg < chroma_avg else "ChromaDB"
                    ratio = max(qdrant_avg, chroma_avg) / min(qdrant_avg, chroma_avg)
                    print(f"{metric.replace('_', ' ').title()}: {winner} ({ratio:.2f}x faster)")
                else:
                    winner = "Qdrant" if qdrant_avg > chroma_avg else "ChromaDB"
                    print(f"{metric.replace('_', ' ').title()}: {winner} ({qdrant_avg:.4f} vs {chroma_avg:.4f})")
        
        # Relevance Quality Analysis
        print(f"\n📈 RELEVANCE QUALITY ANALYSIS")
        print("-" * 50)
        
        for k in [1, 5, 10, 20]:
            print(f"\nTop-{k} Results:")
            qdrant_ndcg = df[f'qdrant_ndcg_at_{k}'].mean()
            chroma_ndcg = df[f'chroma_ndcg_at_{k}'].mean()
            qdrant_precision = df[f'qdrant_precision_at_{k}'].mean()
            chroma_precision = df[f'chroma_precision_at_{k}'].mean()
            
            print(f"  NDCG@{k}:      Qdrant={qdrant_ndcg:.4f}, ChromaDB={chroma_ndcg:.4f}")
            print(f"  Precision@{k}: Qdrant={qdrant_precision:.4f}, ChromaDB={chroma_precision:.4f}")
        
        # Scaling Analysis
        print(f"\n📊 SCALING ANALYSIS")
        print("-" * 50)
        
        if len(df) > 1:
            small_data = df[df['data_size'] <= 1000]
            large_data = df[df['data_size'] > 1000]
            
            if not small_data.empty and not large_data.empty:
                small_qdrant_qps = small_data['qdrant_qps'].mean()
                large_qdrant_qps = large_data['qdrant_qps'].mean()
                small_chroma_qps = small_data['chroma_qps'].mean()
                large_chroma_qps = large_data['chroma_qps'].mean()
                
                print(f"Small datasets (≤1K docs):")
                print(f"  Qdrant QPS: {small_qdrant_qps:.1f}")
                print(f"  ChromaDB QPS: {small_chroma_qps:.1f}")
                print(f"Large datasets (>1K docs):")
                print(f"  Qdrant QPS: {large_qdrant_qps:.1f}")
                print(f"  ChromaDB QPS: {large_chroma_qps:.1f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR PRODUCTION")
        print("-" * 50)
        
        avg_qdrant_ndcg = df['qdrant_ndcg_at_10'].mean()
        avg_chroma_ndcg = df['chroma_ndcg_at_10'].mean()
        avg_qdrant_qps = df['qdrant_qps'].mean()
        avg_chroma_qps = df['chroma_qps'].mean()
        
        if avg_qdrant_ndcg > avg_chroma_ndcg and avg_qdrant_qps > avg_chroma_qps:
            print("✅ QDRANT recommended: Better relevance quality AND performance")
        elif avg_chroma_ndcg > avg_qdrant_ndcg and avg_chroma_qps > avg_qdrant_qps:
            print("✅ CHROMADB recommended: Better relevance quality AND performance")
        else:
            if avg_qdrant_ndcg > avg_chroma_ndcg:
                print("✅ QDRANT recommended: Better relevance quality (more important for RAG)")
            else:
                print("✅ CHROMADB recommended: Better relevance quality (more important for RAG)")
        
        print(f"\n🔧 Integration Considerations:")
        print(f"   • Qdrant: Production-ready, REST/gRPC API, horizontal scaling")
        print(f"   • ChromaDB: Python-native, simpler deployment, good for prototyping")
        print(f"   • BGE-M3 embeddings provide high-quality semantic similarity")
        print(f"   • Consider your scale: <10K docs=ChromaDB, >10K docs=Qdrant")

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import sentence_transformers
        print("✅ sentence-transformers available")
    except ImportError:
        print("❌ Please install: pip install sentence-transformers")
        exit(1)
    
    benchmark = RealisticBGEBenchmark()
    
    # Run benchmark
    results_df = benchmark.run_comprehensive_benchmark()
    
    if not results_df.empty:
        # Create analysis report
        benchmark.create_analysis_report(results_df)
        
        # Save results
        results_df.to_csv('bge_benchmark_results.csv', index=False)
        print(f"\n💾 Results saved to 'bge_benchmark_results.csv'")
        print("\n✅ BGE-M3 Comprehensive Benchmark completed!")
    else:
        print("❌ No benchmark results generated")
