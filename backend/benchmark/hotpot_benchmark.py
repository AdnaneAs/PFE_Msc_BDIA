"""
Enhanced Vector Database Benchmark: Qdrant vs ChromaDB using HotpotQA Dataset
============================================================================

A comprehensive, realistic benchmark comparing Qdrant and ChromaDB vector databases
for Retrieval-Augmented Generation (RAG) systems using the HotpotQA dataset.

HotpotQA is a dataset with 113k Wikipedia-based question-answer pairs that require
reasoning over multiple supporting documents to answer correctly.

Date: May 30, 2025
Version: 3.0 - HotpotQA Dataset Integration

HotpotQA Dataset: https://hotpotqa.github.io/
Paper: https://arxiv.org/abs/1809.09600
"""

import time
import numpy as np
import pandas as pd
import random
import json
import hashlib
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime
import logging
import os
import traceback
import requests
from urllib.parse import unquote

# Vector database imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb

# Embedding models
from sentence_transformers import SentenceTransformer

# Metrics and visualization
from sklearn.metrics import ndcg_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace datasets
from datasets import load_dataset

class HotpotQAVectorDBBenchmark:
    """
    Comprehensive benchmark using HotpotQA dataset for vector database evaluation.
    
    HotpotQA contains complex multi-hop questions that require reasoning across
    multiple Wikipedia articles, making it ideal for testing RAG system performance.
    
    Key Features:
    - 113k question-answer pairs from Wikipedia
    - Multi-hop reasoning requirements
    - Supporting facts for each question
    - Hard and medium difficulty levels
    - Diverse question types (comparison, bridge, etc.)
    """
    
    def __init__(self):
        """Initialize the HotpotQA benchmark."""
        self.setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("HOTPOTQA VECTOR DATABASE BENCHMARK - QDRANT VS CHROMADB")
        self.logger.info("=" * 80)
        
        # Initialize database clients
        self.logger.info("[INIT] Initializing database clients...")
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.chroma_client = chromadb.PersistentClient(path="../db_data")
        
        # Load embedding models
        self.models = self._load_embedding_models()
        
        # Load HotpotQA dataset
        self.logger.info("[HOTPOT] Loading HotpotQA dataset...")
        self.hotpot_data = self._load_hotpot_dataset()
        
        # Configuration
        self.max_questions = 100  # Limit for faster testing, can be increased
        self.chunk_size = 300  # Characters per chunk
        self.overlap_size = 50  # Overlap between chunks
        
        self.logger.info(f"HotpotQA dataset loaded: {len(self.hotpot_data)} questions")
        self.logger.info(f"Max questions for benchmark: {self.max_questions}")

    def setup_logging(self):
        """Configure logging with UTF-8 encoding and clear formatting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"hotpot_benchmark_{timestamp}.log"
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_embedding_models(self) -> Dict[str, SentenceTransformer]:
        """Load both BGE-M3 and all-MiniLM-L6-v2 embedding models."""
        self.logger.info("[MODELS] Loading embedding models...")
        models = {}
        
        try:
            # Load BGE-M3 (best quality)
            self.logger.info("  Loading BGE-M3 (BAAI/bge-m3)...")
            models['bge-m3'] = SentenceTransformer('BAAI/bge-m3')
            self.logger.info(f"    [OK] BGE-M3 loaded (dimension: {models['bge-m3'].get_sentence_embedding_dimension()})")
            
            # Load all-MiniLM-L6-v2 (faster)
            self.logger.info("  Loading all-MiniLM-L6-v2...")
            models['all-minilm'] = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info(f"    [OK] all-MiniLM-L6-v2 loaded (dimension: {models['all-minilm'].get_sentence_embedding_dimension()})")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load embedding models: {e}")
            raise
            
        return models
    
    def _load_hotpot_dataset(self) -> pd.DataFrame:
        """Load the HotpotQA dataset from HuggingFace."""
        try:
            self.logger.info("  Downloading HotpotQA dataset from HuggingFace...")
            # Load the validation set (dev) which has supporting facts
            dataset = load_dataset("hotpot_qa", "distractor")
            
            # Convert to pandas DataFrame
            hotpot_df = pd.DataFrame(dataset['validation'])
            
            self.logger.info(f"  [OK] HotpotQA dataset loaded: {len(hotpot_df)} questions")
            self.logger.info(f"  Columns: {list(hotpot_df.columns)}")
            
            # Display sample question for verification
            sample_question = hotpot_df.iloc[0]
            self.logger.info(f"  Sample question: {sample_question['question'][:100]}...")
            self.logger.info(f"  Sample answer: {sample_question['answer']}")
            self.logger.info(f"  Question type: {sample_question['type']}")
            self.logger.info(f"  Difficulty level: {sample_question['level']}")
            
            return hotpot_df
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load HotpotQA dataset: {e}")
            raise

    def prepare_hotpot_corpus(self, max_questions: int = None) -> Tuple[List[str], List[str], Dict]:
        """
        Prepare the HotpotQA corpus for vector database indexing.
        
        Args:
            max_questions: Maximum number of questions to process (for testing)
            
        Returns:
            corpus: List of text chunks from supporting documents
            queries: List of HotpotQA questions
            ground_truth: Dictionary mapping query indices to relevant document indices
        """
        self.logger.info("[CORPUS] Preparing HotpotQA corpus...")
        
        if max_questions:
            hotpot_subset = self.hotpot_data.head(max_questions)
        else:
            hotpot_subset = self.hotpot_data
            
        corpus = []
        queries = []
        ground_truth = {}
        
        # Track document content to avoid duplicates
        seen_content = set()
        content_to_idx = {}
        
        for question_idx, row in hotpot_subset.iterrows():
            question = row['question']
            context = row['context']  # List of [title, sentences] pairs
            supporting_facts = row['supporting_facts']  # List of [title, sentence_idx] pairs
            
            queries.append(question)
            ground_truth[len(queries) - 1] = []
              # Process supporting documents
            doc_titles = context['title']
            doc_sentences = context['sentences']
            
            for doc_idx, (doc_title, sentences) in enumerate(zip(doc_titles, doc_sentences)):
                # Create chunks from sentences
                for sent_idx, sentence in enumerate(sentences):
                    if sentence.strip():  # Skip empty sentences
                        # Create chunk content
                        chunk_content = f"Title: {doc_title}\n{sentence}"
                        
                        # Check if this content already exists
                        content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            content_to_idx[content_hash] = len(corpus)
                            corpus.append(chunk_content)
                        
                        chunk_idx = content_to_idx[content_hash]
                        
                        # Check if this chunk is a supporting fact
                        supporting_titles = supporting_facts['title']
                        supporting_sent_ids = supporting_facts['sent_id']
                        
                        for support_idx, (support_title, support_sent_idx) in enumerate(zip(supporting_titles, supporting_sent_ids)):
                            if support_title == doc_title and support_sent_idx == sent_idx:
                                ground_truth[len(queries) - 1].append(chunk_idx)
                                break
        
        self.logger.info(f"  [OK] Corpus prepared:")
        self.logger.info(f"    Questions: {len(queries)}")
        self.logger.info(f"    Document chunks: {len(corpus)}")
        self.logger.info(f"    Average supporting facts per question: {np.mean([len(facts) for facts in ground_truth.values()]):.1f}")
        
        return corpus, queries, ground_truth

    def create_vector_collections(self, corpus: List[str], model_name: str, model: SentenceTransformer):
        """Create vector collections in both Qdrant and ChromaDB."""
        self.logger.info(f"[VECTORS] Creating collections for {model_name}...")
        
        # Generate embeddings
        self.logger.info(f"  Generating embeddings for {len(corpus)} documents...")
        start_time = time.time()
        embeddings = model.encode(corpus, show_progress_bar=True)
        embedding_time = time.time() - start_time
        self.logger.info(f"  [OK] Embeddings generated in {embedding_time:.2f}s")
        
        collection_name = f"hotpot_{model_name}_{len(corpus)}"
        
        # Create Qdrant collection
        self.logger.info(f"  Creating Qdrant collection: {collection_name}")
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
            
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
        )
          # Insert into Qdrant in batches
        batch_size = 100  # Smaller batches to avoid payload size limit
        for i in range(0, len(corpus), batch_size):
            batch_end = min(i + batch_size, len(corpus))
            batch_points = [
                PointStruct(id=j, vector=embeddings[j].tolist(), payload={"text": corpus[j]})
                for j in range(i, batch_end)
            ]
            self.qdrant_client.upsert(collection_name=collection_name, points=batch_points)
        
        # Create ChromaDB collection
        self.logger.info(f"  Creating ChromaDB collection: {collection_name}")
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
            
        chroma_collection = self.chroma_client.create_collection(collection_name)
        chroma_collection.add(
            embeddings=embeddings.tolist(),
            documents=corpus,
            ids=[str(i) for i in range(len(corpus))]
        )
        
        return collection_name, embeddings

    def run_retrieval_benchmark(self, queries: List[str], ground_truth: Dict, 
                              collection_name: str, model: SentenceTransformer, 
                              model_name: str) -> Dict:
        """Run retrieval benchmark on both vector databases."""
        self.logger.info(f"[BENCHMARK] Running retrieval tests for {model_name}...")
        
        results = {
            'model': model_name,
            'queries_count': len(queries),
            'qdrant': {},
            'chromadb': {}
        }
        
        k_values = [1, 3, 5, 10]
        
        # Test Qdrant
        self.logger.info(f"  Testing Qdrant performance...")
        qdrant_times = []
        qdrant_results = {k: [] for k in k_values}
        
        for i, query in enumerate(queries):
            query_embedding = model.encode([query])[0]
            
            start_time = time.time()
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=max(k_values)
            )
            query_time = time.time() - start_time
            qdrant_times.append(query_time)
            
            retrieved_ids = [hit.id for hit in search_result]
            
            # Calculate metrics for different k values
            relevant_docs = set(ground_truth.get(i, []))
            for k in k_values:
                retrieved_k = set(retrieved_ids[:k])
                
                # Precision@k
                precision = len(retrieved_k & relevant_docs) / len(retrieved_k) if retrieved_k else 0
                
                # Recall@k
                recall = len(retrieved_k & relevant_docs) / len(relevant_docs) if relevant_docs else 0
                
                # Hit Rate@k (whether any relevant document is in top-k)
                hit_rate = 1 if (retrieved_k & relevant_docs) else 0
                
                qdrant_results[k].append({
                    'precision': precision,
                    'recall': recall,
                    'hit_rate': hit_rate
                })
        
        # Test ChromaDB
        self.logger.info(f"  Testing ChromaDB performance...")
        chroma_collection = self.chroma_client.get_collection(collection_name)
        chroma_times = []
        chroma_results = {k: [] for k in k_values}
        
        for i, query in enumerate(queries):
            query_embedding = model.encode([query])[0]
            
            start_time = time.time()
            search_result = chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max(k_values)
            )
            query_time = time.time() - start_time
            chroma_times.append(query_time)
            
            retrieved_ids = [int(id_str) for id_str in search_result['ids'][0]]
            
            # Calculate metrics for different k values
            relevant_docs = set(ground_truth.get(i, []))
            for k in k_values:
                retrieved_k = set(retrieved_ids[:k])
                
                # Precision@k
                precision = len(retrieved_k & relevant_docs) / len(retrieved_k) if retrieved_k else 0
                
                # Recall@k
                recall = len(retrieved_k & relevant_docs) / len(relevant_docs) if relevant_docs else 0
                
                # Hit Rate@k
                hit_rate = 1 if (retrieved_k & relevant_docs) else 0
                
                chroma_results[k].append({
                    'precision': precision,
                    'recall': recall,
                    'hit_rate': hit_rate
                })
        
        # Aggregate results
        for db_name, db_results, db_times in [
            ('qdrant', qdrant_results, qdrant_times),
            ('chromadb', chroma_results, chroma_times)
        ]:
            results[db_name] = {
                'avg_query_time': np.mean(db_times),
                'total_time': np.sum(db_times),
                'queries_per_second': len(queries) / np.sum(db_times)
            }
            
            for k in k_values:
                metrics = db_results[k]
                results[db_name][f'precision@{k}'] = np.mean([m['precision'] for m in metrics])
                results[db_name][f'recall@{k}'] = np.mean([m['recall'] for m in metrics])
                results[db_name][f'hit_rate@{k}'] = np.mean([m['hit_rate'] for m in metrics])
                
                # Calculate NDCG@k
                y_true = []
                y_scores = []
                for i in range(len(queries)):
                    relevant_docs = set(ground_truth.get(i, []))
                    if db_name == 'qdrant':
                        query_embedding = model.encode([queries[i]])[0]
                        search_result = self.qdrant_client.search(
                            collection_name=collection_name,
                            query_vector=query_embedding.tolist(),
                            limit=k
                        )
                        retrieved_ids = [hit.id for hit in search_result]
                        scores = [hit.score for hit in search_result]
                    else:
                        query_embedding = model.encode([queries[i]])[0]
                        search_result = chroma_collection.query(
                            query_embeddings=[query_embedding.tolist()],
                            n_results=k
                        )
                        retrieved_ids = [int(id_str) for id_str in search_result['ids'][0]]
                        scores = [1 - dist for dist in search_result['distances'][0]]  # Convert distance to similarity
                    
                    # Create relevance scores
                    relevance_scores = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_ids]
                    
                    # Pad to length k if needed
                    while len(relevance_scores) < k:
                        relevance_scores.append(0)
                        scores.append(0)
                    
                    y_true.append(relevance_scores[:k])
                    y_scores.append(scores[:k])
                  # Calculate NDCG@k
                if y_true and any(any(row) for row in y_true):
                    try:
                        # Ensure we have more than 1 query for NDCG calculation
                        if len(y_true) > 1:
                            ndcg = ndcg_score(y_true, y_scores)
                        else:
                            # For single query, calculate manually
                            if len(y_true[0]) > 0 and max(y_true[0]) > 0:
                                # Simple relevance score for single query
                                ndcg = sum(y_true[0][i] * y_scores[0][i] for i in range(len(y_true[0]))) / sum(y_true[0])
                            else:
                                ndcg = 0.0
                        results[db_name][f'ndcg@{k}'] = ndcg
                    except Exception as e:
                        self.logger.warning(f"NDCG calculation failed for {db_name}@{k}: {e}")
                        results[db_name][f'ndcg@{k}'] = 0.0
                else:
                    results[db_name][f'ndcg@{k}'] = 0.0
        
        return results

    def run_full_benchmark(self):
        """Run the complete HotpotQA benchmark."""
        self.logger.info("[START] Beginning comprehensive HotpotQA benchmark...")
        
        all_results = []
        
        # Prepare corpus
        corpus, queries, ground_truth = self.prepare_hotpot_corpus(self.max_questions)
        
        # Test each embedding model
        for model_name, model in self.models.items():
            self.logger.info(f"\n[MODEL] Testing {model_name}...")
            
            # Create vector collections
            collection_name, embeddings = self.create_vector_collections(corpus, model_name, model)
            
            # Run benchmark
            results = self.run_retrieval_benchmark(queries, ground_truth, collection_name, model, model_name)
            all_results.append(results)
            
            # Log results
            self.logger.info(f"\n[RESULTS] {model_name} Performance Summary:")
            self.logger.info("-" * 60)
            
            for db in ['qdrant', 'chromadb']:
                self.logger.info(f"\n{db.upper()}:")
                self.logger.info(f"  Avg Query Time: {results[db]['avg_query_time']*1000:.2f}ms")
                self.logger.info(f"  Queries/Second: {results[db]['queries_per_second']:.1f}")
                self.logger.info(f"  NDCG@10: {results[db]['ndcg@10']:.3f}")
                self.logger.info(f"  Precision@10: {results[db]['precision@10']:.3f}")
                self.logger.info(f"  Recall@10: {results[db]['recall@10']:.3f}")
                self.logger.info(f"  Hit Rate@10: {results[db]['hit_rate@10']:.3f}")
        
        # Save detailed results
        self._save_results(all_results)
        
        # Generate visualizations
        self._create_visualizations(all_results)
        
        # Generate analysis report
        self._generate_analysis_report(all_results)
        
        self.logger.info("\n[COMPLETE] HotpotQA benchmark completed successfully!")
        return all_results

    def _save_results(self, results: List[Dict]):
        """Save benchmark results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed CSV
        rows = []
        for result in results:
            model = result['model']
            for db in ['qdrant', 'chromadb']:
                row = {
                    'model': model,
                    'database': db,
                    'queries_count': result['queries_count'],
                    'avg_query_time_ms': result[db]['avg_query_time'] * 1000,
                    'queries_per_second': result[db]['queries_per_second']
                }
                
                # Add all metric columns
                for key, value in result[db].items():
                    if key not in ['avg_query_time', 'total_time', 'queries_per_second']:
                        row[key] = value
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_filename = f"hotpot_benchmark_results_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"[SAVE] Results saved to {csv_filename}")
        
        # Save raw JSON
        json_filename = f"hotpot_benchmark_raw_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[SAVE] Raw results saved to {json_filename}")

    def _create_visualizations(self, results: List[Dict]):
        """Create comprehensive visualization charts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HotpotQA Vector Database Benchmark Results\nQdrant vs ChromaDB', fontsize=16, fontweight='bold')
        
        # Prepare data
        models = [r['model'] for r in results]
        databases = ['qdrant', 'chromadb']
        
        # 1. Query Performance (Speed)
        ax = axes[0, 0]
        query_times = []
        labels = []
        colors = []
        color_map = {'qdrant': '#FF6B6B', 'chromadb': '#4ECDC4'}
        
        for result in results:
            for db in databases:
                query_times.append(result[db]['avg_query_time'] * 1000)
                labels.append(f"{result['model']}\n({db})")
                colors.append(color_map[db])
        
        bars = ax.bar(labels, query_times, color=colors, alpha=0.8)
        ax.set_title('Average Query Time (ms)', fontweight='bold')
        ax.set_ylabel('Time (milliseconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)
        
        # 2. NDCG@10 Comparison
        ax = axes[0, 1]
        ndcg_scores = []
        
        for result in results:
            for db in databases:
                ndcg_scores.append(result[db]['ndcg@10'])
        
        bars = ax.bar(labels, ndcg_scores, color=colors, alpha=0.8)
        ax.set_title('NDCG@10 (Relevance Quality)', fontweight='bold')
        ax.set_ylabel('NDCG Score')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Precision@10
        ax = axes[0, 2]
        precision_scores = []
        
        for result in results:
            for db in databases:
                precision_scores.append(result[db]['precision@10'])
        
        bars = ax.bar(labels, precision_scores, color=colors, alpha=0.8)
        ax.set_title('Precision@10', fontweight='bold')
        ax.set_ylabel('Precision Score')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Recall@10
        ax = axes[1, 0]
        recall_scores = []
        
        for result in results:
            for db in databases:
                recall_scores.append(result[db]['recall@10'])
        
        bars = ax.bar(labels, recall_scores, color=colors, alpha=0.8)
        ax.set_title('Recall@10', fontweight='bold')
        ax.set_ylabel('Recall Score')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Hit Rate@10
        ax = axes[1, 1]
        hit_rates = []
        
        for result in results:
            for db in databases:
                hit_rates.append(result[db]['hit_rate@10'])
        
        bars = ax.bar(labels, hit_rates, color=colors, alpha=0.8)
        ax.set_title('Hit Rate@10', fontweight='bold')
        ax.set_ylabel('Hit Rate')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Queries per Second
        ax = axes[1, 2]
        qps_scores = []
        
        for result in results:
            for db in databases:
                qps_scores.append(result[db]['queries_per_second'])
        
        bars = ax.bar(labels, qps_scores, color=colors, alpha=0.8)
        ax.set_title('Queries per Second (Throughput)', fontweight='bold')
        ax.set_ylabel('Queries/Second')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', label='Qdrant'),
                          Patch(facecolor='#4ECDC4', label='ChromaDB')]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
        
        plt.tight_layout()
        
        # Save the plot
        chart_filename = f"hotpot_benchmark_comparison_{timestamp}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"[CHARTS] Visualization saved to {chart_filename}")

    def _generate_analysis_report(self, results: List[Dict]):
        """Generate a comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"HOTPOT_ANALYSIS_REPORT_{timestamp}.md"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# HotpotQA Vector Database Benchmark Analysis\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Dataset:** HotpotQA Multi-hop Reasoning Dataset\n")
            f.write(f"**Questions Tested:** {results[0]['queries_count']}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This comprehensive benchmark evaluates Qdrant and ChromaDB vector databases ")
            f.write("using the HotpotQA dataset, which contains complex multi-hop reasoning questions ")
            f.write("requiring information from multiple Wikipedia articles.\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write("**HotpotQA Dataset Characteristics:**\n")
            f.write("- 113,000+ question-answer pairs from Wikipedia\n")
            f.write("- Multi-hop reasoning requirements\n")
            f.write("- Supporting facts explicitly provided\n")
            f.write("- Diverse question types (comparison, bridge reasoning)\n")
            f.write("- Hard and medium difficulty levels\n\n")
            
            f.write("## Performance Analysis\n\n")
            
            for result in results:
                model = result['model']
                f.write(f"### {model.upper()} Results\n\n")
                
                f.write("| Metric | Qdrant | ChromaDB | Winner |\n")
                f.write("|--------|--------|----------|--------|\n")
                
                # Compare key metrics
                metrics = ['ndcg@10', 'precision@10', 'recall@10', 'hit_rate@10', 'avg_query_time', 'queries_per_second']
                
                for metric in metrics:
                    if metric == 'avg_query_time':
                        qdrant_val = result['qdrant'][metric] * 1000  # Convert to ms
                        chroma_val = result['chromadb'][metric] * 1000
                        unit = 'ms'
                        winner = 'Qdrant' if qdrant_val < chroma_val else 'ChromaDB'
                    elif metric == 'queries_per_second':
                        qdrant_val = result['qdrant'][metric]
                        chroma_val = result['chromadb'][metric]
                        unit = 'q/s'
                        winner = 'Qdrant' if qdrant_val > chroma_val else 'ChromaDB'
                    else:
                        qdrant_val = result['qdrant'][metric]
                        chroma_val = result['chromadb'][metric]
                        unit = ''
                        winner = 'Qdrant' if qdrant_val > chroma_val else 'ChromaDB'
                    
                    f.write(f"| {metric.replace('_', ' ').title()} | {qdrant_val:.3f}{unit} | {chroma_val:.3f}{unit} | **{winner}** |\n")
                
                f.write("\n")
                
                # Performance insights
                f.write("#### Key Insights:\n\n")
                
                ndcg_diff = (result['qdrant']['ndcg@10'] - result['chromadb']['ndcg@10']) / result['chromadb']['ndcg@10'] * 100
                speed_diff = (result['chromadb']['queries_per_second'] - result['qdrant']['queries_per_second']) / result['qdrant']['queries_per_second'] * 100
                
                f.write(f"- **Relevance Quality**: Qdrant shows {ndcg_diff:+.1f}% NDCG@10 compared to ChromaDB\n")
                f.write(f"- **Query Speed**: ChromaDB is {speed_diff:+.1f}% faster than Qdrant\n")
                f.write(f"- **Precision@10**: Qdrant achieves {result['qdrant']['precision@10']:.3f} vs ChromaDB's {result['chromadb']['precision@10']:.3f}\n")
                f.write(f"- **Recall@10**: Qdrant achieves {result['qdrant']['recall@10']:.3f} vs ChromaDB's {result['chromadb']['recall@10']:.3f}\n\n")
            
            f.write("## Technical Analysis\n\n")
            f.write("### Multi-hop Reasoning Performance\n\n")
            f.write("HotpotQA's multi-hop questions require systems to:\n")
            f.write("1. Identify multiple relevant documents\n")
            f.write("2. Extract supporting facts from each document\n")
            f.write("3. Combine information across documents\n\n")
            
            f.write("### Vector Database Architecture Impact\n\n")
            f.write("**Qdrant (Server-based):**\n")
            f.write("- Network latency adds to query time\n")
            f.write("- Optimized vector operations\n")
            f.write("- Better relevance ranking algorithms\n")
            f.write("- More sophisticated similarity search\n\n")
            
            f.write("**ChromaDB (Embedded):**\n")
            f.write("- In-process execution reduces latency\n")
            f.write("- Simpler similarity calculations\n")
            f.write("- Less memory overhead per query\n")
            f.write("- Faster for simple retrieval tasks\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("### For Production RAG Systems:\n\n")
            
            # Determine overall winner based on NDCG scores
            avg_qdrant_ndcg = np.mean([r['qdrant']['ndcg@10'] for r in results])
            avg_chroma_ndcg = np.mean([r['chromadb']['ndcg@10'] for r in results])
            
            if avg_qdrant_ndcg > avg_chroma_ndcg:
                f.write("**Recommended: Qdrant**\n\n")
                f.write("Qdrant demonstrates superior relevance quality in multi-hop reasoning tasks, ")
                f.write("which is critical for complex question-answering systems. The performance ")
                f.write("gains in NDCG@10 outweigh the speed disadvantage for most production use cases.\n\n")
            else:
                f.write("**Recommended: ChromaDB**\n\n")
                f.write("ChromaDB provides better overall performance for this benchmark, ")
                f.write("combining competitive relevance quality with superior query speed.\n\n")
            
            f.write("### Use Case Guidelines:\n\n")
            f.write("- **Choose Qdrant if**: Relevance quality is paramount, complex reasoning required\n")
            f.write("- **Choose ChromaDB if**: Query speed is critical, simpler retrieval tasks\n")
            f.write("- **Consider hybrid approach**: Use both databases for different query types\n\n")
            
            f.write("## Methodology\n\n")
            f.write("### Embedding Models Tested:\n")
            for result in results:
                f.write(f"- **{result['model']}**: ")
                if 'bge' in result['model']:
                    f.write("BAAI/bge-m3 (1024 dimensions, state-of-the-art quality)\n")
                else:
                    f.write("all-MiniLM-L6-v2 (384 dimensions, fast inference)\n")
            
            f.write("\n### Evaluation Metrics:\n")
            f.write("- **NDCG@k**: Normalized Discounted Cumulative Gain\n")
            f.write("- **Precision@k**: Fraction of retrieved documents that are relevant\n")
            f.write("- **Recall@k**: Fraction of relevant documents that are retrieved\n")
            f.write("- **Hit Rate@k**: Whether any relevant document appears in top-k\n")
            f.write("- **Query Speed**: Average time per query and queries per second\n\n")
            
            f.write("### Ground Truth:\n")
            f.write("Ground truth relevance is determined by HotpotQA's supporting facts, ")
            f.write("which explicitly identify which sentences from which documents are ")
            f.write("necessary to answer each question.\n\n")
            
            f.write("---\n")
            f.write("*This analysis was generated automatically by the HotpotQA Vector Database Benchmark suite.*\n")
        
        self.logger.info(f"[REPORT] Analysis report saved to {report_filename}")


def main():
    """Main execution function."""
    try:
        benchmark = HotpotQAVectorDBBenchmark()
        results = benchmark.run_full_benchmark()
        return results
        
    except Exception as e:
        logging.error(f"[FATAL] Benchmark failed: {e}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
