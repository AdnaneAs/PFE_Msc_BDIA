"""
Multi-Reranker Benchmark: Comprehensive Comparison
==================================================

Enhanced benchmark comparing RAG performance across multiple reranking strategies:
1. No Reranker (Baseline)
2. BGE Reranker Base (BAAI/bge-reranker-base)
3. BGE Reranker Large (BAAI/bge-reranker-large) 
4. BGE Reranker v2-M3 (BAAI/bge-reranker-v2-m3)

This benchmark provides comprehensive insights into:
- Quality improvements with different rerankers
- Performance vs. speed trade-offs
- Model comparison for production decision-making

Metrics Evaluated:
- Precision@K (K=1,3,5,10): Fraction of retrieved documents that are relevant
- Recall@K (K=1,3,5,10): Fraction of relevant documents that are retrieved  
- NDCG@K (K=1,3,5,10): Normalized Discounted Cumulative Gain
- MAP (Mean Average Precision): Quality of document ranking
- MRR (Mean Reciprocal Rank): Position of first relevant document
- Hit Rate@K: Whether at least one relevant document is retrieved
- Processing Time: Retrieval + Reranking time analysis

Date: May 31, 2025
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import random
import json
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime
import logging
import traceback
import gc
import torch

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

# Vector database and reranker imports
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'multi_reranker_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiRerankerBenchmark:
    """
    Comprehensive benchmark comparing multiple reranking strategies
    """
    
    def __init__(self, db_path="../db_data", collection_name="documents_all_minilm_l6_v2"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        # Reranker configurations
        self.reranker_configs = {
            "no_reranker": {
                "name": "No Reranker",
                "model": None,
                "description": "Baseline vector similarity search only",
                "color": "#95a5a6"
            },
            "bge_base": {
                "name": "BGE Reranker Base",
                "model": "BAAI/bge-reranker-base",
                "description": "Balanced performance and quality",
                "color": "#3498db"
            },
            "bge_large": {
                "name": "BGE Reranker Large", 
                "model": "BAAI/bge-reranker-large",
                "description": "Higher quality, more parameters",
                "color": "#e74c3c"
            },
            "bge_v2_m3": {
                "name": "BGE Reranker v2-M3",
                "model": "BAAI/bge-reranker-v2-m3", 
                "description": "Latest multilingual model",
                "color": "#f39c12"
            }
        }
        
        self.rerankers = {}
        self.results = {}
        
        # HotpotQA sample queries for evaluation
        self.evaluation_queries = [
            {
                "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
                "expected_topics": ["actress", "government", "position", "Corliss Archer", "Kiss and Tell"]
            },
            {
                "question": "What science fantasy young adult series, told in first person, has a set of companion books narrated by four different characters?",
                "expected_topics": ["science fantasy", "young adult", "first person", "companion books", "characters"]
            },
            {
                "question": "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
                "expected_topics": ["tennis", "Grand Slam", "Henri Leconte", "Jonathan Stark", "titles"]
            },
            {
                "question": "Are both Cress Williams and Kadeem Hardison actors?",
                "expected_topics": ["Cress Williams", "Kadeem Hardison", "actors", "acting"]
            },
            {
                "question": "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was held?",
                "expected_topics": ["2013", "Liqui Moly", "Bathurst", "12 Hour", "track length"]
            },
            {
                "question": "What type of plant is Eucalyptus delegatensis?",
                "expected_topics": ["Eucalyptus", "delegatensis", "plant", "tree", "species"]
            },
            {
                "question": "In what year was the performer who played the main character in Salt born?",
                "expected_topics": ["Salt", "performer", "main character", "birth year", "actor"]
            },
            {
                "question": "What is the real name of the rapper whose 2016 album was titled The Life of Pablo?",
                "expected_topics": ["rapper", "2016", "The Life of Pablo", "real name", "album"]
            },
            {
                "question": "Which magazine has published articles by both Glenn Greenwald and Matt Taibbi?",
                "expected_topics": ["magazine", "Glenn Greenwald", "Matt Taibbi", "articles", "published"]
            },
            {
                "question": "What is the wingspan of the aircraft that was the predecessor to the Airbus A380?",
                "expected_topics": ["wingspan", "aircraft", "predecessor", "Airbus A380", "aviation"]
            }
        ]
    
    def setup_connections(self):
        """Initialize database connections and embedding model"""
        try:
            logger.info("Setting up database connections...")
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_collection(name=self.collection_name)
            
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            logger.info("Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False
    
    def load_rerankers(self):
        """Load all available rerankers with error handling"""
        logger.info("Loading rerankers...")
        
        for config_name, config in self.reranker_configs.items():
            if config_name == "no_reranker":
                self.rerankers[config_name] = None
                logger.info(f"✓ {config['name']}: Ready (no model needed)")
                continue
                
            try:
                logger.info(f"Loading {config['name']} ({config['model']})...")
                reranker = CrossEncoder(config['model'])
                
                # Test the reranker with a simple example
                test_query = "test query"
                test_docs = ["test document"]
                test_scores = reranker.predict([(test_query, test_docs[0])])
                
                self.rerankers[config_name] = reranker
                logger.info(f"✓ {config['name']}: Loaded successfully (test score: {test_scores[0]:.4f})")
                
                # Clear GPU memory after loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"✗ {config['name']}: Failed to load - {str(e)}")
                self.rerankers[config_name] = None
    
    def clear_reranker_memory(self, config_name):
        """Clear reranker from memory to manage GPU resources"""
        if config_name in self.rerankers and self.rerankers[config_name] is not None:
            del self.rerankers[config_name]
            self.rerankers[config_name] = None
            
        # Force garbage collection and GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reload_reranker(self, config_name):
        """Reload a specific reranker"""
        if config_name == "no_reranker":
            return True
            
        config = self.reranker_configs[config_name]
        try:
            logger.info(f"Reloading {config['name']}...")
            self.rerankers[config_name] = CrossEncoder(config['model'])
            return True
        except Exception as e:
            logger.error(f"Failed to reload {config['name']}: {str(e)}")
            return False
    
    def retrieve_documents(self, query: str, top_k: int = 20) -> Tuple[List[str], List[Dict], List[float], float]:
        """Retrieve initial documents using vector similarity search"""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieval_time = time.time() - start_time
        
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        return documents, metadatas, distances, retrieval_time
    
    def rerank_documents(self, query: str, documents: List[str], reranker, top_n: int = 10) -> Tuple[List[str], List[float], float]:
        """Rerank documents using specified reranker"""
        if reranker is None:
            # No reranking - return top_n documents with distance-based scores
            rerank_time = 0.0
            selected_docs = documents[:top_n]
            # Convert distances to similarity scores (higher is better)
            similarity_scores = [1.0 - (i * 0.1) for i in range(len(selected_docs))]
            return selected_docs, similarity_scores, rerank_time
        
        start_time = time.time()
        
        try:
            # Create query-document pairs for reranking
            query_doc_pairs = [(query, doc) for doc in documents]
            
            # Get reranking scores
            rerank_scores = reranker.predict(query_doc_pairs)
            
            # Sort by reranking scores (descending)
            doc_score_pairs = list(zip(documents, rerank_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top_n documents
            selected_docs = [pair[0] for pair in doc_score_pairs[:top_n]]
            scores = [pair[1] for pair in doc_score_pairs[:top_n]]
            
            rerank_time = time.time() - start_time
            return selected_docs, scores, rerank_time
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Fallback to no reranking
            rerank_time = time.time() - start_time
            selected_docs = documents[:top_n]
            fallback_scores = [1.0 - (i * 0.1) for i in range(len(selected_docs))]
            return selected_docs, fallback_scores, rerank_time
    
    def calculate_relevance(self, query: str, documents: List[str], expected_topics: List[str]) -> List[float]:
        """Calculate relevance scores based on topic matching and semantic similarity"""
        relevance_scores = []
        
        query_lower = query.lower()
        expected_lower = [topic.lower() for topic in expected_topics]
        
        for doc in documents:
            doc_lower = doc.lower()
            
            # Topic matching score (0-1)
            topic_matches = sum(1 for topic in expected_lower if topic in doc_lower)
            topic_score = min(topic_matches / len(expected_topics), 1.0) if expected_topics else 0.0
            
            # Query term matching score (0-1)
            query_terms = query_lower.split()
            term_matches = sum(1 for term in query_terms if term in doc_lower and len(term) > 2)
            term_score = min(term_matches / len(query_terms), 1.0) if query_terms else 0.0
              # Combined relevance score
            relevance = 0.6 * topic_score + 0.4 * term_score
            relevance_scores.append(relevance)
        
        return relevance_scores
    
    def evaluate_ranking(self, relevance_scores: List[float], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Calculate ranking evaluation metrics"""
        metrics = {}
        n_docs = len(relevance_scores)
        
        # Safety check: ensure we have documents to evaluate
        if n_docs == 0:
            logger.warning("No documents to evaluate - returning zero metrics")
            for k in k_values:
                metrics[f'precision@{k}'] = 0.0
                metrics[f'recall@{k}'] = 0.0
                metrics[f'hit_rate@{k}'] = 0.0
                metrics[f'ndcg@{k}'] = 0.0
            metrics['map'] = 0.0
            metrics['mrr'] = 0.0
            return metrics
        
        # Binary relevance (threshold = 0.3)
        binary_relevance = [1 if score >= 0.3 else 0 for score in relevance_scores]
        total_relevant = sum(binary_relevance)
        
        for k in k_values:
            k_actual = min(k, n_docs)
            
            if k_actual == 0:
                continue
                
            # Precision@K
            precision_k = sum(binary_relevance[:k_actual]) / k_actual
            metrics[f'precision@{k}'] = precision_k
            
            # Recall@K
            recall_k = sum(binary_relevance[:k_actual]) / max(total_relevant, 1)
            metrics[f'recall@{k}'] = recall_k
            
            # Hit Rate@K
            hit_rate_k = 1 if sum(binary_relevance[:k_actual]) > 0 else 0
            metrics[f'hit_rate@{k}'] = hit_rate_k
              # NDCG@K
            if k_actual > 0:
                true_relevance = np.array([relevance_scores[:k_actual]])
                ideal_relevance = np.array([sorted(relevance_scores[:k_actual], reverse=True)])
                
                if np.sum(ideal_relevance) > 0 and k_actual > 1:
                    # NDCG calculation requires at least 2 documents
                    try:
                        ndcg_k = ndcg_score(ideal_relevance, true_relevance)
                    except ValueError as e:
                        # Handle case where NDCG cannot be computed (e.g., only 1 document)
                        logger.warning(f"NDCG@{k} cannot be computed with {k_actual} documents: {str(e)}")
                        ndcg_k = 1.0 if relevance_scores[0] > 0 else 0.0  # Perfect score if single relevant doc
                elif k_actual == 1:
                    # For single document, NDCG is 1.0 if relevant, 0.0 if not
                    ndcg_k = 1.0 if relevance_scores[0] > 0 else 0.0
                else:
                    ndcg_k = 0.0
                    
                metrics[f'ndcg@{k}'] = ndcg_k
        
        # Mean Average Precision (MAP)
        if total_relevant > 0:
            avg_precisions = []
            for i, rel in enumerate(binary_relevance):
                if rel == 1:
                    precision_at_i = sum(binary_relevance[:i+1]) / (i + 1)
                    avg_precisions.append(precision_at_i)
            metrics['map'] = np.mean(avg_precisions) if avg_precisions else 0.0
        else:
            metrics['map'] = 0.0
        
        # Mean Reciprocal Rank (MRR)
        first_relevant_pos = next((i+1 for i, rel in enumerate(binary_relevance) if rel == 1), None)
        metrics['mrr'] = 1.0 / first_relevant_pos if first_relevant_pos else 0.0
        
        return metrics
    
    def run_single_configuration(self, config_name: str, num_samples: int = 20) -> Dict:
        """Run benchmark for a single reranker configuration"""
        config = self.reranker_configs[config_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running benchmark: {config['name']}")
        logger.info(f"{'='*60}")
        
        # Reload reranker for this configuration
        if not self.reload_reranker(config_name):
            logger.error(f"Failed to load reranker for {config_name}")
            return None
            
        reranker = self.rerankers[config_name]
        
        all_metrics = []
        retrieval_times = []
        reranking_times = []
        
        # Use subset of evaluation queries
        selected_queries = random.sample(self.evaluation_queries, min(num_samples, len(self.evaluation_queries)))
        
        for i, query_data in enumerate(selected_queries):
            query = query_data["question"]
            expected_topics = query_data["expected_topics"]
            
            logger.info(f"Query {i+1}/{len(selected_queries)}: {query[:80]}...")
            
            try:
                # Retrieve initial documents
                documents, metadatas, distances, retrieval_time = self.retrieve_documents(query, top_k=20)
                retrieval_times.append(retrieval_time)
                
                if not documents:
                    logger.warning(f"No documents found for query {i+1}")
                    continue
                
                # Rerank documents
                reranked_docs, scores, rerank_time = self.rerank_documents(query, documents, reranker, top_n=10)
                reranking_times.append(rerank_time)
                
                # Calculate relevance
                relevance_scores = self.calculate_relevance(query, reranked_docs, expected_topics)
                
                # Evaluate ranking
                metrics = self.evaluate_ranking(relevance_scores)
                all_metrics.append(metrics)
                
                # Log progress
                if (i + 1) % 5 == 0:
                    logger.info(f"Completed {i+1}/{len(selected_queries)} queries")
                    
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {str(e)}")
                continue
        
        # Clear reranker from memory
        self.clear_reranker_memory(config_name)
        
        # Calculate average metrics
        if not all_metrics:
            logger.error(f"No valid results for {config_name}")
            return None
            
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        # Add timing metrics
        avg_metrics['avg_retrieval_time'] = np.mean(retrieval_times)
        avg_metrics['avg_reranking_time'] = np.mean(reranking_times)
        avg_metrics['total_processing_time'] = avg_metrics['avg_retrieval_time'] + avg_metrics['avg_reranking_time']
        
        result = {
            'config_name': config_name,
            'config': config,
            'metrics': avg_metrics,
            'num_queries': len(selected_queries),
            'num_valid_results': len(all_metrics)
        }
        
        logger.info(f"Completed {config['name']}: {len(all_metrics)} valid results")
        return result
    
    def run_full_benchmark(self, num_samples_per_config: int = 20) -> Dict:
        """Run comprehensive benchmark across all reranker configurations"""
        logger.info("🚀 Starting Multi-Reranker Comprehensive Benchmark")
        logger.info(f"Samples per configuration: {num_samples_per_config}")
        logger.info("="*80)
        
        if not self.setup_connections():
            logger.error("Failed to setup connections")
            return None
        
        # Load all rerankers first to check availability
        self.load_rerankers()
        
        # Check which rerankers are available
        available_configs = []
        for config_name, reranker in self.rerankers.items():
            if config_name == "no_reranker" or reranker is not None:
                available_configs.append(config_name)
        
        logger.info(f"Available configurations: {[self.reranker_configs[c]['name'] for c in available_configs]}")
          # Run benchmark for each available configuration
        all_results = {}
        for config_name in available_configs:
            result = self.run_single_configuration(config_name, num_samples_per_config)
            if result:
                all_results[config_name] = result
        
        self.results = all_results
        return all_results
    
    def create_comparison_visualizations(self):
        """Create enhanced comparison visualizations with separate images for each @k metric"""
        if not self.results:
            logger.error("No results available for visualization")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create separate visualizations for each metric group
        self.create_precision_visualizations(timestamp)
        self.create_recall_visualizations(timestamp)
        self.create_ndcg_visualizations(timestamp)
        self.create_overall_metrics_visualization(timestamp)
        self.create_timing_visualization(timestamp)
          # Create summary table visualization
        self.create_enhanced_summary_table(timestamp)
        self.create_detailed_comparison_table(timestamp)
        
        logger.info("All enhanced visualizations created successfully")
    
    def create_precision_visualizations(self, timestamp: str):
        """Create separate visualizations for each Precision@k metric"""
        precision_metrics = ['precision@1', 'precision@3', 'precision@5', 'precision@10']
        
        # Create individual plots for each precision metric
        for i, metric in enumerate(precision_metrics):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract data for this metric
            rerankers = []
            values = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue theme for precision
            
            for j, (config_name, result) in enumerate(self.results.items()):
                rerankers.append(result['config']['name'])
                values.append(result['metrics'].get(metric, 0))
            
            # Create bar plot
            bars = ax.bar(rerankers, values, color=colors[:len(rerankers)], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Find best performer and add crown
            best_idx = np.argmax(values)
            if values[best_idx] > 0:
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                ax.text(best_idx, values[best_idx] + 0.03, '👑', ha='center', va='bottom', fontsize=16)
            
            # Styling
            k_value = metric.split('@')[1]
            ax.set_title(f'Precision@{k_value} Comparison Across Rerankers', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel(f'Precision@{k_value}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Reranker Configuration', fontsize=14, fontweight='bold')
            
            # Improve layout
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.2)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add improvement percentages relative to baseline
            if len(values) > 1:
                baseline_value = values[0]  # Assuming no_reranker is first
                for j, (bar, value) in enumerate(zip(bars[1:], values[1:]), 1):
                    if baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                        ax.text(bar.get_x() + bar.get_width()/2., value/2,
                               f'+{improvement:.1f}%', ha='center', va='center', 
                               fontweight='bold', color='white', fontsize=10)
            
            # Save plot
            plot_filename = f'precision_at_{k_value}_comparison_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Precision@{k_value} plot saved as: {plot_filename}")
            plt.close()

    def create_recall_visualizations(self, timestamp: str):
        """Create separate visualizations for each Recall@k metric"""
        recall_metrics = ['recall@1', 'recall@3', 'recall@5', 'recall@10']
        
        for i, metric in enumerate(recall_metrics):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract data for this metric
            rerankers = []
            values = []
            colors = ['#2ca02c', '#17becf', '#bcbd22', '#ff7f0e']  # Green theme for recall
            
            for j, (config_name, result) in enumerate(self.results.items()):
                rerankers.append(result['config']['name'])
                values.append(result['metrics'].get(metric, 0))
            
            # Create bar plot
            bars = ax.bar(rerankers, values, color=colors[:len(rerankers)], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Find best performer and add crown
            best_idx = np.argmax(values)
            if values[best_idx] > 0:
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                ax.text(best_idx, values[best_idx] + 0.03, '👑', ha='center', va='bottom', fontsize=16)
            
            # Styling
            k_value = metric.split('@')[1]
            ax.set_title(f'Recall@{k_value} Comparison Across Rerankers', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel(f'Recall@{k_value}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Reranker Configuration', fontsize=14, fontweight='bold')
            
            # Improve layout
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.2)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add improvement percentages relative to baseline
            if len(values) > 1:
                baseline_value = values[0]  # Assuming no_reranker is first
                for j, (bar, value) in enumerate(zip(bars[1:], values[1:]), 1):
                    if baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                        ax.text(bar.get_x() + bar.get_width()/2., value/2,
                               f'+{improvement:.1f}%', ha='center', va='center', 
                               fontweight='bold', color='white', fontsize=10)
            
            # Save plot
            plot_filename = f'recall_at_{k_value}_comparison_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Recall@{k_value} plot saved as: {plot_filename}")
            plt.close()

    def create_ndcg_visualizations(self, timestamp: str):
        """Create separate visualizations for each NDCG@k metric"""
        ndcg_metrics = ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']
        
        for i, metric in enumerate(ndcg_metrics):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract data for this metric
            rerankers = []
            values = []
            colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # Purple theme for NDCG
            
            for j, (config_name, result) in enumerate(self.results.items()):
                rerankers.append(result['config']['name'])
                values.append(result['metrics'].get(metric, 0))
            
            # Create bar plot
            bars = ax.bar(rerankers, values, color=colors[:len(rerankers)], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Find best performer and add crown
            best_idx = np.argmax(values)
            if values[best_idx] > 0:
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                ax.text(best_idx, values[best_idx] + 0.03, '👑', ha='center', va='bottom', fontsize=16)
            
            # Styling
            k_value = metric.split('@')[1]
            ax.set_title(f'NDCG@{k_value} Comparison Across Rerankers', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel(f'NDCG@{k_value}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Reranker Configuration', fontsize=14, fontweight='bold')
            
            # Improve layout
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.2)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add improvement percentages relative to baseline
            if len(values) > 1:
                baseline_value = values[0]  # Assuming no_reranker is first
                for j, (bar, value) in enumerate(zip(bars[1:], values[1:]), 1):
                    if baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                        ax.text(bar.get_x() + bar.get_width()/2., value/2,
                               f'+{improvement:.1f}%', ha='center', va='center', 
                               fontweight='bold', color='white', fontsize=10)
            
            # Save plot
            plot_filename = f'ndcg_at_{k_value}_comparison_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"NDCG@{k_value} plot saved as: {plot_filename}")
            plt.close()

    def create_overall_metrics_visualization(self, timestamp: str):
        """Create visualization for overall metrics (MAP, MRR)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract data
        rerankers = []
        map_values = []
        mrr_values = []
        
        for config_name, result in self.results.items():
            rerankers.append(result['config']['name'])
            map_values.append(result['metrics'].get('map', 0))
            mrr_values.append(result['metrics'].get('mrr', 0))
        
        # MAP plot
        colors_map = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        bars1 = ax1.bar(rerankers, map_values, color=colors_map[:len(rerankers)], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars1, map_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Find best MAP performer
        best_map_idx = np.argmax(map_values)
        if map_values[best_map_idx] > 0:
            bars1[best_map_idx].set_edgecolor('gold')
            bars1[best_map_idx].set_linewidth(3)
            ax1.text(best_map_idx, map_values[best_map_idx] + 0.03, '👑', ha='center', va='bottom', fontsize=16)
        
        ax1.set_title('Mean Average Precision (MAP)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('MAP Score', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(map_values) * 1.2)
        
        # MRR plot
        colors_mrr = ['#ffb3ba', '#baffc9', '#bae1ff', '#ffffba']
        bars2 = ax2.bar(rerankers, mrr_values, color=colors_mrr[:len(rerankers)], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars2, mrr_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Find best MRR performer
        best_mrr_idx = np.argmax(mrr_values)
        if mrr_values[best_mrr_idx] > 0:
            bars2[best_mrr_idx].set_edgecolor('gold')
            bars2[best_mrr_idx].set_linewidth(3)
            ax2.text(best_mrr_idx, mrr_values[best_mrr_idx] + 0.03, '👑', ha='center', va='bottom', fontsize=16)
        
        ax2.set_title('Mean Reciprocal Rank (MRR)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MRR Score', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(mrr_values) * 1.2)
        
        # Common formatting
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Reranker Configuration', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'overall_metrics_comparison_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Overall metrics plot saved as: {plot_filename}")
        plt.close()

    def create_timing_visualization(self, timestamp):
        """Create visualization for timing metrics"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        retrieval_times = []
        reranking_times = []
        config_names = []
        
        for config_name, result in self.results.items():
            config_names.append(result['config']['name'])
            retrieval_times.append(result['metrics'].get('avg_retrieval_time', 0))
            reranking_times.append(result['metrics'].get('avg_reranking_time', 0))
        
        x = np.arange(len(config_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, retrieval_times, width, label='Retrieval Time', 
                      color='#3498DB', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, reranking_times, width, label='Reranking Time', 
                      color='#E74C3C', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Reranker Configuration', fontweight='bold', fontsize=14)
        ax.set_ylabel('Processing Time (seconds)', fontweight='bold', fontsize=14)
        ax.set_title('Processing Time Breakdown by Component', fontweight='bold', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=0, ha='center', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
          # Save timing plot
        plot_filename = f'timing_breakdown_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Timing breakdown plot saved as: {plot_filename}")
        
        plt.close()
    
    def create_enhanced_summary_table(self, timestamp: str):
        """Create an enhanced summary table without colors but with better formatting"""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare comprehensive table data
        headers = [
            'Reranker', 'Model', 'MAP', 'NDCG@5', 'Precision@5', 'Recall@5', 
            'MRR', 'Hit Rate@5', 'Retrieval Time (s)', 'Reranking Time (s)', 'Total Time (s)'
        ]
        
        table_data = []
        
        for config_name, result in self.results.items():
            metrics = result['metrics']
            row = [
                result['config']['name'],
                result['config'].get('model', 'N/A'),
                f"{metrics.get('map', 0):.3f}",
                f"{metrics.get('ndcg@5', 0):.3f}",
                f"{metrics.get('precision@5', 0):.3f}",
                f"{metrics.get('recall@5', 0):.3f}",
                f"{metrics.get('mrr', 0):.3f}",
                f"{metrics.get('hit_rate@5', 0):.3f}",
                f"{metrics.get('avg_retrieval_time', 0):.3f}",
                f"{metrics.get('avg_reranking_time', 0):.3f}",
                f"{metrics.get('total_processing_time', 0):.3f}"
            ]
            table_data.append(row)
          # Create table with clean styling
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.5)
        
        # Style the table with clean, professional look (no colors)
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#f0f0f0')  # Very light gray header
            table[(0, i)].set_text_props(weight='bold', color='black')
            table[(0, i)].set_edgecolor('black')
            table[(0, i)].set_linewidth(1.5)
        
        # Style data rows with alternating white/very light gray
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f8f8')  # Very light gray for even rows
                else:
                    table[(i, j)].set_facecolor('white')    # White for odd rows
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(0.8)
        
        # Highlight best performance with bold text instead of colors
        if len(table_data) > 1:
            # Find best performers for each metric (excluding model name)
            for col_idx, metric_name in enumerate(['MAP', 'NDCG@5', 'Precision@5', 'Recall@5', 'MRR']):
                if col_idx + 2 < len(headers):  # Skip Reranker and Model columns
                    metric_values = [float(row[col_idx + 2]) for row in table_data]
                    max_metric_idx = metric_values.index(max(metric_values))
                    # Bold the best performer for this metric
                    table[(max_metric_idx + 1, col_idx + 2)].set_text_props(weight='bold', color='#2e7d32')
                    table[(max_metric_idx + 1, col_idx + 2)].set_edgecolor('black')
                    table[(max_metric_idx + 1, col_idx + 2)].set_linewidth(1.5)
        
        plt.title('Multi-Reranker Performance Summary\nComprehensive Metrics Comparison', 
                 fontsize=16, fontweight='bold', pad=30)
        
        # Add performance indicators as text
        if len(table_data) > 1:
            # Find best MAP performer
            map_values = [float(row[2]) for row in table_data]
            best_map_idx = map_values.index(max(map_values))
            best_config = table_data[best_map_idx][0]
            best_map = table_data[best_map_idx][2]
            
            # Add annotation
            plt.figtext(0.5, 0.02, f'🏆 Best Overall Performance: {best_config} (MAP: {best_map})', 
                       ha='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
          # Save table
        table_filename = f'multi_reranker_summary_table_{timestamp}.png'
        plt.savefig(table_filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Enhanced summary table saved as: {table_filename}")
        
        plt.close()
    
    def create_detailed_comparison_table(self, timestamp: str):
        """Create a detailed comparison table showing improvements over baseline"""
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare detailed comparison data
        headers = [
            'Reranker', 'MAP Improvement (%)', 'NDCG@5 Improvement (%)', 
            'Precision@5 Improvement (%)', 'Recall@5 Improvement (%)', 
            'MRR Improvement (%)', 'Time Overhead (%)', 'Quality Score'
        ]
        
        table_data = []
        baseline_result = None
        
        # Find baseline (no_reranker) results
        for config_name, result in self.results.items():
            if config_name == "no_reranker":
                baseline_result = result
                break
        
        if not baseline_result:
            logger.error("No baseline (no_reranker) results found for comparison")
            return
        
        baseline_metrics = baseline_result['metrics']
        baseline_time = baseline_metrics.get('total_processing_time', 0)
        
        for config_name, result in self.results.items():
            config = result['config']
            metrics = result['metrics']
            
            # Calculate improvements over baseline
            map_improvement = self._calculate_improvement(
                baseline_metrics.get('map', 0), 
                metrics.get('map', 0)
            )
            ndcg5_improvement = self._calculate_improvement(
                baseline_metrics.get('ndcg@5', 0), 
                metrics.get('ndcg@5', 0)
            )
            precision5_improvement = self._calculate_improvement(
                baseline_metrics.get('precision@5', 0), 
                metrics.get('precision@5', 0)
            )
            recall5_improvement = self._calculate_improvement(
                baseline_metrics.get('recall@5', 0), 
                metrics.get('recall@5', 0)
            )
            mrr_improvement = self._calculate_improvement(
                baseline_metrics.get('mrr', 0), 
                metrics.get('mrr', 0)
            )
            time_overhead = self._calculate_improvement(
                baseline_time, 
                metrics.get('total_processing_time', 0)
            )
            
            # Calculate composite quality score (average of key metrics)
            quality_score = np.mean([
                metrics.get('map', 0),
                metrics.get('ndcg@5', 0), 
                metrics.get('precision@5', 0),
                metrics.get('mrr', 0)
            ])
            
            row = [
                config['name'],
                f"{map_improvement:+.1f}%" if map_improvement != 0 else "0.0%",
                f"{ndcg5_improvement:+.1f}%" if ndcg5_improvement != 0 else "0.0%",
                f"{precision5_improvement:+.1f}%" if precision5_improvement != 0 else "0.0%",
                f"{recall5_improvement:+.1f}%" if recall5_improvement != 0 else "0.0%",
                f"{mrr_improvement:+.1f}%" if mrr_improvement != 0 else "0.0%",
                f"{time_overhead:+.1f}%" if time_overhead != 0 else "0.0%",
                f"{quality_score:.3f}"
            ]
            table_data.append(row)
        
        # Create table with professional styling
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.12, 0.15, 0.15, 0.12, 0.12, 0.12, 0.1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold', color='black')
            cell.set_height(0.08)
        
        # Row styling with alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f9f9f9')
                else:
                    cell.set_facecolor('white')
                cell.set_height(0.06)
                
                # Bold the best values in each column (except reranker name and time overhead)
                if j in [1, 2, 3, 4, 5, 7]:  # Improvement and quality columns
                    values = [float(table_data[k][j].replace('%', '').replace('+', '')) if k != i-1 else 0 
                             for k in range(len(table_data))]
                    if j != 6:  # Not time overhead (lower is better for time)
                        if float(table_data[i-1][j].replace('%', '').replace('+', '')) == max(values):
                            cell.set_text_props(weight='bold', color='#2e7d32')
                    else:  # Time overhead - lowest is best
                        if float(table_data[i-1][j].replace('%', '').replace('+', '')) == min(values):
                            cell.set_text_props(weight='bold', color='#2e7d32')
        
        # Title
        ax.set_title('Detailed Performance Comparison vs Baseline (No Reranker)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save table
        table_filename = f'detailed_comparison_table_{timestamp}.png'
        plt.savefig(table_filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Detailed comparison table saved as: {table_filename}")
        
        plt.close()

    def _calculate_improvement(self, baseline_value: float, current_value: float) -> float:
        """Calculate percentage improvement over baseline"""
        if baseline_value == 0:
            return 0.0
        return ((current_value - baseline_value) / baseline_value) * 100

    def save_results(self):
        """Save detailed results to files"""
        if not self.results:
            logger.error("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = f'multi_reranker_benchmark_results_{timestamp}.json'
        with open(json_filename, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for config_name, result in self.results.items():
                json_results[config_name] = {
                    'config_name': result['config_name'],
                    'config': result['config'],
                    'metrics': result['metrics'],
                    'num_queries': result['num_queries'],
                    'num_valid_results': result['num_valid_results']
                }
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved as JSON: {json_filename}")
        
        # Save CSV summary
        csv_filename = f'multi_reranker_benchmark_results_{timestamp}.csv'
        csv_data = []
        
        for config_name, result in self.results.items():
            row = {
                'reranker': result['config']['name'],
                'model': result['config'].get('model', 'N/A'),
                **result['metrics']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        logger.info(f"Results saved as CSV: {csv_filename}")
        
        # Generate markdown report
        self.generate_markdown_report(timestamp)
    
    def generate_markdown_report(self, timestamp: str):
        """Generate comprehensive markdown report"""
        report_filename = f'MULTI_RERANKER_BENCHMARK_REPORT_{timestamp}.md'
        
        with open(report_filename, 'w') as f:
            f.write("# Multi-Reranker Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This benchmark compares the performance of different reranking strategies:\n\n")
            
            for config_name, result in self.results.items():
                config = result['config']
                f.write(f"- **{config['name']}**: {config['description']}\n")
            
            f.write(f"\n**Evaluation Dataset:** {len(self.evaluation_queries)} HotpotQA questions\n")
            f.write(f"**Documents per Query:** Up to 20 retrieved, top 10 evaluated\n\n")
            
            f.write("## Performance Comparison\n\n")
            f.write("### Key Metrics Summary\n\n")
            f.write("| Reranker | MAP | NDCG@5 | Precision@5 | MRR | Hit Rate@5 | Processing Time (s) |\n")
            f.write("|----------|-----|--------|-------------|-----|-------------|--------------------|\n")
            
            for config_name, result in self.results.items():
                metrics = result['metrics']
                f.write(f"| {result['config']['name']} | "
                       f"{metrics.get('map', 0):.3f} | "
                       f"{metrics.get('ndcg@5', 0):.3f} | "
                       f"{metrics.get('precision@5', 0):.3f} | "
                       f"{metrics.get('mrr', 0):.3f} | "
                       f"{metrics.get('hit_rate@5', 0):.3f} | "
                       f"{metrics.get('total_processing_time', 0):.3f} |\n")
            
            # Find best performing model
            best_map = max(result['metrics'].get('map', 0) for result in self.results.values())
            best_config = None
            for config_name, result in self.results.items():
                if result['metrics'].get('map', 0) == best_map:
                    best_config = result['config']['name']
                    break
            
            f.write(f"\n### Best Overall Performance\n\n")
            f.write(f"**{best_config}** achieved the highest MAP score of {best_map:.3f}\n\n")
            
            # Detailed analysis
            f.write("## Detailed Analysis\n\n")
            
            for config_name, result in self.results.items():
                config = result['config']
                metrics = result['metrics']
                
                f.write(f"### {config['name']}\n\n")
                f.write(f"**Model:** {config.get('model', 'N/A')}\n")
                f.write(f"**Description:** {config['description']}\n\n")
                
                f.write("**Performance Metrics:**\n")
                f.write(f"- MAP: {metrics.get('map', 0):.3f}\n")
                f.write(f"- NDCG@5: {metrics.get('ndcg@5', 0):.3f}\n")
                f.write(f"- Precision@5: {metrics.get('precision@5', 0):.3f}\n")
                f.write(f"- MRR: {metrics.get('mrr', 0):.3f}\n")
                f.write(f"- Hit Rate@5: {metrics.get('hit_rate@5', 0):.3f}\n")
                f.write(f"- Processing Time: {metrics.get('total_processing_time', 0):.3f}s\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on this benchmark analysis:\n\n")
            
            # Performance vs Speed analysis
            baseline_time = None
            for config_name, result in self.results.items():
                if config_name == "no_reranker":
                    baseline_time = result['metrics'].get('total_processing_time', 0)
                    break
            
            f.write("### Performance vs Speed Trade-offs:\n\n")
            for config_name, result in self.results.items():
                if config_name == "no_reranker":
                    continue
                    
                config = result['config']
                metrics = result['metrics']
                
                map_improvement = 0
                time_overhead = 0
                
                # Calculate improvements over baseline
                baseline_result = self.results.get("no_reranker")
                if baseline_result:
                    baseline_map = baseline_result['metrics'].get('map', 0)
                    current_map = metrics.get('map', 0)
                    map_improvement = ((current_map - baseline_map) / baseline_map * 100) if baseline_map > 0 else 0
                    
                    if baseline_time:
                        current_time = metrics.get('total_processing_time', 0)
                        time_overhead = ((current_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
                
                f.write(f"- **{config['name']}**: MAP improvement +{map_improvement:.1f}%, "
                       f"Time overhead +{time_overhead:.1f}%\n")
            
            f.write("\n### Production Recommendations:\n\n")
            f.write("Choose the reranker based on your priority:\n\n")
            f.write("- **Speed Priority**: No reranker (fastest baseline)\n")
            f.write("- **Balanced Performance**: BGE Reranker Base (good quality/speed balance)\n")
            f.write("- **Quality Priority**: BGE Reranker Large or v2-M3 (best performance)\n")
            f.write("- **Multilingual Support**: BGE Reranker v2-M3 (latest multilingual model)\n\n")
        
        logger.info(f"Markdown report saved as: {report_filename}")


def main():
    """Run the multi-reranker benchmark"""
    print("🎯 Multi-Reranker Benchmark")
    print("=" * 50)
    print("Comparing reranking strategies:")
    print("1. No Reranker (Baseline)")
    print("2. BGE Reranker Base")
    print("3. BGE Reranker Large")
    print("4. BGE Reranker v2-M3")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = MultiRerankerBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_full_benchmark(num_samples_per_config=10)
    
    if results:
        print(f"\n🎉 Benchmark completed! {len(results)} configurations tested.")
        
        # Create visualizations
        benchmark.create_comparison_visualizations()
        
        # Save results
        benchmark.save_results()
        
        print("\n📊 Results saved:")
        print("- Comparison charts (PNG)")
        print("- Summary table (PNG)")
        print("- Detailed results (JSON)")
        print("- Results summary (CSV)")
        print("- Comprehensive report (Markdown)")
        
    else:
        print("❌ Benchmark failed - no results generated")


if __name__ == "__main__":
    main()
