"""
BGE Reranker Benchmark: Evaluating BGE Reranking on HotpotQA Dataset
====================================================================

Comprehensive benchmark comparing RAG performance with and without BGE reranking
using all-MiniLM-L6-v2 embeddings on 100 samples from HotpotQA dataset.

Metrics Evaluated:
- Precision@K (K=1,3,5,10): Fraction of retrieved documents that are relevant
- Recall@K (K=1,3,5,10): Fraction of relevant documents that are retrieved  
- NDCG@K (K=1,3,5,10): Normalized Discounted Cumulative Gain
- MAP (Mean Average Precision): Quality of document ranking
- MRR (Mean Reciprocal Rank): Position of first relevant document
- Hit Rate@K: Whether at least one relevant document is retrieved
- Retrieval Time: Time taken for document retrieval
- Reranking Time: Additional time for BGE reranking

Date: May 30, 2025
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

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

# Vector database and BGE reranker imports
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

# BGE Reranker
from services.rerank_service import BGEReranker, is_reranker_available

# HuggingFace datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    os.system("pip install datasets")
    from datasets import load_dataset

# Setup logging with UTF-8 encoding to avoid Unicode errors
def setup_logging():
    """Setup logging with proper UTF-8 encoding"""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        f'bge_reranker_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # Force UTF-8 encoding for console output
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Enhanced BGE Reranker Configuration
class EnhancedBGEBenchmarkConfig:
    """Configuration for enhanced BGE benchmark"""
      # Dataset configuration
    SAMPLE_SIZE = 100
    DATASET_NAME = "hotpot_qa"
    DATASET_SPLIT = "distractor"  # Available: 'distractor', 'fullwiki'
    DATASET_SUBSET = "validation"  # train, validation, test
    
    # Embedding model configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Retrieval configuration
    RETRIEVAL_TOP_K = 20
    RERANKING_INITIAL_K = 50
    
    # Evaluation metrics
    K_VALUES = [1, 3, 5, 10]
    
    # ChromaDB configuration
    CHROMA_PERSIST_PATH = "../db_data"
    
    # BGE Reranker configuration
    BGE_MODEL_NAME = "BAAI/bge-reranker-base"
    
    # Output configuration
    ENABLE_VISUALIZATIONS = True
    SAVE_DETAILED_RESULTS = True

class BGERerankerBenchmark:
    """Enhanced comprehensive benchmark for BGE reranker evaluation"""
    
    def __init__(self, config: EnhancedBGEBenchmarkConfig):
        """
        Initialize the benchmark
        
        Args:
            config: Configuration object for the benchmark
        """
        self.config = config
        self.embedding_model_name = config.EMBEDDING_MODEL
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.bge_reranker = None
        
        # Benchmark configuration
        self.sample_size = config.SAMPLE_SIZE
        self.k_values = config.K_VALUES
        self.collection_name = f"enhanced_hotpot_benchmark_{int(time.time())}"
        
        # Results storage
        self.results = {
            'without_reranking': [],
            'with_reranking': [],
            'metadata': {
                'embedding_model': config.EMBEDDING_MODEL,
                'sample_size': config.SAMPLE_SIZE,
                'timestamp': datetime.now().isoformat(),
                'k_values': config.K_VALUES,
                'dataset': f"{config.DATASET_NAME}_{config.DATASET_SPLIT}",
                'collection_name': self.collection_name,
                'retrieval_top_k': config.RETRIEVAL_TOP_K,
                'reranking_initial_k': config.RERANKING_INITIAL_K,
                'bge_model': config.BGE_MODEL_NAME
            }
        }
        
        # Track sample to relevant documents mapping
        self.sample_to_relevant_docs = {}
        
    def setup(self):
        """Initialize models and database"""
        logger.info("Setting up BGE Reranker Benchmark...")
        
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✅ Embedding model loaded")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path=EnhancedBGEBenchmarkConfig.CHROMA_PERSIST_PATH)
            
            # Create collection
            try:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "BGE Reranker Benchmark Collection"}
                )
                logger.info(f"✅ Created collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Collection might exist, trying to get: {e}")
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            
            # Initialize BGE Reranker
            if is_reranker_available():
                logger.info("Initializing BGE Reranker...")
                self.bge_reranker = BGEReranker(EnhancedBGEBenchmarkConfig.BGE_MODEL_NAME)
                if self.bge_reranker.is_available():
                    logger.info(f"✅ BGE Reranker loaded: {self.bge_reranker.model_name}")
                else:
                    logger.error("❌ BGE Reranker failed to load")
                    return False
            else:
                logger.error("❌ BGE Reranker not available")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            traceback.print_exc()
            return False    
    def load_hotpot_data(self) -> List[Dict]:
        """Load HotpotQA dataset samples"""
        logger.info(f"Loading {self.config.SAMPLE_SIZE} samples from HotpotQA dataset...")
        
        try:
            # Load HotpotQA with correct config and split
            logger.info(f"Loading dataset: {self.config.DATASET_NAME}, config: {self.config.DATASET_SPLIT}")
            dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_SPLIT)
            
            # Use validation split
            val_data = dataset[self.config.DATASET_SUBSET]
            logger.info(f"Dataset loaded: {len(val_data)} samples available")
            
            # Sample random entries
            total_samples = len(val_data)
            sample_size = min(self.config.SAMPLE_SIZE, total_samples)
            indices = random.sample(range(total_samples), sample_size)
            samples = []
            
            for idx in indices:
                sample = val_data[idx]
                
                # Extract relevant document titles for evaluation
                supporting_facts = sample.get('supporting_facts', [])
                relevant_titles = set([fact[0] for fact in supporting_facts])
                
                samples.append({
                    'id': sample.get('id', f'sample_{idx}'),
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'context': sample['context'],
                    'supporting_facts': supporting_facts,
                    'relevant_titles': relevant_titles,
                    'level': sample.get('level', 'unknown')
                })
            
            logger.info(f"✅ Loaded {len(samples)} samples from HotpotQA")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load HotpotQA data: {e}")
            traceback.print_exc()
            return []
    
    def index_documents(self, samples: List[Dict]):
        """Index documents in ChromaDB with enhanced evaluation setup"""
        logger.info("Indexing documents in ChromaDB for enhanced benchmark...")
        
        try:
            documents = []
            metadatas = []
            ids = []
            doc_id = 0
            
            # Track which documents are relevant for each sample
            sample_to_relevant_docs = {}
            
            for sample_idx, sample in enumerate(samples):
                context = sample['context']
                sample_id = sample['id']
                relevant_titles = sample['relevant_titles']
                
                # Track relevant documents for this sample
                sample_to_relevant_docs[sample_id] = []
                
                # Each context contains multiple [title, sentences] pairs
                for title_sentences in context:
                    title = title_sentences[0]
                    sentences = title_sentences[1]
                    
                    # Create document from title and sentences
                    doc_text = f"Title: {title}\n" + " ".join(sentences)
                    
                    is_relevant = title in relevant_titles
                    if is_relevant:
                        sample_to_relevant_docs[sample_id].append(doc_id)
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'title': title,
                        'sample_id': sample_id,
                        'sample_idx': sample_idx,
                        'is_relevant': is_relevant,
                        'doc_type': 'context'
                    })
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1
            
            # Add some noise documents from other samples to make retrieval more challenging
            logger.info("Adding additional documents for more realistic evaluation...")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True, convert_to_tensor=False)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            # Store sample to relevant docs mapping
            self.sample_to_relevant_docs = sample_to_relevant_docs
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            logger.info(f"Average relevant docs per sample: {np.mean([len(docs) for docs in sample_to_relevant_docs.values()]):.2f}")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            traceback.print_exc()
    
    def calculate_metrics(self, retrieved_docs: List[Dict], sample: Dict, 
                         retrieval_time: float, reranking_time: float = 0.0) -> Dict:
        """Calculate comprehensive retrieval metrics with enhanced evaluation"""
        
        sample_id = sample['id']
        relevant_titles = sample['relevant_titles']
        
        # Extract retrieved titles and check relevance more carefully
        retrieved_titles = []
        relevant_positions = []
        
        for i, doc in enumerate(retrieved_docs):
            title = doc['metadata']['title']
            retrieved_titles.append(title)
            
            # Check if this document is relevant to the current sample
            doc_sample_id = doc['metadata'].get('sample_id', '')
            is_relevant = (doc_sample_id == sample_id and title in relevant_titles)
            
            if is_relevant:
                relevant_positions.append(i + 1)  # 1-indexed position
        
        metrics = {
            'sample_id': sample_id,
            'num_relevant_total': len(relevant_titles),
            'num_retrieved': len(retrieved_docs),
            'relevant_positions': relevant_positions,
            'retrieval_time': retrieval_time,
            'reranking_time': reranking_time,
            'total_time': retrieval_time + reranking_time
        }
        
        # Calculate metrics for each k
        for k in self.k_values:
            retrieved_k_titles = set(retrieved_titles[:k])
            relevant_in_k = retrieved_k_titles & relevant_titles
            
            # Precision@k
            precision = len(relevant_in_k) / k if k > 0 else 0.0
            
            # Recall@k  
            recall = len(relevant_in_k) / len(relevant_titles) if len(relevant_titles) > 0 else 0.0
            
            # Hit Rate@k (whether at least one relevant doc is retrieved)
            hit_rate = 1.0 if len(relevant_in_k) > 0 else 0.0
            
            # F1 Score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
            metrics[f'hit_rate@{k}'] = hit_rate
            metrics[f'f1@{k}'] = f1
        
        # Calculate NDCG@k
        for k in self.k_values:
            relevance_scores = []
            for i, title in enumerate(retrieved_titles[:k]):
                # Check if document is relevant to this specific sample
                doc_sample_id = retrieved_docs[i]['metadata'].get('sample_id', '')
                is_relevant = (doc_sample_id == sample_id and title in relevant_titles)
                relevance_scores.append(1.0 if is_relevant else 0.0)
            
            if relevance_scores and max(relevance_scores) > 0:
                try:
                    # Create ideal ranking (all relevant docs first)
                    ideal_scores = sorted(relevance_scores, reverse=True)
                    ndcg = ndcg_score([ideal_scores], [relevance_scores], k=k)
                    metrics[f'ndcg@{k}'] = ndcg
                except:
                    metrics[f'ndcg@{k}'] = 0.0
            else:
                metrics[f'ndcg@{k}'] = 0.0
        
        # Mean Reciprocal Rank (MRR) - position of first relevant document
        mrr = 0.0
        for i, title in enumerate(retrieved_titles):
            doc_sample_id = retrieved_docs[i]['metadata'].get('sample_id', '')
            is_relevant = (doc_sample_id == sample_id and title in relevant_titles)
            if is_relevant:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        # Mean Average Precision (MAP)
        ap = 0.0
        relevant_count = 0
        for i, title in enumerate(retrieved_titles):
            doc_sample_id = retrieved_docs[i]['metadata'].get('sample_id', '')
            is_relevant = (doc_sample_id == sample_id and title in relevant_titles)
            if is_relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        
        if len(relevant_titles) > 0:
            ap = ap / len(relevant_titles)
        metrics['map'] = ap
        
        return metrics
    
    def query_without_reranking(self, question: str, top_k: int = 20) -> Tuple[List[Dict], float]:
        """Query documents without reranking"""
        start_time = time.time()
        
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieval_time = time.time() - start_time
        
        # Format results
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return documents, retrieval_time
    
    def query_with_reranking(self, question: str, top_k: int = 20, initial_k: int = 50) -> Tuple[List[Dict], float, float]:
        """Query documents with BGE reranking"""
        
        # Step 1: Initial retrieval (get more candidates for reranking)
        start_time = time.time()
        
        results = self.collection.query(
            query_texts=[question],
            n_results=initial_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieval_time = time.time() - start_time
        
        # Step 2: BGE reranking
        rerank_start_time = time.time()
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        if self.bge_reranker and self.bge_reranker.is_available():
            # Rerank using BGE
            reranked_docs, reranked_metadatas, relevance_scores = self.bge_reranker.rerank_documents(
                query=question,
                documents=documents,
                metadatas=metadatas,
                top_k=top_k,
                return_scores=True
            )
            
            reranking_time = time.time() - rerank_start_time
            
            # Format results
            formatted_results = []
            for i in range(len(reranked_docs)):
                formatted_results.append({
                    'document': reranked_docs[i],
                    'metadata': reranked_metadatas[i],
                    'relevance_score': relevance_scores[i]
                })
            
            return formatted_results, retrieval_time, reranking_time
        else:
            # Fallback to original results
            reranking_time = time.time() - rerank_start_time
            
            formatted_results = []
            for i in range(min(top_k, len(documents))):
                formatted_results.append({
                    'document': documents[i],
                    'metadata': metadatas[i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results, retrieval_time, reranking_time
    
    def run_benchmark(self, samples: List[Dict]):
        """Run the complete benchmark"""
        logger.info(f"Running BGE reranker benchmark on {len(samples)} samples...")
        
        total_samples = len(samples)
        
        for i, sample in enumerate(samples):
            question = sample['question']
            relevant_titles = sample['relevant_titles']
            
            logger.info(f"Processing sample {i+1}/{total_samples}: {question[:50]}...")
            
            try:                # Test WITHOUT reranking
                docs_without, time_without = self.query_without_reranking(question, top_k=20)
                metrics_without = self.calculate_metrics(
                    docs_without, sample, time_without, 0.0
                )
                self.results['without_reranking'].append(metrics_without)
                
                # Test WITH reranking
                docs_with, time_retrieval, time_reranking = self.query_with_reranking(
                    question, top_k=20, initial_k=50
                )
                metrics_with = self.calculate_metrics(
                    docs_with, sample, time_retrieval, time_reranking
                )
                self.results['with_reranking'].append(metrics_with)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i+1}/{total_samples} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {e}")
                continue
        
        logger.info("✅ Benchmark completed!")
    
    def analyze_results(self) -> Dict:
        """Analyze and compare results"""
        logger.info("Analyzing benchmark results...")
        
        without_df = pd.DataFrame(self.results['without_reranking'])
        with_df = pd.DataFrame(self.results['with_reranking'])
        
        # Calculate summary statistics
        summary = {}
        
        metric_columns = [col for col in without_df.columns if any(
            metric in col for metric in ['precision', 'recall', 'ndcg', 'hit_rate', 'f1', 'mrr', 'map', 'time']
        )]
        
        for metric in metric_columns:
            if metric in without_df.columns and metric in with_df.columns:
                without_mean = without_df[metric].mean()
                with_mean = with_df[metric].mean()
                improvement = ((with_mean - without_mean) / without_mean * 100) if without_mean > 0 else 0
                
                summary[metric] = {
                    'without_reranking': {
                        'mean': without_mean,
                        'std': without_df[metric].std(),
                        'median': without_df[metric].median()
                    },
                    'with_reranking': {
                        'mean': with_mean,
                        'std': with_df[metric].std(),
                        'median': with_df[metric].median()
                    },
                    'improvement_percent': improvement
                }
        
        return summary
    
    def save_results(self):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = f"bge_reranker_benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"✅ Raw results saved to: {results_file}")
        
        # Save summary analysis
        summary = self.analyze_results()
        summary_file = f"bge_reranker_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Summary saved to: {summary_file}")
          # Save as CSV for easy analysis
        without_df = pd.DataFrame(self.results['without_reranking'])
        with_df = pd.DataFrame(self.results['with_reranking'])
        
        csv_file = f"bge_reranker_benchmark_comparison_{timestamp}.csv"
        comparison_df = pd.DataFrame({
            'metric': [],
            'without_reranking_mean': [],
            'with_reranking_mean': [],
            'improvement_percent': []
        })
        
        for metric, data in summary.items():
            if 'improvement_percent' in data:
                comparison_df = pd.concat([comparison_df, pd.DataFrame({
                    'metric': [metric],
                    'without_reranking_mean': [data['without_reranking']['mean']],
                    'with_reranking_mean': [data['with_reranking']['mean']],
                    'improvement_percent': [data['improvement_percent']]
                })], ignore_index=True)
        
        comparison_df.to_csv(csv_file, index=False)
        logger.info(f"✅ Comparison CSV saved to: {csv_file}")
        
        return results_file, summary_file, csv_file
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        logger.info("Creating visualizations...")
        
        try:
            summary = self.analyze_results()
            
            # Extract metrics for plotting
            metrics = []
            without_means = []
            with_means = []
            improvements = []
            
            for metric, data in summary.items():
                if 'improvement_percent' in data and not metric.endswith('_time'):
                    metrics.append(metric)
                    without_means.append(data['without_reranking']['mean'])
                    with_means.append(data['with_reranking']['mean'])
                    improvements.append(data['improvement_percent'])
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Side-by-side comparison
            x = range(len(metrics))
            width = 0.35
            
            ax1.bar([i - width/2 for i in x], without_means, width, label='Without Reranking', alpha=0.8)
            ax1.bar([i + width/2 for i in x], with_means, width, label='With BGE Reranking', alpha=0.8)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('BGE Reranker Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Improvement percentages
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax2.bar(range(len(metrics)), improvements, color=colors, alpha=0.7)
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('BGE Reranking Improvement Percentages')
            ax2.set_xticks(range(len(metrics)))
            ax2.set_xticklabels(metrics, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"bge_reranker_benchmark_comparison_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {plot_file}")
            return plot_file
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up resources with proper error handling"""
        try:
            if self.chroma_client and self.collection:
                # Try to delete collection
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Collection {self.collection_name} cleaned up successfully")
                except Exception as e:
                    logger.warning(f"Could not delete collection {self.collection_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

def main():
    """Main benchmark execution"""
    print("🎯 Enhanced BGE Reranker Benchmark")
    print("=" * 60)
    print(f"Embedding Model: {EnhancedBGEBenchmarkConfig.EMBEDDING_MODEL}")
    print(f"Sample Size: {EnhancedBGEBenchmarkConfig.SAMPLE_SIZE}")
    print(f"Dataset: {EnhancedBGEBenchmarkConfig.DATASET_NAME} ({EnhancedBGEBenchmarkConfig.DATASET_SPLIT})")
    print(f"BGE Model: {EnhancedBGEBenchmarkConfig.BGE_MODEL_NAME}")
    print(f"Metrics: Precision@K, Recall@K, NDCG@K, MAP, MRR, Hit Rate@K")
    print("=" * 60)
    
    # Initialize benchmark
    config = EnhancedBGEBenchmarkConfig()
    benchmark = BGERerankerBenchmark(config)
    
    try:
        # Setup
        if not benchmark.setup():
            logger.error("❌ Benchmark setup failed")
            return
        
        # Load data
        samples = benchmark.load_hotpot_data()
        if not samples:
            logger.error("❌ Failed to load HotpotQA data")
            return
        
        # Index documents
        benchmark.index_documents(samples)
        
        # Run benchmark
        benchmark.run_benchmark(samples)
        
        # Save results
        results_file, summary_file, csv_file = benchmark.save_results()
        
        # Create visualizations
        plot_file = benchmark.create_visualizations()
        
        # Print summary
        summary = benchmark.analyze_results()
        print("\n🏆 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        key_metrics = ['precision@5', 'recall@5', 'ndcg@5', 'map', 'mrr', 'hit_rate@5']
        for metric in key_metrics:
            if metric in summary:
                data = summary[metric]
                print(f"{metric.upper():12s}: "
                      f"Without: {data['without_reranking']['mean']:.4f} | "
                      f"With: {data['with_reranking']['mean']:.4f} | "
                      f"Improvement: {data['improvement_percent']:+.2f}%")
        
        # Time analysis
        if 'total_time' in summary:
            time_data = summary['total_time']
            print(f"{'TOTAL_TIME':12s}: "
                  f"Without: {time_data['without_reranking']['mean']:.4f}s | "
                  f"With: {time_data['with_reranking']['mean']:.4f}s | "
                  f"Overhead: {time_data['improvement_percent']:+.2f}%")
        
        print("\n📊 Generated Files:")
        print(f"📄 Raw Results: {results_file}")
        print(f"📊 Summary: {summary_file}")
        print(f"📈 CSV: {csv_file}")
        if plot_file:
            print(f"📈 Visualization: {plot_file}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        benchmark.cleanup()
    
    print("\n🏁 BGE Reranker Benchmark Completed!")

if __name__ == "__main__":
    main()
