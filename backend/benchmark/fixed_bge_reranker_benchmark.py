"""
Fixed BGE Reranker Benchmark: Corrected HotpotQA Context Processing
================================================================

This version correctly handles the HotpotQA context structure where:
- context['title'] is a list of document titles
- context['sentences'] is a list of sentence lists

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
        f'fixed_bge_reranker_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
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

# BGE Reranker Configuration
class FixedBGEBenchmarkConfig:
    """Configuration for fixed BGE benchmark"""
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

class FixedBGERerankerBenchmark:
    """Fixed comprehensive benchmark for BGE reranker evaluation with correct context processing"""
    
    def __init__(self, config: FixedBGEBenchmarkConfig):
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
        self.collection_name = f"fixed_hotpot_benchmark_{int(time.time())}"
        
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
        
        # Enhanced relevance tracking
        self.all_relevant_titles = {}  # Maps sample_id to set of relevant titles
        self.title_to_content = {}     # Maps title to document content for better matching
        
    def setup(self):
        """Initialize models and database"""
        logger.info("Setting up Fixed BGE Reranker Benchmark...")
        
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✅ Embedding model loaded")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path=FixedBGEBenchmarkConfig.CHROMA_PERSIST_PATH)
            
            # Create collection with unique name
            try:
                # Try to delete existing collection first
                try:
                    existing_collection = self.chroma_client.get_collection(name=self.collection_name)
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except:
                    pass
                
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Fixed BGE Reranker Benchmark Collection"}
                )
                logger.info(f"✅ Created collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                return False
            
            # Initialize BGE Reranker
            if is_reranker_available():
                logger.info("Initializing BGE Reranker...")
                self.bge_reranker = BGEReranker(FixedBGEBenchmarkConfig.BGE_MODEL_NAME)
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
    
    def load_samples(self) -> List[Dict]:
        """Load samples from HotpotQA dataset with corrected context processing"""
        logger.info(f"Loading {self.sample_size} samples from HotpotQA dataset...")
        logger.info(f"Loading dataset: {self.config.DATASET_NAME}, config: {self.config.DATASET_SPLIT}")
        
        try:
            dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_SPLIT)
            dataset_split = dataset[self.config.DATASET_SUBSET]
            
            logger.info(f"Dataset loaded: {len(dataset_split)} samples available")
            
            # Select random samples
            random.seed(42)  # For reproducibility
            total_samples = len(dataset_split)
            selected_indices = random.sample(range(total_samples), min(self.sample_size, total_samples))
            
            samples = []
            for idx in selected_indices:
                sample = dataset_split[idx]
                
                # Extract relevant document titles for evaluation (corrected structure)
                supporting_facts = sample.get('supporting_facts', {})
                relevant_titles = set(supporting_facts.get('title', []))
                
                # Process context correctly - it's a dict with 'title' and 'sentences' keys
                context = sample['context']
                processed_context = []
                
                # Pair titles with their corresponding sentences
                titles = context.get('title', [])
                sentences = context.get('sentences', [])
                
                for i, title in enumerate(titles):
                    if i < len(sentences):
                        processed_context.append([title, sentences[i]])
                    else:
                        processed_context.append([title, []])
                
                # Store enhanced sample data
                enhanced_sample = {
                    'id': sample.get('id', f'sample_{idx}'),
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'context': processed_context,  # Now in the expected format
                    'supporting_facts': supporting_facts,
                    'relevant_titles': relevant_titles,
                    'sample_idx': idx
                }
                samples.append(enhanced_sample)
                
                # Store for global relevance tracking
                self.all_relevant_titles[enhanced_sample['id']] = relevant_titles
            
            logger.info(f"✅ Loaded {len(samples)} samples from HotpotQA")
            
            # Log statistics about the processed data
            total_docs = sum(len(sample['context']) for sample in samples)
            total_relevant = sum(len(sample['relevant_titles']) for sample in samples)
            logger.info(f"📊 Total documents across all samples: {total_docs}")
            logger.info(f"📊 Total relevant documents: {total_relevant}")
            logger.info(f"📊 Average docs per sample: {total_docs / len(samples):.1f}")
            logger.info(f"📊 Average relevant docs per sample: {total_relevant / len(samples):.1f}")
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load samples: {e}")
            traceback.print_exc()
            return []
    
    def index_documents(self, samples: List[Dict]):
        """Index documents in ChromaDB with corrected document processing"""
        logger.info("Indexing documents in ChromaDB for fixed benchmark...")
        
        try:
            documents = []
            metadatas = []
            ids = []
            doc_id = 0
            
            # First pass: collect all unique titles and their content
            title_content_map = {}
            
            for sample in samples:
                context = sample['context']
                for title_sentences in context:
                    title = title_sentences[0]
                    sentences = title_sentences[1]
                    doc_text = f"Title: {title}\n" + " ".join(sentences)
                    
                    # Store the most complete version of each title's content
                    if title not in title_content_map or len(doc_text) > len(title_content_map[title]):
                        title_content_map[title] = doc_text
            
            self.title_to_content = title_content_map
            logger.info(f"Found {len(title_content_map)} unique document titles")
            
            # Second pass: create document collection with all unique documents
            processed_titles = set()
            
            for sample in samples:
                context = sample['context']
                sample_id = sample['id']
                
                for title_sentences in context:
                    title = title_sentences[0]
                    
                    # Only add each unique title once to avoid duplicates
                    if title in processed_titles:
                        continue
                    
                    processed_titles.add(title)
                    sentences = title_sentences[1]
                    doc_text = f"Title: {title}\n" + " ".join(sentences)
                    
                    # Check if this title is relevant to ANY sample
                    is_relevant_to_any = any(title in sample_titles for sample_titles in self.all_relevant_titles.values())
                    
                    documents.append(doc_text)
                    metadatas.append({
                        'title': title,
                        'doc_type': 'context',
                        'is_relevant_to_any': is_relevant_to_any,
                        'original_sample_id': sample_id  # Track which sample this originally came from
                    })
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1
            
            logger.info(f"Generating embeddings for {len(documents)} unique documents...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True, convert_to_tensor=False)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            logger.info(f"Successfully indexed {len(documents)} unique documents")
            relevant_docs = sum(1 for meta in metadatas if meta['is_relevant_to_any'])
            logger.info(f"Documents relevant to any sample: {relevant_docs}")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            traceback.print_exc()
    
    def calculate_relevance_for_sample(self, retrieved_docs: List[Dict], sample: Dict) -> List[float]:
        """Calculate relevance scores for retrieved documents with respect to a specific sample"""
        sample_id = sample['id']
        relevant_titles = sample['relevant_titles']
        
        relevance_scores = []
        for doc in retrieved_docs:
            title = doc['metadata']['title']
            # A document is relevant if its title is in the sample's relevant titles
            is_relevant = title in relevant_titles
            relevance_scores.append(1.0 if is_relevant else 0.0)
            
        return relevance_scores
    
    def calculate_metrics(self, retrieved_docs: List[Dict], sample: Dict, 
                         retrieval_time: float, reranking_time: float = 0.0) -> Dict:
        """Calculate comprehensive retrieval metrics with corrected relevance evaluation"""
        
        sample_id = sample['id']
        relevant_titles = sample['relevant_titles']
        
        # Get relevance scores for all retrieved documents
        relevance_scores = self.calculate_relevance_for_sample(retrieved_docs, sample)
        
        # Extract retrieved titles and find relevant positions
        retrieved_titles = [doc['metadata']['title'] for doc in retrieved_docs]
        relevant_positions = []
        
        for i, (title, relevance) in enumerate(zip(retrieved_titles, relevance_scores)):
            if relevance > 0:
                relevant_positions.append(i + 1)  # 1-indexed position
        
        metrics = {
            'sample_id': sample_id,
            'num_relevant_total': len(relevant_titles),
            'num_retrieved': len(retrieved_docs),
            'relevant_positions': relevant_positions,
            'retrieval_time': retrieval_time,
            'reranking_time': reranking_time,
            'total_time': retrieval_time + reranking_time,
            'relevance_scores': relevance_scores
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
        
        # Calculate NDCG@k using corrected relevance scores
        for k in self.k_values:
            k_relevance = relevance_scores[:k]
            
            if k_relevance and max(k_relevance) > 0:
                try:
                    # Create ideal ranking (all relevant docs first)
                    ideal_scores = sorted(k_relevance, reverse=True)
                    ndcg = ndcg_score([ideal_scores], [k_relevance], k=len(k_relevance))
                    metrics[f'ndcg@{k}'] = ndcg
                except:
                    metrics[f'ndcg@{k}'] = 0.0
            else:
                metrics[f'ndcg@{k}'] = 0.0
        
        # Mean Reciprocal Rank (MRR) - position of first relevant document
        mrr = 0.0
        for i, relevance in enumerate(relevance_scores):
            if relevance > 0:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        # Mean Average Precision (MAP)
        ap = 0.0
        relevant_count = 0
        for i, relevance in enumerate(relevance_scores):
            if relevance > 0:
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
        
        # Step 2: Rerank with BGE
        start_time = time.time()        
        # Format for reranking
        documents = []
        metadatas = []
        for i in range(len(results['documents'][0])):
            documents.append(results['documents'][0][i])
            metadatas.append(results['metadatas'][0][i])
        
        if documents and self.bge_reranker:
            logger.info(f"Reranking {len(documents)} documents with BGE reranker")
            
            # Rerank documents using rerank_documents method
            reranked_docs, reranked_metadatas, relevance_scores = self.bge_reranker.rerank_documents(
                question, documents, metadatas, top_k
            )
            
            reranking_time = time.time() - start_time
            logger.info(f"Reranking completed in {reranking_time:.3f}s")
            
            # Build reordered documents with rerank scores
            reordered_documents = []
            for i, (doc, metadata, score) in enumerate(zip(reranked_docs, reranked_metadatas, relevance_scores)):
                reordered_documents.append({
                    'document': doc,
                    'metadata': metadata,
                    'distance': 0.0,  # Distance not meaningful after reranking
                    'rerank_score': score
                })
            
            # Log reranking insights
            if reordered_documents:
                top_title = reordered_documents[0]['metadata']['title']
                logger.info(f"Top reranked doc: {top_title[:50]}...")
                
            return reordered_documents, retrieval_time, reranking_time
        else:
            # Fallback to original order if reranking fails
            reranking_time = time.time() - start_time
            original_documents = []
            for i in range(min(top_k, len(documents))):
                original_documents.append({
                    'document': documents[i],
                    'metadata': metadatas[i],
                    'distance': results['distances'][0][i]
                })
            return original_documents, retrieval_time, reranking_time
    
    def run_benchmark(self):
        """Run the comprehensive benchmark"""
        logger.info(f"Running Fixed BGE reranker benchmark on {self.sample_size} samples...")
        
        # Load and index samples
        samples = self.load_samples()
        if not samples:
            logger.error("No samples loaded. Aborting benchmark.")
            return
        
        self.index_documents(samples)
        
        # Run evaluations
        for i, sample in enumerate(samples, 1):
            question = sample['question']
            sample_id = sample['id']
            
            logger.info(f"Processing sample {i}/{len(samples)}: {question[:50]}...")
            
            try:
                # Test without reranking
                retrieved_docs, retrieval_time = self.query_without_reranking(
                    question, top_k=self.config.RETRIEVAL_TOP_K
                )
                
                metrics_without = self.calculate_metrics(
                    retrieved_docs, sample, retrieval_time
                )
                self.results['without_reranking'].append(metrics_without)
                
                # Test with reranking
                retrieved_docs_rerank, retrieval_time_rerank, reranking_time = self.query_with_reranking(
                    question, 
                    top_k=self.config.RETRIEVAL_TOP_K,
                    initial_k=self.config.RERANKING_INITIAL_K
                )
                
                metrics_with = self.calculate_metrics(
                    retrieved_docs_rerank, sample, retrieval_time_rerank, reranking_time
                )
                self.results['with_reranking'].append(metrics_with)
                
                # Log sample results
                relevance_scores_without = metrics_without.get('relevance_scores', [])
                relevance_scores_with = metrics_with.get('relevance_scores', [])
                
                logger.info(f"Without reranking - Relevant docs in top-5: {sum(relevance_scores_without[:5])}")
                logger.info(f"With reranking - Relevant docs in top-5: {sum(relevance_scores_with[:5])}")
                
                # Show a preview every 10 samples
                if i % 10 == 0:
                    logger.info(f"📊 Progress: {i}/{len(samples)} samples processed")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                traceback.print_exc()
                continue
        
        logger.info("✅ Benchmark completed successfully!")
        
        # Generate results
        self.analyze_results()
        self.save_results()
        if self.config.ENABLE_VISUALIZATIONS:
            self.create_visualizations()
    
    def analyze_results(self):
        """Analyze and summarize benchmark results"""
        logger.info("Analyzing benchmark results...")
        
        # Calculate summary statistics
        without_results = pd.DataFrame(self.results['without_reranking'])
        with_results = pd.DataFrame(self.results['with_reranking'])
        
        summary = {}
        
        # List of metrics to analyze
        metrics_to_analyze = [
            'retrieval_time', 'reranking_time', 'total_time',
            'precision@1', 'precision@3', 'precision@5', 'precision@10',
            'recall@1', 'recall@3', 'recall@5', 'recall@10',
            'hit_rate@1', 'hit_rate@3', 'hit_rate@5', 'hit_rate@10',
            'f1@1', 'f1@3', 'f1@5', 'f1@10',
            'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10',
            'mrr', 'map'
        ]
        
        for metric in metrics_to_analyze:
            if metric in without_results.columns:
                without_values = without_results[metric].values
                with_values = with_results[metric].values
                
                without_mean = np.mean(without_values)
                with_mean = np.mean(with_values)
                
                improvement_percent = ((with_mean - without_mean) / without_mean * 100) if without_mean != 0 else 0
                
                summary[metric] = {
                    'without_reranking': {
                        'mean': float(without_mean),
                        'std': float(np.std(without_values)),
                        'median': float(np.median(without_values))
                    },
                    'with_reranking': {
                        'mean': float(with_mean),
                        'std': float(np.std(with_values)),
                        'median': float(np.median(with_values))
                    },
                    'improvement_percent': float(improvement_percent)
                }
        
        self.summary = summary
        
        # Log key improvements
        logger.info("🔍 Key Performance Improvements:")
        key_metrics = ['ndcg@5', 'mrr', 'map', 'precision@5', 'hit_rate@5']
        for metric in key_metrics:
            if metric in summary:
                improvement = summary[metric]['improvement_percent']
                without_mean = summary[metric]['without_reranking']['mean']
                with_mean = summary[metric]['with_reranking']['mean']
                logger.info(f"  {metric}: {without_mean:.4f} → {with_mean:.4f} ({improvement:+.2f}%)")
        
        time_overhead = summary.get('total_time', {}).get('improvement_percent', 0)
        logger.info(f"⏱️  Average time overhead: {time_overhead:+.2f}%")
    
    def save_results(self):
        """Save detailed results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"fixed_bge_reranker_benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        # Save summary statistics
        summary_file = f"fixed_bge_reranker_benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Summary statistics saved to: {summary_file}")
        
        # Save comparison CSV
        comparison_data = []
        for metric, data in self.summary.items():
            comparison_data.append({
                'metric': metric,
                'without_reranking_mean': data['without_reranking']['mean'],
                'with_reranking_mean': data['with_reranking']['mean'],
                'improvement_percent': data['improvement_percent']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        csv_file = f"fixed_bge_reranker_benchmark_comparison_{timestamp}.csv"
        comparison_df.to_csv(csv_file, index=False)
        logger.info(f"📈 Comparison data saved to: {csv_file}")
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        logger.info("Creating benchmark visualizations...")
        
        try:
            # Create comparison plot
            metrics_to_plot = ['ndcg@5', 'mrr', 'map', 'precision@5', 'recall@5', 'hit_rate@5']
            
            improvements = []
            metric_names = []
            
            for metric in metrics_to_plot:
                if metric in self.summary:
                    improvements.append(self.summary[metric]['improvement_percent'])
                    metric_names.append(metric.replace('@', ' @'))
            
            if improvements:
                plt.figure(figsize=(12, 8))
                
                colors = ['green' if imp > 0 else 'red' for imp in improvements]
                bars = plt.bar(metric_names, improvements, color=colors, alpha=0.7)
                
                plt.title('Fixed BGE Reranker Performance Improvements', fontsize=16, fontweight='bold')
                plt.ylabel('Improvement (%)', fontsize=12)
                plt.xlabel('Metrics', fontsize=12)
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, improvement in zip(bars, improvements):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{improvement:+.1f}%',
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontweight='bold')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_file = f"fixed_bge_reranker_benchmark_comparison_{timestamp}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Visualization saved to: {plot_file}")
                plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.chroma_client and self.collection:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"🧹 Cleaned up collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup collection: {e}")

def main():
    """Main function to run the fixed benchmark"""
    logger.info("🚀 Starting Fixed BGE Reranker Benchmark")
    logger.info("=" * 60)
    
    # Create configuration
    config = FixedBGEBenchmarkConfig()
    
    # Display configuration
    logger.info("📋 Benchmark Configuration:")
    logger.info(f"  Dataset: {config.DATASET_NAME} ({config.DATASET_SPLIT})")
    logger.info(f"  Sample Size: {config.SAMPLE_SIZE}")
    logger.info(f"  Embedding Model: {config.EMBEDDING_MODEL}")
    logger.info(f"  BGE Reranker: {config.BGE_MODEL_NAME}")
    logger.info(f"  Retrieval Top-K: {config.RETRIEVAL_TOP_K}")
    logger.info(f"  Reranking Initial-K: {config.RERANKING_INITIAL_K}")
    logger.info(f"  Evaluation K-values: {config.K_VALUES}")
    logger.info("=" * 60)
    
    # Initialize and run benchmark
    benchmark = FixedBGERerankerBenchmark(config)
    
    try:
        if benchmark.setup():
            benchmark.run_benchmark()
            logger.info("🎉 Fixed BGE Reranker Benchmark completed successfully!")
        else:
            logger.error("❌ Benchmark setup failed")
    except KeyboardInterrupt:
        logger.info("⚠️ Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
    finally:
        benchmark.cleanup()
        logger.info("👋 Benchmark finished")

if __name__ == "__main__":
    main()
