"""
Academic Vector Database Benchmark: Qdrant vs ChromaDB
=======================================================

A comprehensive, reproducible benchmark comparing Qdrant and ChromaDB vector databases
for Retrieval-Augmented Generation (RAG) systems using multiple embedding models.

Date: May 30, 2025
Version: 1.0

This benchmark follows academic standards for reproducibility and authenticity.
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

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class AcademicVectorDBBenchmark:
    """
    Academic-grade vector database benchmark for RAG systems.
    
    This benchmark evaluates Qdrant and ChromaDB across multiple dimensions:
    - Embedding models: BGE-M3, all-MiniLM-L6-v2
    - Data sizes: 500, 1000, 2500, 5000, 10000 documents
    - Metrics: Precision@k, Recall@k, NDCG@k, Hit Rate@k, Performance metrics
    - Query types: Semantic similarity, domain-specific queries
    """
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the benchmark with logging and reproducibility settings."""
        self.setup_logging(log_level)
        self.logger.info("=" * 80)
        self.logger.info("ACADEMIC VECTOR DATABASE BENCHMARK - QDRANT VS CHROMADB")
        self.logger.info("=" * 80)
          # Initialize database clients
        self.logger.info("[INIT] Initializing database clients...")
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.chroma_client = chromadb.Client()
        
        # Initialize embedding models
        self.logger.info("[MODELS] Loading embedding models...")
        self.embedding_models = self._load_embedding_models()
        
        # Benchmark configuration
        self.data_sizes = [500, 1000, 2500, 5000, 10000]
        self.k_values = [1, 3, 5, 10, 20]
        self.query_count = 100
        
        # Document corpus configuration
        self.domains = {
            'technology': {
                'topics': [
                    "artificial intelligence machine learning deep neural networks",
                    "cloud computing distributed systems microservices architecture",
                    "cybersecurity data protection encryption blockchain technology",
                    "software engineering agile development methodologies",
                    "database management systems NoSQL distributed databases",
                    "computer vision image recognition convolutional networks",
                    "natural language processing transformer models attention",
                    "web development frameworks REST APIs microservices",
                    "mobile application development cross-platform frameworks",
                    "data science analytics statistical modeling visualization"
                ],
                'weight': 0.3
            },
            'science': {
                'topics': [
                    "quantum physics particle mechanics wave functions",
                    "climate science environmental modeling carbon emissions",
                    "biotechnology genetic engineering CRISPR applications",
                    "space exploration planetary science astrophysics",
                    "renewable energy solar photovoltaic wind turbines",
                    "medical research pharmaceutical drug development",
                    "materials science nanotechnology carbon nanotubes",
                    "neuroscience brain imaging cognitive functions",
                    "chemistry molecular dynamics chemical reactions",
                    "mathematics topology differential equations statistics"
                ],
                'weight': 0.25
            },
            'business': {
                'topics': [
                    "strategic management competitive analysis market positioning",
                    "financial planning investment portfolio risk assessment",
                    "operations management supply chain logistics optimization",
                    "human resources talent acquisition performance management",
                    "marketing digital campaigns brand strategy analytics",
                    "entrepreneurship startup funding venture capital",
                    "project management agile scrum methodologies",
                    "business intelligence data analytics decision support",
                    "corporate governance compliance regulatory frameworks",
                    "innovation management technology transfer commercialization"
                ],
                'weight': 0.2
            },
            'healthcare': {
                'topics': [
                    "clinical medicine diagnostic procedures treatment protocols",
                    "public health epidemiology disease prevention strategies",
                    "pharmacology drug interactions therapeutic mechanisms",
                    "medical imaging radiology diagnostic technologies",
                    "nursing care patient safety quality improvement",
                    "mental health psychology therapeutic interventions",
                    "rehabilitation physical therapy occupational health",
                    "nutrition dietary interventions metabolic health",
                    "healthcare informatics electronic health records",
                    "biomedical engineering medical device development"
                ],
                'weight': 0.15
            },
            'education': {
                'topics': [
                    "pedagogical methods teaching strategies learning outcomes",
                    "educational technology online learning platforms",                    "curriculum development assessment methodologies",
                    "higher education research university administration",
                    "early childhood development educational psychology",
                    "special education inclusive learning environments",
                    "language learning second language acquisition",
                    "educational policy reform standards assessment",
                    "teacher training professional development programs",
                    "educational research qualitative quantitative methods"
                ],
                'weight': 0.1
            }
        }
        
        self.logger.info(f"Benchmark initialized with {len(self.embedding_models)} models")
        self.logger.info(f"Test configurations: {len(self.data_sizes)} data sizes × {len(self.embedding_models)} models")
        
    def setup_logging(self, level):
        """Setup comprehensive logging for academic reproducibility."""
        # Fix unicode encoding issues on Windows
        log_filename = f'academic_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Create handlers with UTF-8 encoding
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        # Configure formatting
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        logging.basicConfig(
            level=level,
            handlers=[file_handler, console_handler]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_embedding_models(self) -> Dict[str, Dict]:
        """Load and configure embedding models with metadata."""
        models = {}
        
        try:
            # BGE-M3: State-of-the-art multilingual embedding model
            self.logger.info("  Loading BGE-M3 (BAAI/bge-m3)...")
            bge_model = SentenceTransformer('BAAI/bge-m3')
            models['bge-m3'] = {
                'model': bge_model,
                'dimension': 1024,
                'description': 'BGE-M3: Multilingual, multi-granularity embedding model',
                'paper': 'BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity',
                'strengths': 'High semantic understanding, multilingual, domain adaptation'
            }
            self.logger.info(f"    [OK] BGE-M3 loaded (dimension: 1024)")
            
            # all-MiniLM-L6-v2: Efficient general-purpose model
            self.logger.info("  Loading all-MiniLM-L6-v2...")
            minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
            models['all-minilm-l6-v2'] = {
                'model': minilm_model,
                'dimension': 384,
                'description': 'all-MiniLM-L6-v2: Efficient general-purpose embedding model',
                'paper': 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
                'strengths': 'Fast inference, good general performance, compact'            }
            self.logger.info(f"    [OK] all-MiniLM-L6-v2 loaded (dimension: 384)")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load embedding models: {e}")
            raise
            
        return models
    
    def generate_academic_corpus(self, data_size: int) -> Tuple[List[str], Dict[str, any]]:
        """
        Generate academic corpus with documented methodology.
        
        Returns:
            documents: List of synthetic academic documents
            metadata: Corpus generation metadata for reproducibility
        """
        self.logger.info(f"[CORPUS] Generating academic corpus ({data_size} documents)...")
        
        documents = []
        doc_metadata = []
        
        # Document generation patterns for academic authenticity
        patterns = [
            "Introduction to {topic}: fundamental concepts and applications",
            "Advanced {topic}: theoretical frameworks and methodologies", 
            "Comprehensive survey of {topic} in modern research",
            "Recent developments in {topic}: a systematic review",
            "Practical applications of {topic} in industry settings",
            "Comparative analysis of {topic} approaches and techniques",
            "Future directions in {topic} research and development",
            "Case study: implementing {topic} in real-world scenarios",
            "Methodological considerations for {topic} research",
            "Critical evaluation of {topic} theories and practices"
        ]
        
        # Generate documents following domain distribution
        for doc_id in range(data_size):
            # Select domain based on weights
            domain_names = list(self.domains.keys())
            domain_weights = [self.domains[d]['weight'] for d in domain_names]
            domain = np.random.choice(domain_names, p=domain_weights)
            
            # Select topic and pattern
            topic = random.choice(self.domains[domain]['topics'])
            pattern = random.choice(patterns)
            
            # Generate document text
            doc_text = pattern.format(topic=topic)
            
            # Add domain-specific variations
            if domain == 'technology':
                doc_text += f" with focus on scalability and performance optimization."
            elif domain == 'science':
                doc_text += f" including experimental validation and theoretical modeling."
            elif domain == 'business':
                doc_text += f" considering market dynamics and strategic implications."
            elif domain == 'healthcare':
                doc_text += f" with emphasis on clinical efficacy and patient outcomes."
            elif domain == 'education':
                doc_text += f" incorporating pedagogical best practices and learning assessment."
            
            documents.append(doc_text)
            doc_metadata.append({
                'doc_id': doc_id,
                'domain': domain,
                'topic_keywords': topic.split()[:3],  # First 3 keywords
                'pattern': pattern            })
        
        # Create corpus metadata for reproducibility
        corpus_metadata = {
            'generation_date': datetime.now().isoformat(),
            'data_size': data_size,
            'random_seed': RANDOM_SEED,
            'domain_distribution': {d: sum(1 for m in doc_metadata if m['domain'] == d) 
                                  for d in self.domains.keys()},
            'pattern_distribution': {p: sum(1 for m in doc_metadata if m['pattern'] == p) 
                                   for p in patterns},
            'corpus_hash': hashlib.md5(''.join(documents).encode()).hexdigest(),
            'doc_metadata': doc_metadata  # Include document metadata
        }
        
        self.logger.info(f"  Generated {len(documents)} documents")
        self.logger.info(f"  Domain distribution: {corpus_metadata['domain_distribution']}")
        
        return documents, corpus_metadata
    
    def generate_academic_queries(self, documents: List[str], doc_metadata: List[Dict], 
                                query_count: int) -> Tuple[List[str], Dict[int, Set[int]], Dict]:
        """
        Generate academic queries with semantic relevance ground truth.
        
        Returns:
            queries: List of query strings
            ground_truth: Query ID -> Set of relevant document IDs
            query_metadata: Query generation metadata
        """
        self.logger.info(f"[QUERIES] Generating academic queries ({query_count} queries)...")
        
        queries = []
        ground_truth = {}
        query_metadata = []
        
        # Query templates for different information needs
        query_templates = {
            'definition': [
                "What is {concept}?",
                "Define {concept} and its applications",
                "Explain the fundamentals of {concept}"
            ],
            'methodology': [
                "How to implement {concept}?",
                "Best practices for {concept}",
                "Methodological approaches to {concept}"
            ],
            'comparison': [
                "Compare different approaches to {concept}",
                "Advantages and disadvantages of {concept}",
                "Comparative analysis of {concept} methods"
            ],
            'application': [
                "Real-world applications of {concept}",
                "Industry use cases for {concept}",
                "Practical implementation of {concept}"
            ],
            'research': [
                "Recent research in {concept}",
                "Current developments in {concept}",
                "Future directions for {concept}"
            ]
        }
        
        for query_id in range(query_count):
            # Select a base document for semantic relevance
            base_doc_id = random.randint(0, len(documents) - 1)
            base_doc_meta = doc_metadata[base_doc_id]
            
            # Extract key concepts from base document
            base_keywords = base_doc_meta['topic_keywords']
            concept = ' '.join(random.sample(base_keywords, min(2, len(base_keywords))))
            
            # Select query type and template
            query_type = random.choice(list(query_templates.keys()))
            template = random.choice(query_templates[query_type])
            query = template.format(concept=concept)
            
            queries.append(query)
            
            # Generate ground truth relevance
            relevant_docs = set()
            
            # Always include base document
            relevant_docs.add(base_doc_id)
            
            # Find semantically similar documents
            base_domain = base_doc_meta['domain']
            base_keywords_set = set(base_keywords)
            
            for doc_id, doc_meta in enumerate(doc_metadata):
                if doc_id == base_doc_id:
                    continue
                    
                # Same domain documents are potentially relevant
                if doc_meta['domain'] == base_domain:
                    # Check keyword overlap
                    doc_keywords_set = set(doc_meta['topic_keywords'])
                    overlap = len(base_keywords_set & doc_keywords_set)
                    
                    if overlap >= 1:  # At least one common keyword
                        relevant_docs.add(doc_id)
                
                # Cross-domain relevance for interdisciplinary topics
                elif len(base_keywords_set & set(doc_meta['topic_keywords'])) >= 2:
                    relevant_docs.add(doc_id)
            
            # Ensure minimum relevance set size
            if len(relevant_docs) < 3:
                # Add random documents from same domain
                domain_docs = [i for i, m in enumerate(doc_metadata) 
                             if m['domain'] == base_domain and i not in relevant_docs]
                additional_count = min(3 - len(relevant_docs), len(domain_docs))
                if domain_docs:
                    additional_docs = random.sample(domain_docs, additional_count)
                    relevant_docs.update(additional_docs)
            
            ground_truth[query_id] = relevant_docs
            query_metadata.append({
                'query_id': query_id,
                'query_type': query_type,
                'base_doc_id': base_doc_id,
                'base_domain': base_domain,
                'concept_keywords': base_keywords,
                'relevance_set_size': len(relevant_docs)
            })
        
        # Query generation metadata
        generation_metadata = {
            'generation_date': datetime.now().isoformat(),
            'query_count': query_count,
            'random_seed': RANDOM_SEED,
            'query_type_distribution': {qt: sum(1 for m in query_metadata if m['query_type'] == qt) 
                                      for qt in query_templates.keys()},
            'avg_relevance_set_size': np.mean([len(gt) for gt in ground_truth.values()]),
            'queries_hash': hashlib.md5(''.join(queries).encode()).hexdigest()
        }
        
        self.logger.info(f"  ✅ Generated {len(queries)} queries")
        self.logger.info(f"  📊 Query type distribution: {generation_metadata['query_type_distribution']}")
        self.logger.info(f"  📊 Average relevance set size: {generation_metadata['avg_relevance_set_size']:.2f}")
        
        return queries, ground_truth, generation_metadata
    
    def calculate_precision_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Precision@k with academic precision."""
        if k == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Recall@k with academic precision."""
        if len(relevant) == 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return relevant_retrieved / len(relevant)
    
    def calculate_f1_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate F1@k score."""
        precision = self.calculate_precision_at_k(retrieved, relevant, k)
        recall = self.calculate_recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_hit_rate_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Hit Rate@k (binary relevance)."""
        if len(relevant) == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return 1.0 if len(retrieved_k.intersection(relevant)) > 0 else 0.0
    
    def calculate_map_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Mean Average Precision@k."""
        if len(relevant) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_k):
            if doc_id in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if len(relevant) > 0 else 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate NDCG@k with academic standard implementation."""
        if len(relevant) == 0 or k == 0:
            return 0.0
        
        # Create relevance scores (binary: 1 for relevant, 0 for non-relevant)
        y_true = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
        
        # Pad to exactly k items if needed
        while len(y_true) < k:
            y_true.append(0)
        y_true = y_true[:k]
        
        # Calculate DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(y_true))
        
        # Calculate IDCG
        ideal_relevance = sorted(y_true, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def benchmark_single_configuration(self, model_name: str, data_size: int) -> Dict:
        """Benchmark a single configuration with comprehensive metrics."""
        config_id = f"{model_name}_{data_size}"
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[BENCHMARK] {model_name.upper()} | {data_size} documents")
        self.logger.info(f"{'='*60}")
        
        # Get embedding model
        model_config = self.embedding_models[model_name]
        embedding_model = model_config['model']
        vector_dim = model_config['dimension']
          # Generate corpus and queries
        documents, corpus_metadata = self.generate_academic_corpus(data_size)
        queries, ground_truth, query_metadata = self.generate_academic_queries(
            documents, corpus_metadata['doc_metadata'], self.query_count
        )
        
        # Generate embeddings
        self.logger.info("[EMBEDDINGS] Computing embeddings...")
        embed_start = time.time()
        doc_embeddings = embedding_model.encode(documents, show_progress_bar=False, convert_to_numpy=True)
        query_embeddings = embedding_model.encode(queries, show_progress_bar=False, convert_to_numpy=True)
        embedding_time = time.time() - embed_start
        
        self.logger.info(f"  ✅ Embeddings computed in {embedding_time:.2f}s")
        
        max_k = max(self.k_values)
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_dimension': vector_dim,
            'data_size': data_size,
            'query_count': self.query_count,
            'embedding_time': embedding_time,
            'corpus_hash': corpus_metadata['corpus_hash'],
            'queries_hash': query_metadata['queries_hash'] if 'queries_hash' in query_metadata else None
        }
        
        # Benchmark Qdrant
        self.logger.info("🔧 Benchmarking Qdrant...")
        try:
            qdrant_results = self._benchmark_qdrant(
                config_id, doc_embeddings, query_embeddings, 
                documents, vector_dim, ground_truth, max_k
            )
            results.update({f'qdrant_{k}': v for k, v in qdrant_results.items()})
            self.logger.info("  ✅ Qdrant benchmark completed")
        except Exception as e:
            self.logger.error(f"  ❌ Qdrant benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            # Fill with null values
            for metric in ['insert_time', 'search_time', 'search_per_query', 'qps', 'insert_throughput']:
                results[f'qdrant_{metric}'] = None
            for k in self.k_values:
                for metric in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate', 'map']:
                    results[f'qdrant_{metric}_at_{k}'] = None
        
        # Benchmark ChromaDB
        self.logger.info("🔧 Benchmarking ChromaDB...")
        try:
            chroma_results = self._benchmark_chromadb(
                config_id, doc_embeddings, query_embeddings, 
                documents, ground_truth, max_k
            )
            results.update({f'chroma_{k}': v for k, v in chroma_results.items()})
            self.logger.info("  ✅ ChromaDB benchmark completed")
        except Exception as e:
            self.logger.error(f"  ❌ ChromaDB benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            # Fill with null values
            for metric in ['insert_time', 'search_time', 'search_per_query', 'qps', 'insert_throughput']:
                results[f'chroma_{metric}'] = None
            for k in self.k_values:
                for metric in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate', 'map']:
                    results[f'chroma_{metric}_at_{k}'] = None
        
        self.logger.info(f"✅ Configuration {config_id} completed")
        return results
    
    def _benchmark_qdrant(self, config_id: str, doc_embeddings: np.ndarray, 
                         query_embeddings: np.ndarray, documents: List[str], 
                         vector_dim: int, ground_truth: Dict[int, Set[int]], 
                         max_k: int) -> Dict:
        """Benchmark Qdrant with detailed metrics."""
        collection_name = f"academic_{config_id}"
        
        # Clean up existing collection
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        
        # Insert documents
        insert_start = time.time()
        batch_size = min(100, len(doc_embeddings) // 10 + 1)
        
        for i in range(0, len(doc_embeddings), batch_size):
            end_idx = min(i + batch_size, len(doc_embeddings))
            points = [
                PointStruct(
                    id=j,
                    vector=doc_embeddings[j].tolist(),
                    payload={"text": documents[j], "doc_id": j}
                )
                for j in range(i, end_idx)
            ]
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
        
        insert_time = time.time() - insert_start
        insert_throughput = len(doc_embeddings) / insert_time
        
        # Search documents
        search_start = time.time()
        all_results = []
        
        for query_embedding in query_embeddings:
            search_result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding.tolist(),
                limit=max_k
            )
            retrieved_ids = [hit.id for hit in search_result.points]
            all_results.append(retrieved_ids)
        
        search_time = time.time() - search_start
        search_per_query = search_time / len(query_embeddings)
        qps = len(query_embeddings) / search_time
        
        # Calculate metrics for each k
        metrics = {
            'insert_time': insert_time,
            'search_time': search_time,
            'search_per_query': search_per_query,
            'qps': qps,
            'insert_throughput': insert_throughput
        }
        
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            map_scores = []
            
            for query_idx, retrieved_ids in enumerate(all_results):
                relevant_docs = ground_truth[query_idx]
                
                precision = self.calculate_precision_at_k(retrieved_ids, relevant_docs, k)
                recall = self.calculate_recall_at_k(retrieved_ids, relevant_docs, k)
                f1 = self.calculate_f1_at_k(retrieved_ids, relevant_docs, k)
                ndcg = self.calculate_ndcg_at_k(retrieved_ids, relevant_docs, k)
                hit_rate = self.calculate_hit_rate_at_k(retrieved_ids, relevant_docs, k)
                map_score = self.calculate_map_at_k(retrieved_ids, relevant_docs, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                ndcg_scores.append(ndcg)
                hit_rate_scores.append(hit_rate)
                map_scores.append(map_score)
            
            metrics[f'precision_at_{k}'] = np.mean(precision_scores)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
            metrics[f'f1_at_{k}'] = np.mean(f1_scores)
            metrics[f'ndcg_at_{k}'] = np.mean(ndcg_scores)
            metrics[f'hit_rate_at_{k}'] = np.mean(hit_rate_scores)
            metrics[f'map_at_{k}'] = np.mean(map_scores)
        
        return metrics
    
    def _benchmark_chromadb(self, config_id: str, doc_embeddings: np.ndarray, 
                           query_embeddings: np.ndarray, documents: List[str], 
                           ground_truth: Dict[int, Set[int]], max_k: int) -> Dict:
        """Benchmark ChromaDB with detailed metrics."""
        collection_name = f"academic_{config_id}"
        
        # Clean up existing collection
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection
        chroma_collection = self.chroma_client.create_collection(collection_name)
        
        # Insert documents
        insert_start = time.time()
        batch_size = min(100, len(doc_embeddings) // 10 + 1)
        
        for i in range(0, len(doc_embeddings), batch_size):
            end_idx = min(i + batch_size, len(doc_embeddings))
            chroma_collection.add(
                embeddings=doc_embeddings[i:end_idx].tolist(),
                ids=[f"doc_{j}" for j in range(i, end_idx)],
                metadatas=[{"text": documents[j]} for j in range(i, end_idx)]
            )
        
        insert_time = time.time() - insert_start
        insert_throughput = len(doc_embeddings) / insert_time
        
        # Search documents
        search_start = time.time()
        chroma_search_result = chroma_collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=max_k
        )
        search_time = time.time() - search_start
        search_per_query = search_time / len(query_embeddings)
        qps = len(query_embeddings) / search_time
        
        # Process results
        all_results = []
        for query_idx in range(len(query_embeddings)):
            retrieved_ids = [int(doc_id.split('_')[1]) for doc_id in chroma_search_result['ids'][query_idx]]
            all_results.append(retrieved_ids)
        
        # Calculate metrics for each k
        metrics = {
            'insert_time': insert_time,
            'search_time': search_time,
            'search_per_query': search_per_query,
            'qps': qps,
            'insert_throughput': insert_throughput
        }
        
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            map_scores = []
            
            for query_idx, retrieved_ids in enumerate(all_results):
                relevant_docs = ground_truth[query_idx]
                
                precision = self.calculate_precision_at_k(retrieved_ids, relevant_docs, k)
                recall = self.calculate_recall_at_k(retrieved_ids, relevant_docs, k)
                f1 = self.calculate_f1_at_k(retrieved_ids, relevant_docs, k)
                ndcg = self.calculate_ndcg_at_k(retrieved_ids, relevant_docs, k)
                hit_rate = self.calculate_hit_rate_at_k(retrieved_ids, relevant_docs, k)
                map_score = self.calculate_map_at_k(retrieved_ids, relevant_docs, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                ndcg_scores.append(ndcg)
                hit_rate_scores.append(hit_rate)
                map_scores.append(map_score)
            
            metrics[f'precision_at_{k}'] = np.mean(precision_scores)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
            metrics[f'f1_at_{k}'] = np.mean(f1_scores)
            metrics[f'ndcg_at_{k}'] = np.mean(ndcg_scores)
            metrics[f'hit_rate_at_{k}'] = np.mean(hit_rate_scores)
            metrics[f'map_at_{k}'] = np.mean(map_scores)
        
        return metrics
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run the complete academic benchmark."""
        self.logger.info("\n🚀 STARTING COMPREHENSIVE ACADEMIC BENCHMARK")
        self.logger.info("=" * 80)
        
        benchmark_start = time.time()
        all_results = []
        
        total_configs = len(self.embedding_models) * len(self.data_sizes)
        current_config = 0
        
        for model_name in self.embedding_models.keys():
            for data_size in self.data_sizes:
                current_config += 1
                self.logger.info(f"\n[{current_config}/{total_configs}] Testing {model_name} with {data_size} documents")
                
                try:
                    result = self.benchmark_single_configuration(model_name, data_size)
                    all_results.append(result)
                    
                    # Log progress
                    elapsed = time.time() - benchmark_start
                    estimated_total = elapsed * total_configs / current_config
                    remaining = estimated_total - elapsed
                    
                    self.logger.info(f"  ⏱️  Progress: {current_config}/{total_configs} "
                                   f"({current_config/total_configs*100:.1f}%)")
                    self.logger.info(f"  ⏱️  Elapsed: {elapsed/60:.1f}min, "
                                   f"Remaining: {remaining/60:.1f}min")
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Configuration failed: {e}")
                    self.logger.error(traceback.format_exc())
                    continue
        
        total_time = time.time() - benchmark_start
        self.logger.info(f"\n✅ BENCHMARK COMPLETED in {total_time/60:.2f} minutes")
        self.logger.info(f"📊 Generated {len(all_results)} result sets")
        
        return pd.DataFrame(all_results)
    
    def create_academic_analysis(self, results_df: pd.DataFrame) -> Dict:
        """Create comprehensive academic analysis with statistical rigor."""
        self.logger.info("\n📊 CREATING ACADEMIC ANALYSIS")
        self.logger.info("=" * 50)
        
        if results_df.empty:
            self.logger.warning("❌ No results to analyze")
            return {}
        
        analysis = {
            'benchmark_metadata': {
                'completion_date': datetime.now().isoformat(),
                'total_configurations': len(results_df),
                'embedding_models': list(self.embedding_models.keys()),
                'data_sizes': self.data_sizes,
                'k_values': self.k_values,
                'random_seed': RANDOM_SEED,
                'reproducibility_hash': hashlib.md5(str(results_df.values.tobytes()).encode()).hexdigest()
            }
        }
        
        # Performance Analysis
        self.logger.info("📈 Analyzing performance metrics...")
        performance_metrics = ['qps', 'search_per_query', 'insert_time', 'insert_throughput']
        
        for metric in performance_metrics:
            qdrant_col = f'qdrant_{metric}'
            chroma_col = f'chroma_{metric}'
            
            if qdrant_col in results_df.columns and chroma_col in results_df.columns:
                # Remove null values for analysis
                valid_results = results_df.dropna(subset=[qdrant_col, chroma_col])
                
                if not valid_results.empty:
                    qdrant_values = valid_results[qdrant_col]
                    chroma_values = valid_results[chroma_col]
                    
                    # Statistical comparison
                    from scipy import stats
                    statistic, p_value = stats.ttest_rel(qdrant_values, chroma_values)
                    
                    analysis[f'{metric}_comparison'] = {
                        'qdrant_mean': float(qdrant_values.mean()),
                        'qdrant_std': float(qdrant_values.std()),
                        'chroma_mean': float(chroma_values.mean()),
                        'chroma_std': float(chroma_values.std()),
                        't_statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
        
        # Relevance Quality Analysis
        self.logger.info("🎯 Analyzing relevance quality metrics...")
        quality_metrics = ['precision', 'recall', 'f1', 'ndcg', 'hit_rate', 'map']
        
        for k in self.k_values:
            for metric in quality_metrics:
                qdrant_col = f'qdrant_{metric}_at_{k}'
                chroma_col = f'chroma_{metric}_at_{k}'
                
                if qdrant_col in results_df.columns and chroma_col in results_df.columns:
                    valid_results = results_df.dropna(subset=[qdrant_col, chroma_col])
                    
                    if not valid_results.empty:
                        qdrant_values = valid_results[qdrant_col]
                        chroma_values = valid_results[chroma_col]
                        
                        # Statistical comparison
                        from scipy import stats
                        statistic, p_value = stats.ttest_rel(qdrant_values, chroma_values)
                        
                        analysis[f'{metric}_at_{k}_comparison'] = {
                            'qdrant_mean': float(qdrant_values.mean()),
                            'qdrant_std': float(qdrant_values.std()),
                            'chroma_mean': float(chroma_values.mean()),
                            'chroma_std': float(chroma_values.std()),
                            't_statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size': float((qdrant_values.mean() - chroma_values.mean()) / 
                                               np.sqrt((qdrant_values.var() + chroma_values.var()) / 2))
                        }
        
        # Model Comparison Analysis
        self.logger.info("🔍 Analyzing embedding model performance...")
        model_analysis = {}
        
        for model in self.embedding_models.keys():
            model_results = results_df[results_df['model_name'] == model]
            if not model_results.empty:
                model_analysis[model] = {
                    'avg_qdrant_ndcg_10': float(model_results['qdrant_ndcg_at_10'].mean()),
                    'avg_chroma_ndcg_10': float(model_results['chroma_ndcg_at_10'].mean()),
                    'avg_qdrant_qps': float(model_results['qdrant_qps'].mean()),
                    'avg_chroma_qps': float(model_results['chroma_qps'].mean()),
                    'dimension': int(model_results['model_dimension'].iloc[0])
                }
        
        analysis['model_comparison'] = model_analysis
        
        # Scaling Analysis
        self.logger.info("📊 Analyzing scaling behavior...")
        scaling_analysis = {}
        
        for data_size in self.data_sizes:
            size_results = results_df[results_df['data_size'] == data_size]
            if not size_results.empty:
                scaling_analysis[str(data_size)] = {
                    'avg_qdrant_qps': float(size_results['qdrant_qps'].mean()),
                    'avg_chroma_qps': float(size_results['chroma_qps'].mean()),
                    'avg_qdrant_ndcg_10': float(size_results['qdrant_ndcg_at_10'].mean()),
                    'avg_chroma_ndcg_10': float(size_results['chroma_ndcg_at_10'].mean())
                }
        
        analysis['scaling_analysis'] = scaling_analysis
        
        # Overall Recommendations
        self.logger.info("💡 Generating recommendations...")
        
        # Quality-based recommendation
        avg_qdrant_ndcg = results_df['qdrant_ndcg_at_10'].mean()
        avg_chroma_ndcg = results_df['chroma_ndcg_at_10'].mean()
        
        # Performance-based recommendation
        avg_qdrant_qps = results_df['qdrant_qps'].mean()
        avg_chroma_qps = results_df['chroma_qps'].mean()
        
        recommendations = {
            'quality_leader': 'qdrant' if avg_qdrant_ndcg > avg_chroma_ndcg else 'chromadb',
            'performance_leader': 'qdrant' if avg_qdrant_qps > avg_chroma_qps else 'chromadb',
            'quality_advantage': abs(avg_qdrant_ndcg - avg_chroma_ndcg) / max(avg_qdrant_ndcg, avg_chroma_ndcg),
            'performance_advantage': abs(avg_qdrant_qps - avg_chroma_qps) / max(avg_qdrant_qps, avg_chroma_qps),
            'overall_recommendation': 'qdrant' if avg_qdrant_ndcg > avg_chroma_ndcg else 'chromadb'
        }
        
        analysis['recommendations'] = recommendations
        
        self.logger.info("✅ Academic analysis completed")
        return analysis
    
    def save_results(self, results_df: pd.DataFrame, analysis: Dict):
        """Save all benchmark results and analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_filename = f'academic_benchmark_results_{timestamp}.csv'
        results_df.to_csv(results_filename, index=False)
        self.logger.info(f"💾 Raw results saved to: {results_filename}")
        
        # Save analysis
        analysis_filename = f'academic_benchmark_analysis_{timestamp}.json'
        with open(analysis_filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        self.logger.info(f"💾 Analysis saved to: {analysis_filename}")
        
        # Save benchmark metadata
        metadata = {
            'benchmark_version': '1.0',
            'completion_date': datetime.now().isoformat(),
            'embedding_models': {name: {
                'dimension': config['dimension'],
                'description': config['description']
            } for name, config in self.embedding_models.items()},
            'data_sizes': self.data_sizes,
            'k_values': self.k_values,
            'query_count': self.query_count,
            'random_seed': RANDOM_SEED,
            'total_configurations': len(results_df),
            'python_version': '3.x',
            'libraries': {
                'qdrant_client': 'latest',
                'chromadb': 'latest',
                'sentence_transformers': 'latest'
            }
        }
        
        metadata_filename = f'academic_benchmark_metadata_{timestamp}.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"💾 Metadata saved to: {metadata_filename}")
        
        return results_filename, analysis_filename, metadata_filename

def main():
    """Main execution function for academic benchmark."""
    print("🎓 ACADEMIC VECTOR DATABASE BENCHMARK")
    print("=" * 80)
    print("Comparing Qdrant vs ChromaDB for RAG systems")
    print("Models: BGE-M3, all-MiniLM-L6-v2")
    print("Data sizes: 500, 1K, 2.5K, 5K, 10K documents")
    print("Metrics: Precision, Recall, F1, NDCG, Hit Rate, MAP @ k=[1,3,5,10,20]")
    print("=" * 80)
    
    # Initialize benchmark
    benchmark = AcademicVectorDBBenchmark()
    
    try:
        # Run comprehensive benchmark
        results_df = benchmark.run_comprehensive_benchmark()
        
        if not results_df.empty:
            # Create academic analysis
            analysis = benchmark.create_academic_analysis(results_df)
            
            # Save results
            benchmark.save_results(results_df, analysis)
            
            print("\n" + "=" * 80)
            print("✅ ACADEMIC BENCHMARK COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"📊 Total configurations tested: {len(results_df)}")
            print(f"🎯 Embedding models: {', '.join(benchmark.embedding_models.keys())}")
            print(f"📈 Data sizes: {', '.join(map(str, benchmark.data_sizes))}")
            print(f"🔍 Metrics computed: Precision, Recall, F1, NDCG, Hit Rate, MAP")
            print(f"⚡ Performance metrics: QPS, latency, throughput")
            print("\nResults are saved with academic reproducibility standards.")
            print("All random seeds are fixed for reproducible experiments.")
            
        else:
            print("❌ No benchmark results generated")
            
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
