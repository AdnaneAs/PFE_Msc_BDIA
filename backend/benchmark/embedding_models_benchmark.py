"""
Comprehensive Embedding Models Benchmark
========================================

A thorough benchmark comparing 10 different embedding models on intrinsic quality metrics
including semantic similarity, clustering quality, classification transfer learning,
encoding efficiency, and robustness.

Date: May 30, 2025
Version: 1.0 - Multi-Model Embedding Quality Assessment
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Embedding models
from sentence_transformers import SentenceTransformer
import torch

# Metrics and evaluation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform

# Datasets
from datasets import load_dataset
import nltk
try:
    nltk.data.find('tokenizer/punkt')
except LookupError:
    nltk.download('punkt')

class EmbeddingModelsBenchmark:
    """Comprehensive benchmark for embedding models quality assessment."""
    
    def __init__(self):
        """Initialize the benchmark with models and datasets."""
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.setup_logging()
        
        # Define embedding models to test
        self.models_config = {
            'bge-m3': 'BAAI/bge-m3',
            'all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
            'all-mpnet-base-v2': 'all-mpnet-base-v2',
            'all-distilroberta-v1': 'all-distilroberta-v1',
            'paraphrase-MiniLM-L6-v2': 'paraphrase-MiniLM-L6-v2',
            'paraphrase-albert-small-v2': 'paraphrase-albert-small-v2',
            'multi-qa-MiniLM-L6-cos-v1': 'multi-qa-MiniLM-L6-cos-v1',
            'multi-qa-distilbert-cos-v1': 'multi-qa-distilbert-cos-v1',
            'msmarco-distilbert-base-v4': 'msmarco-distilbert-base-v4',
            'e5-small-v2': 'intfloat/e5-small-v2'
        }
        
        self.models = {}
        self.results = {}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = f'embedding_benchmark_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load all embedding models."""
        self.logger.info("=" * 80)
        self.logger.info("EMBEDDING MODELS BENCHMARK - QUALITY ASSESSMENT")
        self.logger.info("=" * 80)
        self.logger.info("[MODELS] Loading embedding models...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {device}")
        
        for model_name, model_path in self.models_config.items():
            try:
                self.logger.info(f"  Loading {model_name} ({model_path})...")
                start_time = time.time()
                
                model = SentenceTransformer(model_path, device=device)
                load_time = time.time() - start_time
                
                # Get model info
                embedding_dim = model.get_sentence_embedding_dimension()
                
                self.models[model_name] = {
                    'model': model,
                    'dimension': embedding_dim,
                    'load_time': load_time,
                    'path': model_path
                }
                
                self.logger.info(f"    [OK] {model_name} loaded (dim: {embedding_dim}, time: {load_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"    [FAILED] {model_name}: {e}")
                
        self.logger.info(f"[OK] Loaded {len(self.models)} models successfully")
        
    def prepare_datasets(self):
        """Prepare evaluation datasets."""
        self.logger.info("\n[DATASETS] Preparing evaluation datasets...")
        
        # 1. Semantic Textual Similarity (STS)
        try:
            self.logger.info("  Loading STS Benchmark...")
            sts_data = load_dataset("mteb/stsbenchmark-sts")['test']
            self.sts_pairs = []
            self.sts_scores = []
            
            for item in sts_data:
                self.sts_pairs.append((item['sentence1'], item['sentence2']))
                self.sts_scores.append(item['score'])
                
            self.logger.info(f"    [OK] STS Benchmark: {len(self.sts_pairs)} sentence pairs")
            
        except Exception as e:
            self.logger.warning(f"    [SKIP] STS Benchmark failed: {e}")
            # Fallback to simple similarity pairs
            self.sts_pairs = [
                ("The cat sat on the mat", "A cat was sitting on a mat"),
                ("I love programming", "Programming is my passion"),
                ("The weather is nice today", "Today has beautiful weather"),
                ("Machine learning is complex", "AI and ML are complicated subjects"),
                ("I hate this movie", "This film is terrible")
            ]
            self.sts_scores = [4.5, 4.0, 4.2, 3.8, 4.1]
            self.logger.info(f"    [OK] Fallback STS: {len(self.sts_pairs)} pairs")
            
        # 2. Classification dataset (20 Newsgroups subset)
        try:
            self.logger.info("  Loading classification dataset...")
            from sklearn.datasets import fetch_20newsgroups
            
            categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
            newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
            
            # Sample for faster processing
            sample_size = min(1000, len(newsgroups_train.data))
            indices = np.random.choice(len(newsgroups_train.data), sample_size, replace=False)
            
            self.classification_texts = [newsgroups_train.data[i] for i in indices]
            self.classification_labels = [newsgroups_train.target[i] for i in indices]
            
            self.logger.info(f"    [OK] Classification dataset: {len(self.classification_texts)} samples, {len(set(self.classification_labels))} classes")
            
        except Exception as e:
            self.logger.warning(f"    [SKIP] Classification dataset failed: {e}")
            # Fallback dataset
            self.classification_texts = [
                "This is about technology and computers",
                "Medical research shows interesting results", 
                "Religious beliefs vary across cultures",
                "Computer graphics are advancing rapidly",
                "Healthcare improvements benefit everyone",
                "Spiritual practices bring peace",
                "Software development requires skill",
                "Clinical trials test new treatments"
            ]
            self.classification_labels = [0, 1, 2, 0, 1, 2, 0, 1]
            
        # 3. Clustering dataset
        self.clustering_texts = self.classification_texts[:min(500, len(self.classification_texts))]
        self.clustering_labels = self.classification_labels[:len(self.clustering_texts)]
        
        self.logger.info(f"  [OK] Datasets prepared successfully")
        
    def benchmark_semantic_similarity(self, model_name: str, model_info: Dict) -> Dict:
        """Benchmark semantic textual similarity."""
        model = model_info['model']
        results = {}
        
        try:
            # Encode sentence pairs
            sentences1 = [pair[0] for pair in self.sts_pairs]
            sentences2 = [pair[1] for pair in self.sts_pairs]
            
            embeddings1 = model.encode(sentences1, show_progress_bar=False)
            embeddings2 = model.encode(sentences2, show_progress_bar=False)
            
            # Calculate cosine similarities
            similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                sim = cosine_similarity([emb1], [emb2])[0][0]
                similarities.append(sim)
            
            # Correlation with human scores
            spearman_corr, _ = spearmanr(similarities, self.sts_scores)
            pearson_corr, _ = pearsonr(similarities, self.sts_scores)
            
            results['spearman_correlation'] = spearman_corr
            results['pearson_correlation'] = pearson_corr
            results['avg_similarity'] = np.mean(similarities)
            results['similarity_std'] = np.std(similarities)
            
        except Exception as e:
            self.logger.warning(f"    STS failed for {model_name}: {e}")
            results = {'spearman_correlation': 0, 'pearson_correlation': 0, 'avg_similarity': 0, 'similarity_std': 0}
            
        return results
        
    def benchmark_clustering_quality(self, model_name: str, model_info: Dict) -> Dict:
        """Benchmark clustering quality of embeddings."""
        model = model_info['model']
        results = {}
        
        try:
            # Encode texts
            embeddings = model.encode(self.clustering_texts, show_progress_bar=False)
            
            # Perform clustering
            n_clusters = len(set(self.clustering_labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate clustering metrics
            silhouette = silhouette_score(embeddings, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
            
            results['silhouette_score'] = silhouette
            results['calinski_harabasz_score'] = calinski_harabasz
            results['davies_bouldin_score'] = davies_bouldin
            
            # Clustering accuracy (best match between predicted and true labels)
            from sklearn.metrics import adjusted_rand_score
            clustering_accuracy = adjusted_rand_score(self.clustering_labels, cluster_labels)
            results['clustering_accuracy'] = clustering_accuracy
            
        except Exception as e:
            self.logger.warning(f"    Clustering failed for {model_name}: {e}")
            results = {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'davies_bouldin_score': 1, 'clustering_accuracy': 0}
            
        return results
        
    def benchmark_classification_transfer(self, model_name: str, model_info: Dict) -> Dict:
        """Benchmark transfer learning for classification."""
        model = model_info['model']
        results = {}
        
        try:
            # Encode texts
            embeddings = model.encode(self.classification_texts, show_progress_bar=False)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, self.classification_labels, test_size=0.3, random_state=42, stratify=self.classification_labels
            )
            
            # Train classifier
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results['classification_accuracy'] = accuracy
            results['classification_f1'] = f1
            
        except Exception as e:
            self.logger.warning(f"    Classification failed for {model_name}: {e}")
            results = {'classification_accuracy': 0, 'classification_f1': 0}
            
        return results
        
    def benchmark_efficiency(self, model_name: str, model_info: Dict) -> Dict:
        """Benchmark encoding efficiency."""
        model = model_info['model']
        results = {}
        
        try:
            # Test sentences of different lengths
            test_sentences = [
                "Short text.",
                "This is a medium length sentence with some more words to test encoding speed.",
                "This is a much longer sentence that contains significantly more words and should take more time to encode, allowing us to test the scalability and efficiency of the embedding model when processing longer text sequences."
            ]
            
            # Measure encoding time
            times = []
            for _ in range(5):  # Multiple runs for average
                start_time = time.time()
                _ = model.encode(test_sentences * 10, show_progress_bar=False)  # 30 sentences total
                encoding_time = time.time() - start_time
                times.append(encoding_time)
            
            avg_encoding_time = np.mean(times)
            sentences_per_second = (len(test_sentences) * 10) / avg_encoding_time
            
            results['avg_encoding_time'] = avg_encoding_time
            results['sentences_per_second'] = sentences_per_second
            results['dimension'] = model_info['dimension']
            results['load_time'] = model_info['load_time']
            
        except Exception as e:
            self.logger.warning(f"    Efficiency test failed for {model_name}: {e}")
            results = {'avg_encoding_time': 0, 'sentences_per_second': 0, 'dimension': 0, 'load_time': 0}
            
        return results
        
    def benchmark_dimensionality_analysis(self, model_name: str, model_info: Dict) -> Dict:
        """Analyze embedding dimensionality characteristics."""
        model = model_info['model']
        results = {}
        
        try:
            # Encode a sample of texts
            sample_texts = self.classification_texts[:min(200, len(self.classification_texts))]
            embeddings = model.encode(sample_texts, show_progress_bar=False)
            
            # PCA analysis
            pca = PCA()
            pca.fit(embeddings)
            
            # Explained variance ratios
            explained_variance_95 = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(explained_variance_95 >= 0.95) + 1
            
            # Intrinsic dimensionality estimation (simplified)
            intrinsic_dim = n_components_95
            
            # Isotropy measure (how uniformly distributed embeddings are)
            mean_embedding = np.mean(embeddings, axis=0)
            centered_embeddings = embeddings - mean_embedding
            cov_matrix = np.cov(centered_embeddings.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove negative eigenvalues due to numerical errors
            isotropy = np.min(eigenvalues) / np.max(eigenvalues) if len(eigenvalues) > 0 else 0
            
            results['intrinsic_dimensionality'] = intrinsic_dim
            results['explained_variance_95'] = explained_variance_95[n_components_95-1] if n_components_95 > 0 else 0
            results['isotropy_score'] = isotropy
            results['full_dimensionality'] = model_info['dimension']
            
        except Exception as e:
            self.logger.warning(f"    Dimensionality analysis failed for {model_name}: {e}")
            results = {'intrinsic_dimensionality': 0, 'explained_variance_95': 0, 'isotropy_score': 0, 'full_dimensionality': 0}
            
        return results
        
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite."""
        self.logger.info("\n[BENCHMARK] Running comprehensive embedding models benchmark...")
        
        for model_name, model_info in self.models.items():
            self.logger.info(f"\n  [MODEL] Benchmarking {model_name}...")
            
            model_results = {
                'model_name': model_name,
                'model_path': model_info['path'],
                'dimension': model_info['dimension']
            }
            
            # 1. Semantic similarity
            self.logger.info(f"    Testing semantic similarity...")
            sts_results = self.benchmark_semantic_similarity(model_name, model_info)
            model_results.update({f'sts_{k}': v for k, v in sts_results.items()})
            
            # 2. Clustering quality
            self.logger.info(f"    Testing clustering quality...")
            clustering_results = self.benchmark_clustering_quality(model_name, model_info)
            model_results.update({f'clustering_{k}': v for k, v in clustering_results.items()})
            
            # 3. Classification transfer
            self.logger.info(f"    Testing classification transfer...")
            classification_results = self.benchmark_classification_transfer(model_name, model_info)
            model_results.update({f'transfer_{k}': v for k, v in classification_results.items()})
            
            # 4. Efficiency
            self.logger.info(f"    Testing efficiency...")
            efficiency_results = self.benchmark_efficiency(model_name, model_info)
            model_results.update({f'efficiency_{k}': v for k, v in efficiency_results.items()})
            
            # 5. Dimensionality analysis
            self.logger.info(f"    Testing dimensionality...")
            dim_results = self.benchmark_dimensionality_analysis(model_name, model_info)
            model_results.update({f'dim_{k}': v for k, v in dim_results.items()})
            
            self.results[model_name] = model_results
            
            self.logger.info(f"    [OK] {model_name} benchmark completed")
            
    def generate_visualizations(self):
        """Generate comprehensive visualization plots."""
        self.logger.info("\n[VISUALIZATION] Generating benchmark visualizations...")
        
        # Convert results to DataFrame
        df = pd.DataFrame(list(self.results.values()))
        df = df.round(4)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        plt.suptitle('Embedding Models Comprehensive Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Semantic Similarity Performance
        ax1 = plt.subplot(3, 3, 1)
        models = df['model_name']
        spearman_scores = df['sts_spearman_correlation']
        bars = ax1.bar(models, spearman_scores, color='skyblue', alpha=0.8)
        ax1.set_title('Semantic Similarity (Spearman Correlation)', fontweight='bold')
        ax1.set_ylabel('Correlation Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, spearman_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Clustering Quality
        ax2 = plt.subplot(3, 3, 2)
        silhouette_scores = df['clustering_silhouette_score']
        bars = ax2.bar(models, silhouette_scores, color='lightgreen', alpha=0.8)
        ax2.set_title('Clustering Quality (Silhouette Score)', fontweight='bold')
        ax2.set_ylabel('Silhouette Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, silhouette_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Classification Transfer Learning
        ax3 = plt.subplot(3, 3, 3)
        classification_scores = df['transfer_classification_accuracy']
        bars = ax3.bar(models, classification_scores, color='coral', alpha=0.8)
        ax3.set_title('Classification Transfer Learning', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, classification_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Encoding Efficiency
        ax4 = plt.subplot(3, 3, 4)
        efficiency_scores = df['efficiency_sentences_per_second']
        bars = ax4.bar(models, efficiency_scores, color='gold', alpha=0.8)
        ax4.set_title('Encoding Efficiency', fontweight='bold')
        ax4.set_ylabel('Sentences/Second')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, efficiency_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Model Dimensions
        ax5 = plt.subplot(3, 3, 5)
        dimensions = df['dimension']
        bars = ax5.bar(models, dimensions, color='plum', alpha=0.8)
        ax5.set_title('Embedding Dimensions', fontweight='bold')
        ax5.set_ylabel('Dimensions')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        for bar, dim in zip(bars, dimensions):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(dim)}', ha='center', va='bottom', fontsize=8)
        
        # 6. Isotropy Score
        ax6 = plt.subplot(3, 3, 6)
        isotropy_scores = df['dim_isotropy_score']
        bars = ax6.bar(models, isotropy_scores, color='lightcyan', alpha=0.8)
        ax6.set_title('Embedding Isotropy', fontweight='bold')
        ax6.set_ylabel('Isotropy Score')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, isotropy_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 7. Overall Performance Radar Chart
        ax7 = plt.subplot(3, 3, 7, projection='polar')
        
        # Normalize metrics for radar chart
        metrics = ['sts_spearman_correlation', 'clustering_silhouette_score', 
                  'transfer_classification_accuracy', 'efficiency_sentences_per_second']
        
        normalized_df = df.copy()
        for metric in metrics:
            max_val = df[metric].max()
            min_val = df[metric].min()
            if max_val != min_val:
                normalized_df[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[metric] = 1.0
        
        # Plot top 3 models
        top_models = df.nlargest(3, 'sts_spearman_correlation')['model_name'].tolist()
        colors = ['red', 'blue', 'green']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(top_models[:3]):
            model_data = normalized_df[normalized_df['model_name'] == model]
            values = [model_data[metric].iloc[0] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax7.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax7.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(['STS Correlation', 'Clustering', 'Classification', 'Efficiency'])
        ax7.set_title('Top 3 Models - Overall Performance', fontweight='bold')
        ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 8. Performance vs Efficiency Scatter
        ax8 = plt.subplot(3, 3, 8)
        performance_score = (df['sts_spearman_correlation'] + df['clustering_silhouette_score'] + 
                           df['transfer_classification_accuracy']) / 3
        efficiency_score = df['efficiency_sentences_per_second']
        
        scatter = ax8.scatter(efficiency_score, performance_score, 
                             c=df['dimension'], cmap='viridis', s=100, alpha=0.7)
        ax8.set_xlabel('Encoding Speed (sentences/sec)')
        ax8.set_ylabel('Average Performance Score')
        ax8.set_title('Performance vs Efficiency', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # Add model names as annotations
        for i, model in enumerate(models):
            ax8.annotate(model, (efficiency_score.iloc[i], performance_score.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=ax8, label='Dimensions')
        
        # 9. Summary Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary table
        summary_metrics = ['sts_spearman_correlation', 'clustering_silhouette_score', 
                          'transfer_classification_accuracy', 'efficiency_sentences_per_second']
        summary_data = df[['model_name'] + summary_metrics].round(3)
        summary_data.columns = ['Model', 'STS Corr', 'Clustering', 'Classification', 'Speed']
        
        table = ax9.table(cellText=summary_data.values,
                         colLabels=summary_data.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        ax9.set_title('Summary Table', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'embedding_models_benchmark_{self.timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"  [OK] Visualization saved: {plot_filename}")
        
        plt.show()
        
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        self.logger.info("\n[REPORT] Generating benchmark report...")
        
        # Convert results to DataFrame
        df = pd.DataFrame(list(self.results.values()))
        
        # Save detailed results to CSV
        csv_filename = f'embedding_models_benchmark_results_{self.timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        
        # Generate markdown report
        report_filename = f'EMBEDDING_MODELS_BENCHMARK_REPORT_{self.timestamp}.md'
        
        with open(report_filename, 'w') as f:
            f.write(f"# Embedding Models Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Models Tested:** {len(self.models)}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Best performing models
            best_sts = df.loc[df['sts_spearman_correlation'].idxmax(), 'model_name']
            best_clustering = df.loc[df['clustering_silhouette_score'].idxmax(), 'model_name']
            best_classification = df.loc[df['transfer_classification_accuracy'].idxmax(), 'model_name']
            best_efficiency = df.loc[df['efficiency_sentences_per_second'].idxmax(), 'model_name']
            
            f.write(f"- **Best Semantic Similarity:** {best_sts} ({df['sts_spearman_correlation'].max():.3f})\n")
            f.write(f"- **Best Clustering Quality:** {best_clustering} ({df['clustering_silhouette_score'].max():.3f})\n")
            f.write(f"- **Best Classification Transfer:** {best_classification} ({df['transfer_classification_accuracy'].max():.3f})\n")
            f.write(f"- **Most Efficient:** {best_efficiency} ({df['efficiency_sentences_per_second'].max():.1f} sent/sec)\n\n")
            
            f.write("## Detailed Results\n\n")
            
            # Sort by overall performance
            df['overall_score'] = (df['sts_spearman_correlation'] + df['clustering_silhouette_score'] + 
                                 df['transfer_classification_accuracy']) / 3
            df_sorted = df.sort_values('overall_score', ascending=False)
            
            f.write("### Models Ranked by Overall Performance\n\n")
            for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                f.write(f"{i}. **{row['model_name']}** (Overall: {row['overall_score']:.3f})\n")
                f.write(f"   - Semantic Similarity: {row['sts_spearman_correlation']:.3f}\n")
                f.write(f"   - Clustering Quality: {row['clustering_silhouette_score']:.3f}\n")
                f.write(f"   - Classification Accuracy: {row['transfer_classification_accuracy']:.3f}\n")
                f.write(f"   - Encoding Speed: {row['efficiency_sentences_per_second']:.1f} sent/sec\n")
                f.write(f"   - Dimensions: {int(row['dimension'])}\n\n")
            
            f.write("## Key Insights\n\n")
            
            # Analysis
            high_dim_models = df[df['dimension'] > 500]['model_name'].tolist()
            low_dim_models = df[df['dimension'] <= 500]['model_name'].tolist()
            
            f.write(f"### Dimensionality Analysis\n")
            f.write(f"- **High-dimensional models (>500D):** {', '.join(high_dim_models)}\n")
            f.write(f"- **Low-dimensional models (≤500D):** {', '.join(low_dim_models)}\n\n")
            
            # Efficiency vs Performance
            efficient_models = df.nlargest(3, 'efficiency_sentences_per_second')['model_name'].tolist()
            accurate_models = df.nlargest(3, 'overall_score')['model_name'].tolist()
            
            f.write(f"### Efficiency vs Performance\n")
            f.write(f"- **Most Efficient:** {', '.join(efficient_models)}\n")
            f.write(f"- **Most Accurate:** {', '.join(accurate_models)}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the benchmark results:\n\n")
            
            # Overall best
            best_overall = df_sorted.iloc[0]['model_name']
            f.write(f"- **Best Overall Model:** {best_overall} - Excellent balance of accuracy and performance\n")
            
            # Speed recommendation
            f.write(f"- **For Speed-Critical Applications:** {best_efficiency} - Fastest encoding speed\n")
            
            # Quality recommendation
            f.write(f"- **For Maximum Accuracy:** {best_sts} - Best semantic understanding\n\n")
            
            f.write("## Methodology\n\n")
            f.write("The benchmark evaluated models across five key dimensions:\n\n")
            f.write("1. **Semantic Textual Similarity:** Correlation with human similarity judgments\n")
            f.write("2. **Clustering Quality:** How well embeddings cluster semantically similar texts\n")
            f.write("3. **Classification Transfer:** Performance on downstream classification tasks\n")
            f.write("4. **Encoding Efficiency:** Speed of generating embeddings\n")
            f.write("5. **Dimensionality Analysis:** Intrinsic dimensionality and isotropy\n\n")
            
            f.write("All tests were conducted on the same hardware and datasets for fair comparison.\n")
        
        self.logger.info(f"  [OK] Report saved: {report_filename}")
        self.logger.info(f"  [OK] Results saved: {csv_filename}")
        
        return report_filename, csv_filename

def main():
    """Main execution function."""
    benchmark = EmbeddingModelsBenchmark()
    
    try:
        # Load models
        benchmark.load_models()
        
        if not benchmark.models:
            benchmark.logger.error("No models loaded successfully. Exiting.")
            return
        
        # Prepare datasets
        benchmark.prepare_datasets()
        
        # Run benchmarks
        benchmark.run_comprehensive_benchmark()
        
        # Generate visualizations
        benchmark.generate_visualizations()
        
        # Generate report
        report_file, csv_file = benchmark.generate_report()
        
        benchmark.logger.info("\n" + "=" * 80)
        benchmark.logger.info("BENCHMARK COMPLETED SUCCESSFULLY!")
        benchmark.logger.info("=" * 80)
        benchmark.logger.info(f"Report: {report_file}")
        benchmark.logger.info(f"Results: {csv_file}")
        benchmark.logger.info(f"Visualization: embedding_models_benchmark_{benchmark.timestamp}.png")
        
    except Exception as e:
        benchmark.logger.error(f"Benchmark failed: {e}")
        import traceback
        benchmark.logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
