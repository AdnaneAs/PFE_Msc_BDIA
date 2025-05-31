"""
Academic Visualization Generator for BGE Reranker Benchmark Results
================================================================

Creates comprehensive academic-quality visualizations for the BGE reranker benchmark,
including bar charts, line charts, and improvement percentage visualizations.

Features:
- Side-by-side metric comparisons (with/without reranking)
- Improvement percentage charts with color coding
- Line charts showing metric trends
- Professional academic styling
- Publication-ready figures

Date: May 30, 2025
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AcademicVisualizationGenerator:
    """Generates academic-quality visualizations for BGE reranker benchmark results."""
    
    def __init__(self, results_file: str):
        """Initialize with benchmark results file."""
        self.results_file = results_file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            self.summary_data = json.load(f)
            
        # Extract metrics for visualization
        self.metrics_data = self._extract_metrics_data()
        
    def _extract_metrics_data(self) -> dict:
        """Extract and organize metrics data for visualization."""
        metrics = {}
        
        # Define metric categories
        ranking_metrics = ['precision@1', 'precision@3', 'precision@5', 'precision@10',
                          'recall@1', 'recall@3', 'recall@5', 'recall@10',
                          'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10',
                          'map', 'mrr']
        
        efficiency_metrics = ['hit_rate@1', 'hit_rate@3', 'hit_rate@5', 'hit_rate@10']
        
        time_metrics = ['retrieval_time', 'reranking_time', 'total_time']
        
        # Extract data for each metric
        for metric in ranking_metrics + efficiency_metrics + time_metrics:
            if metric in self.summary_data:
                metrics[metric] = {
                    'without_reranking': self.summary_data[metric]['without_reranking']['mean'],
                    'with_reranking': self.summary_data[metric]['with_reranking']['mean'],
                    'improvement_percent': self.summary_data[metric]['improvement_percent']
                }
        
        return metrics
    
    def create_comprehensive_visualization(self):
        """Create all academic visualizations."""
        print("🎨 Creating comprehensive academic visualizations...")
        
        # 1. Main metrics comparison (side-by-side bars)
        self.create_main_metrics_comparison()
        
        # 2. Improvement percentage chart
        self.create_improvement_percentage_chart()
        
        # 3. Precision metrics line chart
        self.create_precision_line_chart()
        
        # 4. Recall metrics line chart  
        self.create_recall_line_chart()
        
        # 5. NDCG metrics line chart
        self.create_ndcg_line_chart()
        
        # 6. Hit rate comparison
        self.create_hit_rate_comparison()
        
        # 7. Time performance analysis
        self.create_time_analysis()
        
        # 8. Key metrics spotlight (publication figure)
        self.create_key_metrics_spotlight()
        
        # 9. Comprehensive dashboard
        self.create_comprehensive_dashboard()
        
        print(f"✅ All academic visualizations created with timestamp: {self.timestamp}")
    
    def create_main_metrics_comparison(self):
        """Create side-by-side bar comparison of main metrics."""
        # Select key metrics for main comparison
        key_metrics = ['precision@5', 'recall@5', 'ndcg@5', 'map', 'mrr', 'hit_rate@5']
        
        # Prepare data
        metrics_names = [m.upper().replace('@', '@') for m in key_metrics]
        without_values = [self.metrics_data[m]['without_reranking'] for m in key_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in key_metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, without_values, width, label='Without BGE Reranking', 
                      color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, with_values, width, label='With BGE Reranking',
                      color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize chart
        ax.set_xlabel('Evaluation Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('BGE Reranker Performance Comparison: Key Metrics\n' +
                    'HotpotQA Dataset (100 samples) with all-MiniLM-L6-v2 Embeddings', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        output_file = f'academic_main_metrics_comparison_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {output_file}")
    
    def create_improvement_percentage_chart(self):
        """Create improvement percentage chart with color coding."""
        # Get all metrics with improvements
        metrics = list(self.metrics_data.keys())
        improvements = [self.metrics_data[m]['improvement_percent'] for m in metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Color code improvements (green for positive, red for negative)
        colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(metrics)), improvements, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        
        # Customize chart
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m.upper().replace('@', '@').replace('_', ' ') for m in metrics])
        ax.set_xlabel('Improvement Percentage (%)', fontsize=14, fontweight='bold')
        ax.set_title('BGE Reranker Performance Improvements by Metric\n' +
                    'Positive values indicate improvement with reranking', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            width = bar.get_width()
            label_x = width + (max(improvements) * 0.01) if width >= 0 else width - (max(improvements) * 0.01)
            ha = 'left' if width >= 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{imp:+.2f}%', ha=ha, va='center', fontweight='bold')
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        output_file = f'academic_improvement_percentage_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Created: {output_file}")
    
    def create_precision_line_chart(self):
        """Create line chart for Precision@K metrics."""
        k_values = [1, 3, 5, 10]
        precision_metrics = [f'precision@{k}' for k in k_values]
        
        without_values = [self.metrics_data[m]['without_reranking'] for m in precision_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in precision_metrics]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(k_values, without_values, marker='o', linewidth=3, markersize=8, 
               label='Without BGE Reranking', color='#FF6B6B')
        ax.plot(k_values, with_values, marker='s', linewidth=3, markersize=8,
               label='With BGE Reranking', color='#4ECDC4')
        
        ax.set_xlabel('K (Top-K Retrieved Documents)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision@K Score', fontsize=14, fontweight='bold')
        ax.set_title('Precision@K Performance Comparison\nBGE Reranker Impact on Retrieval Precision', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1)
        
        # Add value annotations
        for i, k in enumerate(k_values):
            ax.annotate(f'{without_values[i]:.3f}', 
                       (k, without_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
            ax.annotate(f'{with_values[i]:.3f}', 
                       (k, with_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        output_file = f'academic_precision_line_chart_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {output_file}")
    
    def create_recall_line_chart(self):
        """Create line chart for Recall@K metrics."""
        k_values = [1, 3, 5, 10]
        recall_metrics = [f'recall@{k}' for k in k_values]
        
        without_values = [self.metrics_data[m]['without_reranking'] for m in recall_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in recall_metrics]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(k_values, without_values, marker='o', linewidth=3, markersize=8, 
               label='Without BGE Reranking', color='#FF6B6B')
        ax.plot(k_values, with_values, marker='s', linewidth=3, markersize=8,
               label='With BGE Reranking', color='#4ECDC4')
        
        ax.set_xlabel('K (Top-K Retrieved Documents)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Recall@K Score', fontsize=14, fontweight='bold')
        ax.set_title('Recall@K Performance Comparison\nBGE Reranker Impact on Retrieval Recall', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1)
        
        # Add value annotations
        for i, k in enumerate(k_values):
            ax.annotate(f'{without_values[i]:.3f}', 
                       (k, without_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
            ax.annotate(f'{with_values[i]:.3f}', 
                       (k, with_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        output_file = f'academic_recall_line_chart_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {output_file}")
    
    def create_ndcg_line_chart(self):
        """Create line chart for NDCG@K metrics."""
        k_values = [1, 3, 5, 10]
        ndcg_metrics = [f'ndcg@{k}' for k in k_values]
        
        without_values = [self.metrics_data[m]['without_reranking'] for m in ndcg_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in ndcg_metrics]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(k_values, without_values, marker='o', linewidth=3, markersize=8, 
               label='Without BGE Reranking', color='#FF6B6B')
        ax.plot(k_values, with_values, marker='s', linewidth=3, markersize=8,
               label='With BGE Reranking', color='#4ECDC4')
        
        ax.set_xlabel('K (Top-K Retrieved Documents)', fontsize=14, fontweight='bold')
        ax.set_ylabel('NDCG@K Score', fontsize=14, fontweight='bold')
        ax.set_title('NDCG@K Performance Comparison\nBGE Reranker Impact on Ranking Quality', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1)
        
        # Add value annotations
        for i, k in enumerate(k_values):
            ax.annotate(f'{without_values[i]:.3f}', 
                       (k, without_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
            ax.annotate(f'{with_values[i]:.3f}', 
                       (k, with_values[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        output_file = f'academic_ndcg_line_chart_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {output_file}")
    
    def create_hit_rate_comparison(self):
        """Create hit rate comparison chart."""
        k_values = [1, 3, 5, 10]
        hit_rate_metrics = [f'hit_rate@{k}' for k in k_values]
        
        without_values = [self.metrics_data[m]['without_reranking'] for m in hit_rate_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in hit_rate_metrics]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, without_values, width, label='Without BGE Reranking', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, with_values, width, label='With BGE Reranking',
                      color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('K (Top-K Retrieved Documents)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Hit Rate@K', fontsize=14, fontweight='bold')
        ax.set_title('Hit Rate@K Performance Comparison\nBGE Reranker Success Rate Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        output_file = f'academic_hit_rate_comparison_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {output_file}")
    
    def create_time_analysis(self):
        """Create time performance analysis chart."""
        time_metrics = ['retrieval_time', 'reranking_time', 'total_time']
        without_values = [self.metrics_data[m]['without_reranking'] for m in time_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in time_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Absolute times
        x = np.arange(len(time_metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, without_values, width, label='Without BGE Reranking', 
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, with_values, width, label='With BGE Reranking',
                       color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('Time Components', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Time Performance Analysis\nAbsolute Times', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Retrieval', 'Reranking', 'Total'])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Time breakdown (stacked bar for with reranking)
        rerank_retrieval = self.metrics_data['retrieval_time']['with_reranking']
        rerank_reranking = self.metrics_data['reranking_time']['with_reranking']
        
        ax2.bar(['Without\nReranking'], [self.metrics_data['total_time']['without_reranking']], 
               color='#FF6B6B', alpha=0.8, label='Retrieval Only')
        ax2.bar(['With\nReranking'], [rerank_retrieval], 
               color='#4ECDC4', alpha=0.8, label='Retrieval Time')
        ax2.bar(['With\nReranking'], [rerank_reranking], bottom=[rerank_retrieval],
               color='#F39C12', alpha=0.8, label='Reranking Time')
        
        ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Time Breakdown Analysis\nComponent Contribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f'academic_time_analysis_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⏱️ Created: {output_file}")
    
    def create_key_metrics_spotlight(self):
        """Create publication-ready spotlight on key improvements."""
        # Focus on most important metrics
        key_metrics = {
            'MAP': {'without': self.metrics_data['map']['without_reranking'],
                   'with': self.metrics_data['map']['with_reranking'],
                   'improvement': self.metrics_data['map']['improvement_percent']},
            'Precision@5': {'without': self.metrics_data['precision@5']['without_reranking'],
                           'with': self.metrics_data['precision@5']['with_reranking'],
                           'improvement': self.metrics_data['precision@5']['improvement_percent']},
            'NDCG@5': {'without': self.metrics_data['ndcg@5']['without_reranking'],
                      'with': self.metrics_data['ndcg@5']['with_reranking'],
                      'improvement': self.metrics_data['ndcg@5']['improvement_percent']},
            'MRR': {'without': self.metrics_data['mrr']['without_reranking'],
                   'with': self.metrics_data['mrr']['with_reranking'],
                   'improvement': self.metrics_data['mrr']['improvement_percent']}
        }
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (metric_name, data) in enumerate(key_metrics.items()):
            ax = axes[i]
            
            # Bar chart
            bars = ax.bar(['Without\nReranking', 'With\nReranking'], 
                         [data['without'], data['with']],
                         color=['#FF6B6B', '#4ECDC4'], alpha=0.8, 
                         edgecolor='black', linewidth=1)
            
            # Add improvement percentage
            ax.text(0.5, max(data['without'], data['with']) * 1.05,
                   f'Improvement: {data["improvement"]:+.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(data['without'], data['with']) * 1.2)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('BGE Reranker: Key Performance Improvements\n' +
                    'Publication-Ready Results for Academic Research', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_file = f'academic_key_metrics_spotlight_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🌟 Created: {output_file}")
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all key visualizations."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a complex grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Top row: Key metrics comparison (spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        key_metrics = ['precision@5', 'recall@5', 'ndcg@5', 'map']
        metrics_names = [m.upper().replace('@', '@') for m in key_metrics]
        without_values = [self.metrics_data[m]['without_reranking'] for m in key_metrics]
        with_values = [self.metrics_data[m]['with_reranking'] for m in key_metrics]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        bars1 = ax1.bar(x - width/2, without_values, width, label='Without BGE', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, with_values, width, label='With BGE', color='#4ECDC4', alpha=0.8)
        ax1.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top right: Improvement percentages (spans 1x2)
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        improvements = [self.metrics_data[m]['improvement_percent'] for m in key_metrics]
        colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
        bars = ax2.bar(metrics_names, improvements, color=colors, alpha=0.8)
        ax2.set_title('Performance Improvements (%)', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Bottom left: Precision@K line chart
        ax3 = fig.add_subplot(gs[2, 0:2])
        k_values = [1, 3, 5, 10]
        precision_without = [self.metrics_data[f'precision@{k}']['without_reranking'] for k in k_values]
        precision_with = [self.metrics_data[f'precision@{k}']['with_reranking'] for k in k_values]
        ax3.plot(k_values, precision_without, 'o-', label='Without BGE', color='#FF6B6B', linewidth=2)
        ax3.plot(k_values, precision_with, 's-', label='With BGE', color='#4ECDC4', linewidth=2)
        ax3.set_title('Precision@K Trends', fontsize=14, fontweight='bold')
        ax3.set_xlabel('K')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bottom middle: NDCG@K line chart
        ax4 = fig.add_subplot(gs[2, 2:4])
        ndcg_without = [self.metrics_data[f'ndcg@{k}']['without_reranking'] for k in k_values]
        ndcg_with = [self.metrics_data[f'ndcg@{k}']['with_reranking'] for k in k_values]
        ax4.plot(k_values, ndcg_without, 'o-', label='Without BGE', color='#FF6B6B', linewidth=2)
        ax4.plot(k_values, ndcg_with, 's-', label='With BGE', color='#4ECDC4', linewidth=2)
        ax4.set_title('NDCG@K Trends', fontsize=14, fontweight='bold')
        ax4.set_xlabel('K')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Bottom row: Time analysis
        ax5 = fig.add_subplot(gs[3, :])
        time_categories = ['Retrieval\nTime', 'Reranking\nTime', 'Total\nTime']
        without_times = [self.metrics_data['retrieval_time']['without_reranking'], 0, 
                        self.metrics_data['total_time']['without_reranking']]
        with_times = [self.metrics_data['retrieval_time']['with_reranking'],
                     self.metrics_data['reranking_time']['with_reranking'],
                     self.metrics_data['total_time']['with_reranking']]
        
        x_time = np.arange(len(time_categories))
        ax5.bar(x_time - 0.2, without_times, 0.4, label='Without BGE', color='#FF6B6B', alpha=0.8)
        ax5.bar(x_time + 0.2, with_times, 0.4, label='With BGE', color='#4ECDC4', alpha=0.8)
        ax5.set_title('Time Performance Analysis', fontsize=14, fontweight='bold')
        ax5.set_xticks(x_time)
        ax5.set_xticklabels(time_categories)
        ax5.set_ylabel('Time (seconds)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle('BGE Reranker Comprehensive Performance Analysis\n' +
                    'HotpotQA Dataset | 100 Samples | all-MiniLM-L6-v2 Embeddings', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        output_file = f'academic_comprehensive_dashboard_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {output_file}")


def main():
    """Main function to generate all academic visualizations."""
    print("🎓 Academic Visualization Generator for BGE Reranker Benchmark")
    print("=" * 70)
      # Find the most recent summary file
    import glob
    # Look in parent directory as well
    summary_files = glob.glob("fixed_bge_reranker_benchmark_summary_*.json")
    if not summary_files:
        summary_files = glob.glob("../fixed_bge_reranker_benchmark_summary_*.json")
    
    if not summary_files:
        print("❌ No summary files found! Please run the benchmark first.")
        return
    
    # Use the most recent file
    latest_file = max(summary_files, key=os.path.getctime)
    print(f"📁 Using results file: {latest_file}")
    
    # Generate visualizations
    generator = AcademicVisualizationGenerator(latest_file)
    generator.create_comprehensive_visualization()
    
    print("\n🎉 Academic visualization generation completed!")
    print("📁 All charts saved with academic-quality formatting for publication use.")


if __name__ == "__main__":
    main()
