import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_comparison_visualizations():
    """Create comprehensive comparison visualizations"""
    print("📊 Creating benchmark comparison visualizations...")
    
    # Load both datasets
    try:
        random_df = pd.read_csv('detailed_benchmark_results.csv')
        bge_df = pd.read_csv('bge_benchmark_results.csv')
    except FileNotFoundError as e:
        print(f"❌ Could not load benchmark files: {e}")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. NDCG@10 Comparison
    ax1 = axes[0, 0]
    
    # Random vectors data (aggregate by data size)
    random_ndcg_data = []
    for size in [1000, 5000, 10000, 50000]:
        size_data = random_df[random_df['data_size'] == size]
        if not size_data.empty:
            random_ndcg_data.append({
                'Data Size': size,
                'Database': 'Qdrant (Random)',
                'NDCG@10': size_data['qdrant_ndcg_at_10'].mean()
            })
            random_ndcg_data.append({
                'Data Size': size,
                'Database': 'ChromaDB (Random)', 
                'NDCG@10': size_data['chroma_ndcg_at_10'].mean()
            })
    
    # BGE data
    bge_ndcg_data = []
    for _, row in bge_df.iterrows():
        bge_ndcg_data.append({
            'Data Size': row['data_size'],
            'Database': 'Qdrant (BGE-M3)',
            'NDCG@10': row['qdrant_ndcg_at_10']
        })
        bge_ndcg_data.append({
            'Data Size': row['data_size'],
            'Database': 'ChromaDB (BGE-M3)',
            'NDCG@10': row['chroma_ndcg_at_10']
        })
    
    # Combine data
    ndcg_combined = pd.DataFrame(random_ndcg_data + bge_ndcg_data)
    
    sns.lineplot(data=ndcg_combined, x='Data Size', y='NDCG@10', 
                hue='Database', marker='o', ax=ax1)
    ax1.set_title('NDCG@10: Random vs BGE-M3 Embeddings', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NDCG@10 Score')
    ax1.set_xlabel('Dataset Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Search Performance Comparison
    ax2 = axes[0, 1]
    
    # Performance data
    perf_data = []
    
    # Random vectors
    for size in [1000, 5000, 10000, 50000]:
        size_data = random_df[random_df['data_size'] == size]
        if not size_data.empty:
            perf_data.append({
                'Data Size': size,
                'Database': 'Qdrant (Random)',
                'QPS': size_data['qdrant_qps'].mean()
            })
            perf_data.append({
                'Data Size': size,
                'Database': 'ChromaDB (Random)',
                'QPS': size_data['chroma_qps'].mean()
            })
    
    # BGE data
    for _, row in bge_df.iterrows():
        perf_data.append({
            'Data Size': row['data_size'],
            'Database': 'Qdrant (BGE-M3)',
            'QPS': row['qdrant_qps']
        })
        perf_data.append({
            'Data Size': row['data_size'],
            'Database': 'ChromaDB (BGE-M3)',
            'QPS': row['chroma_qps']
        })
    
    perf_df = pd.DataFrame(perf_data)
    
    sns.lineplot(data=perf_df, x='Data Size', y='QPS', 
                hue='Database', marker='s', ax=ax2)
    ax2.set_title('Query Performance: QPS Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Queries per Second')
    ax2.set_xlabel('Dataset Size')
    ax2.set_yscale('log')  # Log scale due to large differences
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Relevance Quality by Database
    ax3 = axes[0, 2]
    
    # Average relevance metrics for BGE
    bge_relevance_data = []
    for db in ['qdrant', 'chroma']:
        for k in [1, 5, 10, 20]:
            avg_ndcg = bge_df[f'{db}_ndcg_at_{k}'].mean()
            bge_relevance_data.append({
                'Database': 'Qdrant' if db == 'qdrant' else 'ChromaDB',
                'k': k,
                'NDCG': avg_ndcg
            })
    
    relevance_df = pd.DataFrame(bge_relevance_data)
    
    sns.barplot(data=relevance_df, x='k', y='NDCG', hue='Database', ax=ax3)
    ax3.set_title('BGE-M3 Relevance Quality by K', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average NDCG@k')
    ax3.set_xlabel('Top-k Results')
    ax3.grid(True, alpha=0.3)
    
    # 4. Embedding Quality Impact
    ax4 = axes[1, 0]
    
    # Compare average NDCG@10 between random and BGE
    embedding_comparison = [
        {'Embedding Type': 'Random Vectors', 'Database': 'Qdrant', 'NDCG@10': random_df['qdrant_ndcg_at_10'].mean()},
        {'Embedding Type': 'Random Vectors', 'Database': 'ChromaDB', 'NDCG@10': random_df['chroma_ndcg_at_10'].mean()},
        {'Embedding Type': 'BGE-M3', 'Database': 'Qdrant', 'NDCG@10': bge_df['qdrant_ndcg_at_10'].mean()},
        {'Embedding Type': 'BGE-M3', 'Database': 'ChromaDB', 'NDCG@10': bge_df['chroma_ndcg_at_10'].mean()}
    ]
    
    embedding_df = pd.DataFrame(embedding_comparison)
    
    sns.barplot(data=embedding_df, x='Embedding Type', y='NDCG@10', hue='Database', ax=ax4)
    ax4.set_title('Embedding Quality Impact on NDCG@10', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average NDCG@10')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.3f', fontsize=10)
    
    # 5. Performance vs Quality Trade-off
    ax5 = axes[1, 1]
    
    # Create scatter plot of QPS vs NDCG@10 for BGE data
    scatter_data = []
    for _, row in bge_df.iterrows():
        scatter_data.append({
            'Database': 'Qdrant',
            'QPS': row['qdrant_qps'],
            'NDCG@10': row['qdrant_ndcg_at_10'],
            'Data Size': row['data_size']
        })
        scatter_data.append({
            'Database': 'ChromaDB',
            'QPS': row['chroma_qps'],
            'NDCG@10': row['chroma_ndcg_at_10'],
            'Data Size': row['data_size']
        })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    sns.scatterplot(data=scatter_df, x='QPS', y='NDCG@10', 
                   hue='Database', size='Data Size', sizes=(50, 200), ax=ax5)
    ax5.set_title('Performance vs Quality Trade-off (BGE-M3)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Queries per Second')
    ax5.set_ylabel('NDCG@10 Score')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. Scaling Analysis
    ax6 = axes[1, 2]
    
    # Show how performance scales with data size
    scaling_data = []
    for _, row in bge_df.iterrows():
        scaling_data.append({
            'Data Size': row['data_size'],
            'Database': 'Qdrant',
            'Search Time (ms)': row['qdrant_search_per_query'] * 1000
        })
        scaling_data.append({
            'Data Size': row['data_size'],
            'Database': 'ChromaDB',
            'Search Time (ms)': row['chroma_search_per_query'] * 1000
        })
    
    scaling_df = pd.DataFrame(scaling_data)
    
    sns.lineplot(data=scaling_df, x='Data Size', y='Search Time (ms)', 
                hue='Database', marker='o', ax=ax6)
    ax6.set_title('Search Latency Scaling (BGE-M3)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Search Time per Query (ms)')
    ax6.set_xlabel('Dataset Size')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive comparison saved to 'comprehensive_benchmark_comparison.png'")
    
    # Create summary statistics table
    create_summary_table(random_df, bge_df)

def create_summary_table(random_df, bge_df):
    """Create a summary statistics table"""
    print("\n📊 Creating summary statistics table...")
    
    summary_stats = {
        'Metric': [
            'Average NDCG@10 (Random)',
            'Average NDCG@10 (BGE-M3)',
            'NDCG Improvement with BGE-M3',
            'Average QPS (Random)',
            'Average QPS (BGE-M3)',
            'Qdrant Relevance Advantage (BGE-M3)',
            'ChromaDB Speed Advantage (BGE-M3)'
        ],
        'Qdrant': [
            f"{random_df['qdrant_ndcg_at_10'].mean():.4f}",
            f"{bge_df['qdrant_ndcg_at_10'].mean():.4f}",
            f"{(bge_df['qdrant_ndcg_at_10'].mean() / random_df['qdrant_ndcg_at_10'].mean() - 1) * 100:.1f}%",
            f"{random_df['qdrant_qps'].mean():.1f}",
            f"{bge_df['qdrant_qps'].mean():.1f}",
            f"{(bge_df['qdrant_ndcg_at_10'].mean() / bge_df['chroma_ndcg_at_10'].mean() - 1) * 100:.1f}%",
            f"-"
        ],
        'ChromaDB': [
            f"{random_df['chroma_ndcg_at_10'].mean():.4f}",
            f"{bge_df['chroma_ndcg_at_10'].mean():.4f}",
            f"{(bge_df['chroma_ndcg_at_10'].mean() / random_df['chroma_ndcg_at_10'].mean() - 1) * 100:.1f}%",
            f"{random_df['chroma_qps'].mean():.1f}",
            f"{bge_df['chroma_qps'].mean():.1f}",
            f"-",
            f"{(bge_df['chroma_qps'].mean() / bge_df['qdrant_qps'].mean()):.1f}x"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('benchmark_summary_table.csv', index=False)
    
    print("✅ Summary table saved to 'benchmark_summary_table.csv'")
    print("\n📋 Key Statistics Summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    create_comparison_visualizations()
