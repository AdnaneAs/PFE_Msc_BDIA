#!/usr/bin/env python3
"""
BGE Reranker Benchmark on HotpotQA Dataset
Comprehensive evaluation of BGE reranking vs standard retrieval
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import asyncio
import aiohttp
from collections import defaultdict

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class BenchmarkResult:
    """Structure for storing benchmark results"""
    query: str
    ground_truth_answer: str
    retrieval_method: str
    retrieved_docs: List[str]
    relevance_scores: List[float]
    llm_answer: str
    response_time_ms: int
    num_sources: int
    avg_relevance: float
    top_relevance: float
    search_strategy: str
    reranker_model: str = None

class BGERerankerBenchmark:
    """Comprehensive benchmark for BGE reranker evaluation"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000/api/query"):
        self.api_base_url = api_base_url
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"benchmark_results_{self.timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Benchmark configuration
        self.reranker_models = [
            "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-large", 
            "BAAI/bge-reranker-v2-m3"
        ]
        
        self.search_strategies = ["semantic", "hybrid"]
        self.max_sources_configs = [3, 5, 8, 10]
        
        print(f"🎯 BGE Reranker Benchmark Initialized")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
    def load_hotpot_dataset(self, dataset_path: str = None, sample_size: int = 100) -> List[Dict]:
        """Load HotpotQA dataset samples"""
        print(f"📚 Loading HotpotQA dataset...")
        
        # For this demo, create synthetic HotpotQA-style questions
        # In practice, you would load from the actual dataset file
        synthetic_hotpot_questions = [
            {
                "question": "What is the capital of the country where the Eiffel Tower is located?",
                "answer": "Paris",
                "type": "bridge",
                "level": "medium"
            },
            {
                "question": "Which university did the author of 'To Kill a Mockingbird' attend?",
                "answer": "University of Alabama",
                "type": "bridge", 
                "level": "hard"
            },
            {
                "question": "What is the population of the largest city in Japan?",
                "answer": "Tokyo has approximately 14 million people",
                "type": "comparison",
                "level": "easy"
            },
            {
                "question": "In what year was the company that created the iPhone founded?",
                "answer": "Apple was founded in 1976",
                "type": "bridge",
                "level": "medium"
            },
            {
                "question": "What is the height of the tallest building in the city known as the Big Apple?",
                "answer": "One World Trade Center is 1,776 feet tall",
                "type": "bridge",
                "level": "hard"
            },
            {
                "question": "Which planet is closest to the sun in our solar system?",
                "answer": "Mercury",
                "type": "factoid",
                "level": "easy"
            },
            {
                "question": "What programming language was used to create the Linux kernel?",
                "answer": "C programming language",
                "type": "factoid",
                "level": "medium"
            },
            {
                "question": "Who wrote the novel that was adapted into the movie 'The Shawshank Redemption'?",
                "answer": "Stephen King",
                "type": "bridge",
                "level": "medium"
            },
            {
                "question": "What is the chemical symbol for the element with atomic number 79?",
                "answer": "Au (Gold)",
                "type": "factoid",
                "level": "hard"
            },
            {
                "question": "In which decade was the World Wide Web invented?",
                "answer": "1990s (1989-1991)",
                "type": "factoid",
                "level": "easy"
            },
            # Machine Learning and AI questions
            {
                "question": "What type of neural network architecture is primarily used for image classification?",
                "answer": "Convolutional Neural Networks (CNNs)",
                "type": "factoid",
                "level": "medium"
            },
            {
                "question": "Which algorithm is commonly used for training deep neural networks?",
                "answer": "Backpropagation with gradient descent",
                "type": "factoid", 
                "level": "medium"
            },
            {
                "question": "What is the difference between supervised and unsupervised learning?",
                "answer": "Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data",
                "type": "comparison",
                "level": "medium"
            },
            {
                "question": "Which company developed the transformer architecture that revolutionized NLP?",
                "answer": "Google (in the paper 'Attention Is All You Need')",
                "type": "bridge",
                "level": "hard"
            },
            {
                "question": "What does GPU stand for and why is it important for machine learning?",
                "answer": "Graphics Processing Unit - important for parallel processing in ML training",
                "type": "bridge",
                "level": "medium"
            }
        ]
        
        # Sample the requested number of questions
        import random
        random.seed(42)  # For reproducible results
        questions = random.sample(synthetic_hotpot_questions, min(sample_size, len(synthetic_hotpot_questions)))
        
        print(f"✅ Loaded {len(questions)} HotpotQA questions")
        return questions
    
    async def query_api_async(self, session: aiohttp.ClientSession, query_data: Dict) -> Dict:
        """Make async API call to query endpoint"""
        try:
            async with session.post(f"{self.api_base_url}/", json=query_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def query_api_sync(self, query_data: Dict) -> Dict:
        """Make synchronous API call to query endpoint"""
        try:
            response = requests.post(f"{self.api_base_url}/", json=query_data, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_single_query(self, question: Dict, config: Dict) -> BenchmarkResult:
        """Evaluate a single query with given configuration"""
        query_data = {
            "question": question["question"],
            "max_sources": config["max_sources"],
            "search_strategy": config["search_strategy"],
            "use_reranking": config["use_reranking"],
            "reranker_model": config.get("reranker_model", "BAAI/bge-reranker-base")
        }
        
        start_time = time.time()
        result = self.query_api_sync(query_data)
        end_time = time.time()
        
        if "error" in result:
            print(f"❌ Error querying '{question['question'][:50]}...': {result['error']}")
            return None
        
        # Calculate metrics
        relevance_scores = []
        if "sources" in result:
            for source in result["sources"]:
                # Extract relevance score from metadata if available
                score = source.get("relevance_score", 0.0)
                relevance_scores.append(score)
        
        retrieval_method = f"{config['search_strategy']}"
        if config["use_reranking"]:
            retrieval_method += f" + {config['reranker_model'].split('/')[-1]}"
        
        return BenchmarkResult(
            query=question["question"],
            ground_truth_answer=question["answer"],
            retrieval_method=retrieval_method,
            retrieved_docs=[source.get("content", "")[:200] for source in result.get("sources", [])],
            relevance_scores=relevance_scores,
            llm_answer=result.get("answer", ""),
            response_time_ms=int((end_time - start_time) * 1000),
            num_sources=result.get("num_sources", 0),
            avg_relevance=result.get("average_relevance", 0.0) if result.get("average_relevance") else (sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0),
            top_relevance=result.get("top_relevance", 0.0) if result.get("top_relevance") else (max(relevance_scores) if relevance_scores else 0.0),
            search_strategy=result.get("search_strategy", config["search_strategy"]),
            reranker_model=result.get("reranker_model") if config["use_reranking"] else None
        )
    
    def run_comprehensive_benchmark(self, questions: List[Dict]) -> pd.DataFrame:
        """Run comprehensive benchmark across all configurations"""
        print(f"🚀 Starting comprehensive benchmark with {len(questions)} questions")
        
        all_results = []
        total_configs = 0
        
        # Count total configurations
        for search_strategy in self.search_strategies:
            for max_sources in self.max_sources_configs:
                total_configs += 1  # Without reranking
                total_configs += len(self.reranker_models)  # With each reranker model
        
        current_config = 0
        
        for search_strategy in self.search_strategies:
            for max_sources in self.max_sources_configs:
                
                # Test without reranking
                current_config += 1
                print(f"\n📊 Configuration {current_config}/{total_configs}: {search_strategy}, {max_sources} sources, NO reranking")
                
                config = {
                    "search_strategy": search_strategy,
                    "max_sources": max_sources,
                    "use_reranking": False
                }
                
                for i, question in enumerate(questions):
                    print(f"   Query {i+1}/{len(questions)}: {question['question'][:50]}...")
                    result = self.evaluate_single_query(question, config)
                    if result:
                        all_results.append(result)
                    time.sleep(0.1)  # Brief pause to avoid overwhelming the API
                
                # Test with each reranker model
                for reranker_model in self.reranker_models:
                    current_config += 1
                    model_name = reranker_model.split("/")[-1]
                    print(f"\n📊 Configuration {current_config}/{total_configs}: {search_strategy}, {max_sources} sources, {model_name}")
                    
                    config = {
                        "search_strategy": search_strategy,
                        "max_sources": max_sources,
                        "use_reranking": True,
                        "reranker_model": reranker_model
                    }
                    
                    for i, question in enumerate(questions):
                        print(f"   Query {i+1}/{len(questions)}: {question['question'][:50]}...")
                        result = self.evaluate_single_query(question, config)
                        if result:
                            all_results.append(result)
                        time.sleep(0.1)  # Brief pause
        
        # Convert to DataFrame
        df_data = []
        for result in all_results:
            df_data.append({
                "query": result.query,
                "ground_truth": result.ground_truth_answer,
                "retrieval_method": result.retrieval_method,
                "llm_answer": result.llm_answer,
                "response_time_ms": result.response_time_ms,
                "num_sources": result.num_sources,
                "avg_relevance": result.avg_relevance,
                "top_relevance": result.top_relevance,
                "search_strategy": result.search_strategy,
                "reranker_model": result.reranker_model,
                "uses_reranking": result.reranker_model is not None
            })
        
        df = pd.DataFrame(df_data)
        
        # Save raw results
        results_file = self.results_dir / f"bge_reranker_benchmark_results_{self.timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"💾 Raw results saved to: {results_file}")
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        print("📈 Calculating evaluation metrics...")
        
        metrics = {}
        
        # Group by retrieval method
        grouped = df.groupby('retrieval_method')
        
        for method, group in grouped:
            method_metrics = {
                "avg_response_time_ms": group["response_time_ms"].mean(),
                "median_response_time_ms": group["response_time_ms"].median(),
                "std_response_time_ms": group["response_time_ms"].std(),
                "avg_num_sources": group["num_sources"].mean(),
                "avg_relevance_score": group["avg_relevance"].mean(),
                "avg_top_relevance": group["top_relevance"].mean(),
                "median_relevance_score": group["avg_relevance"].median(),
                "std_relevance_score": group["avg_relevance"].std(),
                "total_queries": len(group),
                "queries_with_sources": len(group[group["num_sources"] > 0]),
                "zero_source_rate": len(group[group["num_sources"] == 0]) / len(group)
            }
            metrics[method] = method_metrics
        
        return metrics
    
    def generate_comparison_charts(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """Generate comprehensive comparison charts"""
        print("📊 Generating comparison charts...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Response Time Comparison
        plt.subplot(3, 3, 1)
        response_times = []
        method_names = []
        for method, group in df.groupby('retrieval_method'):
            response_times.append(group["response_time_ms"].values)
            method_names.append(method.replace(" + ", "\n+\n"))
        
        plt.boxplot(response_times, labels=method_names)
        plt.title("Response Time Distribution", fontsize=12, fontweight='bold')
        plt.ylabel("Response Time (ms)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 2. Relevance Score Comparison
        plt.subplot(3, 3, 2)
        relevance_scores = []
        for method, group in df.groupby('retrieval_method'):
            relevance_scores.append(group["avg_relevance"].values)
        
        plt.boxplot(relevance_scores, labels=method_names)
        plt.title("Average Relevance Score Distribution", fontsize=12, fontweight='bold')
        plt.ylabel("Relevance Score")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 3. Number of Sources Retrieved
        plt.subplot(3, 3, 3)
        num_sources = []
        for method, group in df.groupby('retrieval_method'):
            num_sources.append(group["num_sources"].values)
        
        plt.boxplot(num_sources, labels=method_names)
        plt.title("Number of Sources Retrieved", fontsize=12, fontweight='bold')
        plt.ylabel("Number of Sources")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 4. Reranking vs No Reranking Performance
        plt.subplot(3, 3, 4)
        reranking_comparison = df.groupby('uses_reranking').agg({
            'avg_relevance': 'mean',
            'response_time_ms': 'mean'
        })
        
        x_pos = [0, 1]
        plt.bar(x_pos, reranking_comparison['avg_relevance'], 
                color=['lightcoral', 'lightblue'], alpha=0.7)
        plt.title("Relevance: Reranking vs No Reranking", fontsize=12, fontweight='bold')
        plt.ylabel("Average Relevance Score")
        plt.xticks(x_pos, ['No Reranking', 'With Reranking'])
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(reranking_comparison['avg_relevance']):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Response Time: Reranking vs No Reranking
        plt.subplot(3, 3, 5)
        plt.bar(x_pos, reranking_comparison['response_time_ms'], 
                color=['lightcoral', 'lightblue'], alpha=0.7)
        plt.title("Response Time: Reranking vs No Reranking", fontsize=12, fontweight='bold')
        plt.ylabel("Average Response Time (ms)")
        plt.xticks(x_pos, ['No Reranking', 'With Reranking'])
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(reranking_comparison['response_time_ms']):
            plt.text(i, v + 20, f'{v:.0f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 6. Top Relevance Score by Method
        plt.subplot(3, 3, 6)
        top_relevance_by_method = df.groupby('retrieval_method')['top_relevance'].mean().sort_values(ascending=False)
        
        bars = plt.bar(range(len(top_relevance_by_method)), top_relevance_by_method.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(top_relevance_by_method))))
        plt.title("Top Relevance Score by Method", fontsize=12, fontweight='bold')
        plt.ylabel("Average Top Relevance Score")
        plt.xticks(range(len(top_relevance_by_method)), 
                  [method.replace(" + ", "\n+\n") for method in top_relevance_by_method.index], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 7. Search Strategy Performance
        plt.subplot(3, 3, 7)
        strategy_performance = df.groupby('search_strategy').agg({
            'avg_relevance': 'mean',
            'response_time_ms': 'mean'
        })
        
        x_strategies = range(len(strategy_performance))
        width = 0.35
        
        plt.bar([x - width/2 for x in x_strategies], strategy_performance['avg_relevance'], 
                width, label='Avg Relevance', color='skyblue', alpha=0.7)
        
        ax2 = plt.twinx()
        ax2.bar([x + width/2 for x in x_strategies], strategy_performance['response_time_ms'], 
                width, label='Response Time (ms)', color='lightcoral', alpha=0.7)
        
        plt.title("Search Strategy Performance", fontsize=12, fontweight='bold')
        plt.ylabel("Average Relevance Score")
        ax2.set_ylabel("Average Response Time (ms)")
        plt.xticks(x_strategies, strategy_performance.index)
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 8. Reranker Model Comparison (if applicable)
        plt.subplot(3, 3, 8)
        reranker_data = df[df['uses_reranking'] == True]
        if not reranker_data.empty:
            reranker_performance = reranker_data.groupby('reranker_model').agg({
                'avg_relevance': 'mean',
                'response_time_ms': 'mean'
            })
            
            x_models = range(len(reranker_performance))
            plt.bar(x_models, reranker_performance['avg_relevance'], 
                   color='lightgreen', alpha=0.7)
            plt.title("Reranker Model Comparison", fontsize=12, fontweight='bold')
            plt.ylabel("Average Relevance Score")
            plt.xticks(x_models, [model.split('/')[-1] if model else 'None' 
                                for model in reranker_performance.index], rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 9. Success Rate (Queries with Sources)
        plt.subplot(3, 3, 9)
        success_rates = []
        methods = []
        for method, group in df.groupby('retrieval_method'):
            success_rate = len(group[group["num_sources"] > 0]) / len(group) * 100
            success_rates.append(success_rate)
            methods.append(method.replace(" + ", "\n+\n"))
        
        bars = plt.bar(range(len(success_rates)), success_rates, 
                      color=plt.cm.RdYlGn(np.array(success_rates)/100))
        plt.title("Success Rate (Queries with Sources)", fontsize=12, fontweight='bold')
        plt.ylabel("Success Rate (%)")
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(success_rates):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        chart_file = self.results_dir / f"bge_reranker_comparison_charts_{self.timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison charts saved to: {chart_file}")
        
        plt.show()
    
    def generate_detailed_report(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """Generate detailed benchmark report"""
        print("📝 Generating detailed benchmark report...")
        
        report_file = self.results_dir / f"BGE_RERANKER_BENCHMARK_REPORT_{self.timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# BGE Reranker Benchmark Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** HotpotQA-style questions ({len(df['query'].unique())} unique queries)\n\n")
            f.write(f"**Total Evaluations:** {len(df)} query-configuration pairs\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Overall performance comparison
            reranking_comparison = df.groupby('uses_reranking').agg({
                'avg_relevance': ['mean', 'std'],
                'response_time_ms': ['mean', 'std'],
                'num_sources': 'mean'
            }).round(3)
            
            f.write("### Reranking vs No Reranking\n\n")
            f.write("| Metric | No Reranking | With Reranking | Improvement |\n")
            f.write("|--------|--------------|----------------|-------------|\n")
            
            no_rerank_relevance = reranking_comparison.loc[False, ('avg_relevance', 'mean')]
            with_rerank_relevance = reranking_comparison.loc[True, ('avg_relevance', 'mean')]
            relevance_improvement = ((with_rerank_relevance - no_rerank_relevance) / no_rerank_relevance * 100)
            
            no_rerank_time = reranking_comparison.loc[False, ('response_time_ms', 'mean')]
            with_rerank_time = reranking_comparison.loc[True, ('response_time_ms', 'mean')]
            time_overhead = ((with_rerank_time - no_rerank_time) / no_rerank_time * 100)
            
            f.write(f"| Avg Relevance | {no_rerank_relevance:.3f} | {with_rerank_relevance:.3f} | {relevance_improvement:+.1f}% |\n")
            f.write(f"| Response Time (ms) | {no_rerank_time:.0f} | {with_rerank_time:.0f} | {time_overhead:+.1f}% |\n")
            f.write(f"| Avg Sources | {reranking_comparison.loc[False, ('num_sources', 'mean')]:.1f} | {reranking_comparison.loc[True, ('num_sources', 'mean')]:.1f} | - |\n\n")
            
            f.write("## Detailed Results by Method\n\n")
            
            # Detailed breakdown by method
            for method, method_metrics in metrics.items():
                f.write(f"### {method}\n\n")
                f.write(f"- **Average Response Time:** {method_metrics['avg_response_time_ms']:.1f}ms (±{method_metrics['std_response_time_ms']:.1f}ms)\n")
                f.write(f"- **Average Relevance Score:** {method_metrics['avg_relevance_score']:.3f} (±{method_metrics['std_relevance_score']:.3f})\n")
                f.write(f"- **Average Top Relevance:** {method_metrics['avg_top_relevance']:.3f}\n")
                f.write(f"- **Average Sources Retrieved:** {method_metrics['avg_num_sources']:.1f}\n")
                f.write(f"- **Success Rate:** {(1 - method_metrics['zero_source_rate'])*100:.1f}%\n")
                f.write(f"- **Total Queries:** {method_metrics['total_queries']}\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Best performing methods
            best_relevance_method = max(metrics.keys(), key=lambda x: metrics[x]['avg_relevance_score'])
            fastest_method = min(metrics.keys(), key=lambda x: metrics[x]['avg_response_time_ms'])
            
            f.write(f"### Best Performance\n")
            f.write(f"- **Highest Relevance:** {best_relevance_method} ({metrics[best_relevance_method]['avg_relevance_score']:.3f})\n")
            f.write(f"- **Fastest Response:** {fastest_method} ({metrics[fastest_method]['avg_response_time_ms']:.1f}ms)\n\n")
            
            f.write("### Recommendations\n\n")
            
            if relevance_improvement > 5:
                f.write(f"✅ **BGE Reranking shows significant improvement** ({relevance_improvement:.1f}% better relevance)\n\n")
                f.write("**Recommended for:**\n")
                f.write("- Production environments where answer quality is critical\n")
                f.write("- Complex multi-hop reasoning queries\n")
                f.write("- Applications where users expect high-quality results\n\n")
            else:
                f.write(f"⚠️ **BGE Reranking shows marginal improvement** ({relevance_improvement:.1f}% better relevance)\n\n")
            
            if time_overhead > 50:
                f.write(f"⚠️ **Significant latency overhead** ({time_overhead:.1f}% slower)\n")
                f.write("- Consider using faster reranker models for latency-sensitive applications\n")
                f.write("- Implement caching strategies for repeated queries\n\n")
            
            f.write("## Configuration Details\n\n")
            f.write(f"- **Reranker Models Tested:** {', '.join(self.reranker_models)}\n")
            f.write(f"- **Search Strategies:** {', '.join(self.search_strategies)}\n")
            f.write(f"- **Source Counts:** {', '.join(map(str, self.max_sources_configs))}\n")
            f.write(f"- **API Endpoint:** {self.api_base_url}\n\n")
            
            f.write("## Raw Data\n\n")
            f.write("Detailed results are available in the CSV file:\n")
            f.write(f"`bge_reranker_benchmark_results_{self.timestamp}.csv`\n\n")
        
        print(f"📝 Detailed report saved to: {report_file}")
        return report_file

def main():
    """Main benchmark execution"""
    print("🎯 BGE Reranker Benchmark on HotpotQA Dataset")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = BGERerankerBenchmark()
    
    # Check if API is accessible
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend API is not accessible. Please start the backend server first.")
            print("   Run: cd backend && uvicorn app.main:app --reload")
            return
    except requests.exceptions.RequestException:
        print("❌ Backend API is not accessible. Please start the backend server first.")
        print("   Run: cd backend && uvicorn app.main:app --reload")
        return
    
    print("✅ Backend API is accessible")
    
    # Load dataset
    questions = benchmark.load_hotpot_dataset(sample_size=15)  # Start with smaller sample
    
    # Run benchmark
    df = benchmark.run_comprehensive_benchmark(questions)
    
    if df.empty:
        print("❌ No benchmark results obtained. Check API connectivity and configuration.")
        return
    
    # Calculate metrics
    metrics = benchmark.calculate_metrics(df)
    
    # Generate visualizations
    benchmark.generate_comparison_charts(df, metrics)
    
    # Generate report
    report_file = benchmark.generate_detailed_report(df, metrics)
    
    print(f"\n🎉 Benchmark completed successfully!")
    print(f"📁 Results saved to: {benchmark.results_dir}")
    print(f"📊 Charts: bge_reranker_comparison_charts_{benchmark.timestamp}.png")
    print(f"📝 Report: {report_file.name}")
    print(f"💾 Data: bge_reranker_benchmark_results_{benchmark.timestamp}.csv")

if __name__ == "__main__":
    main()
