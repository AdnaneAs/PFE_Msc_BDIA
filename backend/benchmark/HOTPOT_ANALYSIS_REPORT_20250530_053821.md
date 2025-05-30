# HotpotQA Vector Database Benchmark Analysis

**Analysis Date:** May 30, 2025
**Dataset:** HotpotQA Multi-hop Reasoning Dataset
**Questions Tested:** 100

## Executive Summary

This comprehensive benchmark evaluates Qdrant and ChromaDB vector databases using the HotpotQA dataset, which contains complex multi-hop reasoning questions requiring information from multiple Wikipedia articles.

## Dataset Overview

**HotpotQA Dataset Characteristics:**
- 113,000+ question-answer pairs from Wikipedia
- Multi-hop reasoning requirements
- Supporting facts explicitly provided
- Diverse question types (comparison, bridge reasoning)
- Hard and medium difficulty levels

## Performance Analysis

### BGE-M3 Results

| Metric | Qdrant | ChromaDB | Winner |
|--------|--------|----------|--------|
| Ndcg@10 | 0.862 | 0.839 | **Qdrant** |
| Precision@10 | 0.205 | 0.198 | **Qdrant** |
| Recall@10 | 0.868 | 0.833 | **Qdrant** |
| Hit Rate@10 | 0.980 | 0.950 | **Qdrant** |
| Avg Query Time | 16.386ms | 3.380ms | **ChromaDB** |
| Queries Per Second | 61.026q/s | 295.841q/s | **ChromaDB** |

#### Key Insights:

- **Relevance Quality**: Qdrant shows +2.8% NDCG@10 compared to ChromaDB
- **Query Speed**: ChromaDB is +384.8% faster than Qdrant
- **Precision@10**: Qdrant achieves 0.205 vs ChromaDB's 0.198
- **Recall@10**: Qdrant achieves 0.868 vs ChromaDB's 0.833

### ALL-MINILM Results

| Metric | Qdrant | ChromaDB | Winner |
|--------|--------|----------|--------|
| Ndcg@10 | 0.793 | 0.790 | **Qdrant** |
| Precision@10 | 0.173 | 0.167 | **Qdrant** |
| Recall@10 | 0.733 | 0.704 | **Qdrant** |
| Hit Rate@10 | 0.980 | 0.980 | **ChromaDB** |
| Avg Query Time | 13.870ms | 2.272ms | **ChromaDB** |
| Queries Per Second | 72.100q/s | 440.051q/s | **ChromaDB** |

#### Key Insights:

- **Relevance Quality**: Qdrant shows +0.4% NDCG@10 compared to ChromaDB
- **Query Speed**: ChromaDB is +510.3% faster than Qdrant
- **Precision@10**: Qdrant achieves 0.173 vs ChromaDB's 0.167
- **Recall@10**: Qdrant achieves 0.733 vs ChromaDB's 0.704

## Technical Analysis

### Multi-hop Reasoning Performance

HotpotQA's multi-hop questions require systems to:
1. Identify multiple relevant documents
2. Extract supporting facts from each document
3. Combine information across documents

### Vector Database Architecture Impact

**Qdrant (Server-based):**
- Network latency adds to query time
- Optimized vector operations
- Better relevance ranking algorithms
- More sophisticated similarity search

**ChromaDB (Embedded):**
- In-process execution reduces latency
- Simpler similarity calculations
- Less memory overhead per query
- Faster for simple retrieval tasks

## Recommendations

### For Production RAG Systems:

**Recommended: Qdrant**

Qdrant demonstrates superior relevance quality in multi-hop reasoning tasks, which is critical for complex question-answering systems. The performance gains in NDCG@10 outweigh the speed disadvantage for most production use cases.

### Use Case Guidelines:

- **Choose Qdrant if**: Relevance quality is paramount, complex reasoning required
- **Choose ChromaDB if**: Query speed is critical, simpler retrieval tasks
- **Consider hybrid approach**: Use both databases for different query types

## Methodology

### Embedding Models Tested:
- **bge-m3**: BAAI/bge-m3 (1024 dimensions, state-of-the-art quality)
- **all-minilm**: all-MiniLM-L6-v2 (384 dimensions, fast inference)

### Evaluation Metrics:
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved
- **Hit Rate@k**: Whether any relevant document appears in top-k
- **Query Speed**: Average time per query and queries per second

### Ground Truth:
Ground truth relevance is determined by HotpotQA's supporting facts, which explicitly identify which sentences from which documents are necessary to answer each question.

---
*This analysis was generated automatically by the HotpotQA Vector Database Benchmark suite.*
