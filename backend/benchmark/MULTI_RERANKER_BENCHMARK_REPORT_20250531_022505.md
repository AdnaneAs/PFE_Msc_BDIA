# Multi-Reranker Benchmark Report

**Generated:** 2025-05-31 02:25:05

## Executive Summary

This benchmark compares the performance of different reranking strategies:

- **No Reranker**: Baseline vector similarity search only
- **BGE Reranker Base**: Balanced performance and quality
- **BGE Reranker Large**: Higher quality, more parameters
- **BGE Reranker v2-M3**: Latest multilingual model

**Evaluation Dataset:** 10 HotpotQA questions
**Documents per Query:** Up to 20 retrieved, top 10 evaluated

## Performance Comparison

### Key Metrics Summary

| Reranker | MAP | NDCG@5 | Precision@5 | MRR | Hit Rate@5 | Processing Time (s) |
|----------|-----|--------|-------------|-----|-------------|--------------------|
| No Reranker | 0.000 | 0.640 | 0.000 | 0.000 | 0.000 | 0.029 |
| BGE Reranker Base | 0.000 | 0.727 | 0.000 | 0.000 | 0.000 | 0.331 |
| BGE Reranker Large | 0.000 | 0.498 | 0.000 | 0.000 | 0.000 | 1.015 |
| BGE Reranker v2-M3 | 0.000 | 0.464 | 0.000 | 0.000 | 0.000 | 29.259 |

### Best Overall Performance

**No Reranker** achieved the highest MAP score of 0.000

## Detailed Analysis

### No Reranker

**Model:** None
**Description:** Baseline vector similarity search only

**Performance Metrics:**
- MAP: 0.000
- NDCG@5: 0.640
- Precision@5: 0.000
- MRR: 0.000
- Hit Rate@5: 0.000
- Processing Time: 0.029s

### BGE Reranker Base

**Model:** BAAI/bge-reranker-base
**Description:** Balanced performance and quality

**Performance Metrics:**
- MAP: 0.000
- NDCG@5: 0.727
- Precision@5: 0.000
- MRR: 0.000
- Hit Rate@5: 0.000
- Processing Time: 0.331s

### BGE Reranker Large

**Model:** BAAI/bge-reranker-large
**Description:** Higher quality, more parameters

**Performance Metrics:**
- MAP: 0.000
- NDCG@5: 0.498
- Precision@5: 0.000
- MRR: 0.000
- Hit Rate@5: 0.000
- Processing Time: 1.015s

### BGE Reranker v2-M3

**Model:** BAAI/bge-reranker-v2-m3
**Description:** Latest multilingual model

**Performance Metrics:**
- MAP: 0.000
- NDCG@5: 0.464
- Precision@5: 0.000
- MRR: 0.000
- Hit Rate@5: 0.000
- Processing Time: 29.259s

## Recommendations

Based on this benchmark analysis:

### Performance vs Speed Trade-offs:

- **BGE Reranker Base**: MAP improvement +0.0%, Time overhead +1056.0%
- **BGE Reranker Large**: MAP improvement +0.0%, Time overhead +3442.2%
- **BGE Reranker v2-M3**: MAP improvement +0.0%, Time overhead +101984.3%

### Production Recommendations:

Choose the reranker based on your priority:

- **Speed Priority**: No reranker (fastest baseline)
- **Balanced Performance**: BGE Reranker Base (good quality/speed balance)
- **Quality Priority**: BGE Reranker Large or v2-M3 (best performance)
- **Multilingual Support**: BGE Reranker v2-M3 (latest multilingual model)

