# Embedding Models Benchmark Report

**Generated:** 2025-05-30 06:00:20

**Models Tested:** 10

## Executive Summary

- **Best Semantic Similarity:** e5-small-v2 (0.849)
- **Best Clustering Quality:** paraphrase-MiniLM-L6-v2 (0.087)
- **Best Classification Transfer:** all-distilroberta-v1 (0.880)
- **Most Efficient:** all-MiniLM-L6-v2 (3491.8 sent/sec)

## Detailed Results

### Models Ranked by Overall Performance

1. **e5-small-v2** (Overall: 0.594)
   - Semantic Similarity: 0.849
   - Clustering Quality: 0.068
   - Classification Accuracy: 0.863
   - Encoding Speed: 1958.1 sent/sec
   - Dimensions: 384

2. **paraphrase-MiniLM-L6-v2** (Overall: 0.588)
   - Semantic Similarity: 0.841
   - Clustering Quality: 0.087
   - Classification Accuracy: 0.837
   - Encoding Speed: 3468.6 sent/sec
   - Dimensions: 384

3. **all-mpnet-base-v2** (Overall: 0.587)
   - Semantic Similarity: 0.834
   - Clustering Quality: 0.072
   - Classification Accuracy: 0.857
   - Encoding Speed: 711.3 sent/sec
   - Dimensions: 768

4. **all-distilroberta-v1** (Overall: 0.587)
   - Semantic Similarity: 0.825
   - Clustering Quality: 0.056
   - Classification Accuracy: 0.880
   - Encoding Speed: 1437.9 sent/sec
   - Dimensions: 768

5. **all-MiniLM-L6-v2** (Overall: 0.576)
   - Semantic Similarity: 0.820
   - Clustering Quality: 0.055
   - Classification Accuracy: 0.853
   - Encoding Speed: 3491.8 sent/sec
   - Dimensions: 384

6. **paraphrase-albert-small-v2** (Overall: 0.572)
   - Semantic Similarity: 0.834
   - Clustering Quality: 0.080
   - Classification Accuracy: 0.803
   - Encoding Speed: 1351.4 sent/sec
   - Dimensions: 768

7. **multi-qa-distilbert-cos-v1** (Overall: 0.556)
   - Semantic Similarity: 0.747
   - Clustering Quality: 0.065
   - Classification Accuracy: 0.857
   - Encoding Speed: 1299.8 sent/sec
   - Dimensions: 768

8. **multi-qa-MiniLM-L6-cos-v1** (Overall: 0.556)
   - Semantic Similarity: 0.760
   - Clustering Quality: 0.061
   - Classification Accuracy: 0.847
   - Encoding Speed: 3431.1 sent/sec
   - Dimensions: 384

9. **msmarco-distilbert-base-v4** (Overall: 0.551)
   - Semantic Similarity: 0.782
   - Clustering Quality: 0.062
   - Classification Accuracy: 0.810
   - Encoding Speed: 1439.4 sent/sec
   - Dimensions: 768

10. **bge-m3** (Overall: 0.283)
   - Semantic Similarity: 0.848
   - Clustering Quality: 0.000
   - Classification Accuracy: 0.000
   - Encoding Speed: 210.3 sent/sec
   - Dimensions: 1024

## Key Insights

### Dimensionality Analysis
- **High-dimensional models (>500D):** bge-m3, all-mpnet-base-v2, all-distilroberta-v1, paraphrase-albert-small-v2, multi-qa-distilbert-cos-v1, msmarco-distilbert-base-v4
