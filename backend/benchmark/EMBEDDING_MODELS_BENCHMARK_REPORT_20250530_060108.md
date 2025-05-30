# Embedding Models Benchmark Report

**Generated:** 2025-05-30 06:06:55

**Models Tested:** 10

## Executive Summary

- **Best Semantic Similarity:** e5-small-v2 (0.849)
- **Best Clustering Quality:** paraphrase-albert-small-v2 (0.071)
- **Best Classification Transfer:** all-mpnet-base-v2 (0.883)
- **Most Efficient:** all-MiniLM-L6-v2 (3537.2 sent/sec)

## Detailed Results

### Models Ranked by Overall Performance

1. **all-mpnet-base-v2** (Overall: 0.587)
   - Semantic Similarity: 0.834
   - Clustering Quality: 0.045
   - Classification Accuracy: 0.883
   - Encoding Speed: 588.3 sent/sec
   - Dimensions: 768

2. **e5-small-v2** (Overall: 0.585)
   - Semantic Similarity: 0.849
   - Clustering Quality: 0.060
   - Classification Accuracy: 0.847
   - Encoding Speed: 1924.6 sent/sec
   - Dimensions: 384

3. **paraphrase-MiniLM-L6-v2** (Overall: 0.580)
   - Semantic Similarity: 0.841
   - Clustering Quality: 0.067
   - Classification Accuracy: 0.830
   - Encoding Speed: 3352.8 sent/sec
   - Dimensions: 384

4. **paraphrase-albert-small-v2** (Overall: 0.578)
   - Semantic Similarity: 0.834
   - Clustering Quality: 0.071
   - Classification Accuracy: 0.830
   - Encoding Speed: 1185.8 sent/sec
   - Dimensions: 768

5. **all-MiniLM-L6-v2** (Overall: 0.577)
   - Semantic Similarity: 0.820
   - Clustering Quality: 0.054
   - Classification Accuracy: 0.857
   - Encoding Speed: 3537.2 sent/sec
   - Dimensions: 384

6. **all-distilroberta-v1** (Overall: 0.574)
   - Semantic Similarity: 0.825
   - Clustering Quality: 0.045
   - Classification Accuracy: 0.853
   - Encoding Speed: 1371.7 sent/sec
   - Dimensions: 768

7. **msmarco-distilbert-base-v4** (Overall: 0.558)
   - Semantic Similarity: 0.782
   - Clustering Quality: 0.057
   - Classification Accuracy: 0.833
   - Encoding Speed: 1472.2 sent/sec
   - Dimensions: 768

8. **multi-qa-MiniLM-L6-cos-v1** (Overall: 0.553)
   - Semantic Similarity: 0.760
   - Clustering Quality: 0.061
   - Classification Accuracy: 0.837
   - Encoding Speed: 3358.6 sent/sec
   - Dimensions: 384

9. **multi-qa-distilbert-cos-v1** (Overall: 0.549)
   - Semantic Similarity: 0.747
   - Clustering Quality: 0.048
   - Classification Accuracy: 0.853
   - Encoding Speed: 1228.3 sent/sec
   - Dimensions: 768

10. **bge-m3** (Overall: 0.283)
   - Semantic Similarity: 0.848
   - Clustering Quality: 0.000
   - Classification Accuracy: 0.000
   - Encoding Speed: 214.0 sent/sec
   - Dimensions: 1024

## Key Insights

### Dimensionality Analysis
- **High-dimensional models (>500D):** bge-m3, all-mpnet-base-v2, all-distilroberta-v1, paraphrase-albert-small-v2, multi-qa-distilbert-cos-v1, msmarco-distilbert-base-v4
