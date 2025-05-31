# BGE Reranker Integration Guide

## Overview

This document describes the BGE (BAAI General Embedding) reranker integration in the RAG pipeline. BGE reranking provides a two-stage retrieval process that significantly improves document relevance scoring and answer quality.

## What is BGE Reranking?

BGE reranking is a technique that uses specialized cross-encoder models to re-score and reorder documents retrieved by the initial vector similarity search. This two-stage approach:

1. **Stage 1**: Retrieves candidate documents using traditional vector similarity search (retrieves 3-4x more documents than requested)
2. **Stage 2**: Uses BGE cross-encoder models to score document-query relevance and selects the top-k most relevant documents

## Available BGE Models

### 1. BAAI/bge-reranker-base (Default)
- **Size**: ~280MB
- **Speed**: Fast
- **Use Case**: General purpose reranking, production environments
- **Performance**: Good balance of speed and accuracy

### 2. BAAI/bge-reranker-large
- **Size**: ~1.3GB
- **Speed**: Slower
- **Use Case**: High-accuracy scenarios where quality is prioritized over speed
- **Performance**: Highest accuracy

### 3. BAAI/bge-reranker-v2-m3
- **Size**: ~560MB
- **Speed**: Medium
- **Use Case**: Multilingual content, latest improvements
- **Performance**: Improved multilingual support

## API Usage

### 1. Standard Query with Reranking

```json
POST /api/v1/query/
{
    "question": "What are the benefits of machine learning?",
    "use_reranking": true,
    "reranker_model": "BAAI/bge-reranker-base",
    "max_sources": 5,
    "search_strategy": "semantic"
}
```

### 2. Decomposed Query with Reranking

```json
POST /api/v1/query/decomposed
{
    "question": "Compare machine learning and deep learning approaches for image recognition",
    "use_reranking": true,
    "reranker_model": "BAAI/bge-reranker-large",
    "use_decomposition": true,
    "max_sources": 8
}
```

### 3. Get Available Reranker Models

```json
GET /api/v1/query/reranker/models
```

Response:
```json
{
    "available_models": [
        {
            "model_id": "BAAI/bge-reranker-base",
            "name": "BGE Reranker Base",
            "description": "Fast and efficient reranker for general use",
            "size": "Small (~280MB)"
        }
        // ... more models
    ],
    "service_available": true,
    "default_model": "BAAI/bge-reranker-base"
}
```

### 4. Test Reranking Functionality

```json
POST /api/v1/query/reranker/test
{
    "question": "How does machine learning work?",
    "use_reranking": true,
    "reranker_model": "BAAI/bge-reranker-base",
    "max_sources": 5
}
```

This endpoint returns a comparison between original retrieval results and reranked results.

## Response Format

When reranking is used, the response includes additional fields:

```json
{
    "answer": "Machine learning works by...",
    "sources": [...],
    "reranking_used": true,
    "reranker_model": "BAAI/bge-reranker-base",
    "search_strategy": "semantic + bge-reranking",
    "query_time_ms": 1250,
    "num_sources": 5
}
```

## Implementation Details

### Backend Components

1. **BGE Reranker Service** (`app/services/rerank_service.py`)
   - Handles model loading and cross-encoder scoring
   - Supports device detection (CUDA/CPU)
   - Provides utility functions for ChromaDB integration

2. **Enhanced Vector Database Service** (`app/services/vector_db_service.py`)
   - Modified `query_documents_advanced()` to support reranking
   - Intelligent initial retrieval scaling (3-4x documents for reranking)
   - Fallback mechanisms when reranking fails

3. **Updated API Models** (`app/models/query_models.py`)
   - Added `use_reranking` and `reranker_model` parameters
   - Enhanced response models with reranking information

4. **Enhanced Query Endpoints** (`app/api/v1/query.py`)
   - Support for reranking in all query endpoints
   - New endpoints for model information and testing
   - Comprehensive logging and error handling

### How It Works

1. **Initial Retrieval**: When `use_reranking=true`, the system retrieves 3-4x more documents than requested
2. **Cross-Encoder Scoring**: BGE model scores each document-query pair for relevance
3. **Reranking**: Documents are reordered by BGE relevance scores
4. **Selection**: Top-k documents are selected for LLM processing
5. **Response**: Enhanced response includes reranking metadata

## Performance Considerations

### Speed vs. Accuracy Trade-offs

- **BGE-base**: ~100-200ms additional latency, good accuracy improvement
- **BGE-large**: ~300-500ms additional latency, best accuracy
- **BGE-v2-m3**: ~200-300ms additional latency, multilingual support

### Resource Usage

- **Memory**: Additional 280MB-1.3GB depending on model
- **GPU**: Can utilize CUDA if available for faster inference
- **CPU**: Falls back to CPU inference if GPU unavailable

## Best Practices

### When to Use Reranking

✅ **Use reranking when:**
- Document quality is critical
- You have complex or nuanced queries
- Acceptable to trade speed for accuracy
- Working with large document collections

❌ **Avoid reranking when:**
- Ultra-low latency is required
- Simple keyword-based queries
- Very small document collections
- Limited computational resources

### Model Selection Guidelines

- **Production/General Use**: `BAAI/bge-reranker-base`
- **High Accuracy Needed**: `BAAI/bge-reranker-large`
- **Multilingual Content**: `BAAI/bge-reranker-v2-m3`

### Configuration Recommendations

```json
{
    "use_reranking": true,
    "reranker_model": "BAAI/bge-reranker-base",
    "max_sources": 8,  // Allows reranker to select from more candidates
    "search_strategy": "semantic"  // Works best with semantic search
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure `sentence-transformers` is installed
   - Check internet connectivity for model downloads
   - Verify sufficient disk space

2. **Performance Issues**
   - Use BGE-base for faster inference
   - Enable GPU if available
   - Reduce `max_sources` if needed

3. **Memory Issues**
   - Use smaller models (BGE-base)
   - Reduce batch sizes
   - Monitor memory usage

### Error Handling

The system includes robust fallback mechanisms:
- If BGE model fails to load, falls back to standard retrieval
- If reranking fails, returns original search results
- Comprehensive error logging for debugging

## Example Usage

### Python Client Example

```python
import requests

# Query with reranking
response = requests.post("http://localhost:8000/api/v1/query/", json={
    "question": "What are the latest developments in AI?",
    "use_reranking": True,
    "reranker_model": "BAAI/bge-reranker-base",
    "max_sources": 6
})

result = response.json()
print(f"Reranking used: {result['reranking_used']}")
print(f"Search strategy: {result['search_strategy']}")
```

### JavaScript/Frontend Example

```javascript
const queryWithReranking = async (question) => {
    const response = await fetch('/api/v1/query/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            use_reranking: true,
            reranker_model: 'BAAI/bge-reranker-base',
            max_sources: 5
        })
    });
    
    const result = await response.json();
    return result;
};
```

## Future Enhancements

- **Custom Model Support**: Add support for custom-trained reranker models
- **Caching**: Implement reranking result caching for repeated queries
- **Batch Processing**: Support for batch reranking of multiple queries
- **Performance Metrics**: Additional metrics for reranking effectiveness
- **A/B Testing**: Built-in comparison tools for reranking evaluation
