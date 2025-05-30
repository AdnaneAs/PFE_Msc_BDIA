# PFE_sys Configuration System

## Overview

The PFE_sys RAG application now features a comprehensive configuration system organized into two main sections:

1. **Model Selection** - Configure AI models (LLM providers and embedding models)
2. **Search Configuration** - Configure search behavior and parameters

## Configuration Structure

### Model Selection Section

#### LLM Providers
- **Ollama (Local)** - Local LLM inference with multiple model options
  - llama3.2:latest
  - llama3.2:1b, llama3.2:3b
  - qwen2.5:latest, qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b, qwen2.5:7b
- **OpenAI** - GPT models (requires API key)
- **Google Gemini** - Gemini models (requires API key)  
- **Hugging Face** - Custom transformers (requires API key)

#### Embedding Model Selection
- **all-MiniLM-L6-v2 (Fast)**
  - Dimensions: 384
  - Size: ~22MB
  - Speed: Fast
  - Quality: Good
  - Use case: General purpose, fast retrieval

- **BAAI/bge-m3 (High Quality)**
  - Dimensions: 1024
  - Size: ~1.5GB
  - Speed: Slower
  - Quality: Excellent
  - Use case: High-quality semantic search, multilingual support

### Search Configuration Section

#### Query Decomposition (Beta)
- **Enable/Disable**: Automatically breaks down complex questions into sub-queries
- **Best for**: Multi-part questions or detailed analysis requirements
- **Default**: Disabled

#### Search Strategy
- **🔍 Hybrid Search (Recommended)**: Combines semantic understanding with keyword matching
- **🧠 Semantic Search**: Uses AI to understand meaning and context
- **🔎 Keyword Search**: Traditional text matching search
- **Default**: Hybrid

#### Max Sources
- **Options**: 3, 5, 10, 15, 20 sources
- **Purpose**: Number of relevant documents to use for generating answers
- **Default**: 5 sources

## API Endpoints

### Configuration Management

#### Get Complete Configuration
```http
GET /api/v1/config/
```
Returns the complete organized configuration structure.

#### Update Embedding Model
```http
POST /api/v1/config/embedding/model
Content-Type: application/json

{
  "model_name": "BAAI/bge-m3"
}
```

#### Update LLM Provider
```http
POST /api/v1/config/llm/provider
Content-Type: application/json

{
  "provider": "ollama"
}
```

#### Update LLM Model
```http
POST /api/v1/config/llm/model
Content-Type: application/json

{
  "model": "llama3.2:latest"
}
```

#### Update Search Strategy
```http
POST /api/v1/config/search/strategy
Content-Type: application/json

{
  "strategy": "hybrid"
}
```

#### Update Max Sources
```http
POST /api/v1/config/search/max-sources
Content-Type: application/json

{
  "max_sources": 10
}
```

#### Toggle Query Decomposition
```http
POST /api/v1/config/search/query-decomposition
Content-Type: application/json

{
  "enabled": true
}
```

## Persistent Settings

All configuration changes are automatically saved to `backend/app/data/user_settings.json` and persist across application restarts.

### Default Settings
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_provider": "ollama",
  "llm_model": "llama3.2:latest",
  "search_strategy": "hybrid",
  "max_sources": 5,
  "query_decomposition_enabled": false
}
```

## Multi-Model Architecture

### Embedding Model Separation
- Each embedding model uses its own ChromaDB collection
- **MiniLM collections**: `documents_all_minilm_l6_v2`
- **BGE-M3 collections**: `documents_baai_bge_m3`
- Avoids dimension conflicts (384-dim vs 1024-dim)

### Cross-Model Operations
- Document viewing works across all embedding collections
- Document deletion removes from all collections
- Search can be performed with any selected model

## Benefits of This Organization

1. **Logical Grouping**: Related settings are grouped together
2. **Specialty Focus**: Each section has a clear purpose
3. **User-Friendly**: Clear descriptions and intuitive organization
4. **Persistent**: Settings are automatically saved
5. **Flexible**: Easy to add new providers or models
6. **Scalable**: Architecture supports multiple embedding models simultaneously

## Future Enhancements

- Real-time provider status detection
- Model download and management
- Advanced search configuration options
- Performance monitoring per configuration
- Configuration presets/profiles
