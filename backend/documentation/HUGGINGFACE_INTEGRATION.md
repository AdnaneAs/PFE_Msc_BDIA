# Hugging Face Integration

This document describes the new Hugging Face integration added to the RAG system.

## Overview

The system now supports Hugging Face as a model provider alongside Ollama, OpenAI, and Gemini. This integration allows you to use various open-source models hosted on Hugging Face's Inference API.

## Featured Model: PleIAs/Pleias-RAG-1B

The primary addition is **PleIAs/Pleias-RAG-1B**, a specialized model for RAG (Retrieval-Augmented Generation) tasks:

- **Size**: 1.2 billion parameters
- **Type**: Small Reasoning Model  
- **Specialization**: Trained specifically for retrieval-augmented generation, search, and source summarization
- **Generation**: First generation of PleIAs specialized reasoning models
- **Ideal for**: RAG workflows, document Q&A, and source-based reasoning

## Available Models

The integration includes several Hugging Face models:

1. **PleIAs/Pleias-RAG-1B** ⭐ (Recommended for RAG)
   - 1.2B parameters specialized for RAG tasks
   - Optimized for retrieval-augmented generation and source summarization

2. **microsoft/DialoGPT-medium**
   - Conversational AI model
   - Good for dialogue-based interactions

3. **meta-llama/Llama-2-7b-chat-hf**
   - 7B parameter chat model
   - General-purpose conversational AI

4. **mistralai/Mistral-7B-Instruct-v0.1**
   - 7B parameter instruction-following model
   - Good for general tasks

## Setup Instructions

### 1. Get Hugging Face API Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token for use in the application

### 2. Configure API Key

**Option A: Environment Variable**
```bash
export HUGGINGFACE_API_KEY="your_token_here"
```

**Option B: UI Configuration**
- Select "Hugging Face" as the model provider in the web interface
- Enter your API key in the provided field

### 3. Select Model and Query

1. Choose "Hugging Face" as the model provider
2. Select your preferred model (PleIAs/Pleias-RAG-1B recommended for RAG)
3. Submit your query as usual

## Technical Implementation

### Backend Changes

- **New Function**: `query_huggingface_llm()` in `llm_service.py`
- **API Integration**: Uses Hugging Face Inference API
- **Model Management**: Added Hugging Face models to `get_available_models()`
- **Configuration**: Added `HUGGINGFACE_API_KEY` environment variable support

### Frontend Changes

- **Provider Selection**: Added "Hugging Face" radio button option
- **Model Dropdown**: Dedicated Hugging Face model selection
- **API Key Input**: Support for Hugging Face API key entry
- **Help Text**: Updated to include Hugging Face references

### Configuration Parameters

The integration uses optimized parameters for Hugging Face models:

```python
{
    "max_new_tokens": 2048,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "return_full_text": False
}
```

## Usage Examples

### Basic Query
```python
# Example model configuration
model_config = {
    "provider": "huggingface",
    "model": "PleIAs/Pleias-RAG-1B",
    "api_key": "your_hf_token"
}
```

### Error Handling

The integration includes comprehensive error handling:

- **Missing API Key**: Clear error message with setup instructions
- **Model Loading**: Handles Hugging Face model loading delays
- **API Errors**: Detailed error messages from Hugging Face API
- **Timeouts**: 60-second timeout with retry suggestions

## Benefits of PleIAs/Pleias-RAG-1B

1. **RAG Optimization**: Specifically trained for retrieval-augmented tasks
2. **Efficient Size**: 1.2B parameters provide good performance with lower resource usage
3. **Source Summarization**: Excellent at synthesizing information from multiple sources
4. **Citation Support**: Trained to work well with citation-based responses
5. **Open Source**: Fully open and transparent model weights

## Testing

Run the integration test to verify setup:

```bash
cd backend
python tests/test_huggingface_integration.py
```

This test verifies:
- Model availability in the system
- Proper error handling for missing API keys
- Correct model configuration

## Comparison with Other Providers

| Provider | Best For | Setup Complexity | Cost | Local/Cloud |
|----------|----------|------------------|------|-------------|
| Ollama | Local inference, privacy | Medium | Free | Local |
| OpenAI | General purpose, reliability | Easy | Paid per token | Cloud |
| Gemini | Google integration, multimodal | Easy | Paid per token | Cloud |
| **Hugging Face** | **Open source, RAG specialization** | **Easy** | **Free/Paid tiers** | **Cloud** |

## Troubleshooting

### Common Issues

1. **"API key not configured"**
   - Ensure HUGGINGFACE_API_KEY is set or provided via UI

2. **Model loading timeout**
   - Some models may take time to load on first use
   - Retry after a few minutes

3. **Rate limiting**
   - Free tier has usage limits
   - Consider upgrading to paid tier for higher usage

### Support

For issues specific to:
- **PleIAs model**: Check [PleIAs Hugging Face page](https://huggingface.co/PleIAs/Pleias-RAG-1B)
- **Hugging Face API**: Consult [Hugging Face documentation](https://huggingface.co/docs/api-inference/)
- **System integration**: Check application logs in `backend/logs/`

## Future Enhancements

Potential improvements for the Hugging Face integration:

1. **Streaming Support**: Add real-time response streaming
2. **Model Caching**: Cache frequently used models for faster responses  
3. **Fine-tuning**: Support for custom fine-tuned models
4. **Batch Processing**: Handle multiple queries efficiently
5. **Model Metrics**: Display model performance statistics
