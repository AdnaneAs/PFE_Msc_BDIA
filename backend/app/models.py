from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request model for querying documents
    """
    question: str = Field(..., description="The question to ask")
    config_for_model: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration for the language model (provider, model name, API key, etc.)"
    )


class StreamingQueryRequest(BaseModel):
    """
    Request model for streaming query responses
    """
    question: str = Field(..., description="The question to ask")
    config_for_model: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration for the language model (provider, model name, API key, etc.)"
    )


class QueryResponse(BaseModel):
    """
    Response model for document queries
    """
    answer: str = Field(..., description="The answer generated from the LLM")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used to generate the answer")
    query_time_ms: int = Field(..., description="Total query time in milliseconds")
    retrieval_time_ms: Optional[int] = Field(None, description="Time taken to retrieve documents in milliseconds")
    llm_time_ms: Optional[int] = Field(None, description="Time taken by the LLM in milliseconds")
    num_sources: int = Field(..., description="Number of source documents used")
    model: Optional[str] = Field(None, description="Information about the model used")


class DocumentMetadata(BaseModel):
    """
    Metadata for a document
    """
    filename: str = Field(..., description="Original filename")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    total_chunks: int = Field(..., description="Total number of chunks in the document")


class UploadResponse(BaseModel):
    """
    Response model for document uploads
    """
    filename: str = Field(..., description="Original filename")
    chunks_added: int = Field(..., description="Number of chunks added to the vector store")
    status: str = Field("processed", description="Status of the upload")


class AsyncUploadResponse(BaseModel):
    """
    Response model for asynchronous document uploads
    """
    doc_id: str = Field(..., description="ID of the document being processed")
    filename: str = Field(..., description="Original filename")
    status: str = Field("processing", description="Status of the upload")


class DocumentStatusResponse(BaseModel):
    """
    Response model for document processing status
    """
    doc_id: str = Field(..., description="ID of the document being processed")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Status of the processing")
    message: str = Field(..., description="Status message")
    progress: int = Field(..., description="Progress percentage")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks processed")


class LLMStatusResponse(BaseModel):
    """
    Response model for LLM service status
    """
    is_processing: bool = Field(..., description="Whether the LLM is currently processing a request")
    last_model_used: Optional[str] = Field(None, description="The last model used")
    last_query_time: Optional[float] = Field(None, description="Unix timestamp of the last query")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    cache_size: Optional[int] = Field(None, description="Size of the response cache")
    time_since_last_query: Optional[float] = Field(None, description="Time since the last query in seconds")


class ModelInfo(BaseModel):
    """
    Information about an LLM model
    """
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider (ollama, openai, gemini, etc.)")
    description: Optional[str] = Field(None, description="Model description")


class ModelsResponse(BaseModel):
    """
    Response model for listing available models
    """
    models: List[ModelInfo] = Field(..., description="List of available models")


class DocumentUploadResponse(BaseModel):
    """
    Response model for document uploads with document ID
    """
    filename: str = Field(..., description="Original filename")
    chunks_added: int = Field(..., description="Number of chunks added to the vector store")
    status: str = Field("processed", description="Status of the upload")
    doc_id: str = Field(..., description="ID of the document")