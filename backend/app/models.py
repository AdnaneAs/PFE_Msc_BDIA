from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Model for a query request
    """
    question: str = Field(..., description="The question to ask")


class StreamingQueryRequest(BaseModel):
    """
    Model for a streaming query request
    """
    question: str = Field(..., description="The question to ask")
    stream_tokens: bool = Field(True, description="Whether to stream tokens or full response")


class QueryResponse(BaseModel):
    """
    Model for a query response
    """
    answer: str = Field(..., description="The answer from the LLM")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents that were used to answer the question")
    query_time_ms: Optional[int] = Field(None, description="Total query processing time in milliseconds")
    retrieval_time_ms: Optional[int] = Field(None, description="Time taken to retrieve relevant documents in milliseconds")
    llm_time_ms: Optional[int] = Field(None, description="Time taken by the LLM to generate the answer in milliseconds")
    num_sources: Optional[int] = Field(None, description="Number of source documents used for the answer")


class DocumentUploadResponse(BaseModel):
    """
    Model for a document upload response
    """
    filename: str = Field(..., description="The name of the uploaded file")
    status: str = Field(..., description="The status of the upload")
    chunks_added: int = Field(..., description="The number of chunks added to the vector database")
    doc_id: Optional[str] = Field(None, description="Unique identifier for the document processing job")


class ProcessingStatus(BaseModel):
    """
    Model for document processing status
    """
    status: str = Field(..., description="Current status of the processing (starting, parsing, extracting, completed, error, etc.)")
    filename: str = Field(..., description="The name of the file being processed")
    message: str = Field(..., description="A descriptive message about the current status")
    progress: int = Field(..., description="Processing progress percentage (0-100)")
    error: Optional[str] = Field(None, description="Error message, if any")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks extracted from the document")
    parsing_time: Optional[str] = Field(None, description="Time taken to parse the document")


class LLMStatusResponse(BaseModel):
    """
    Model for LLM service status
    """
    is_processing: bool = Field(..., description="Whether the LLM is currently processing a request")
    last_model_used: Optional[str] = Field(None, description="The last model used for a query")
    last_query_time: Optional[float] = Field(None, description="Timestamp of the last query")
    time_since_last_query: Optional[float] = Field(None, description="Time in seconds since the last query")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    cache_size: int = Field(..., description="Number of cached query responses")