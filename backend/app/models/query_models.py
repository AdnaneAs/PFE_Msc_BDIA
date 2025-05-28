from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    config_for_model: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration for the language model (provider, model name, API key, etc.)"
    )
    search_strategy: Optional[str] = Field(
        "semantic",
        description="Search strategy: 'semantic', 'hybrid', or 'keyword'"
    )
    max_sources: Optional[int] = Field(
        5,
        description="Maximum number of source documents to retrieve"
    )

class StreamingQueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    config_for_model: Optional[Dict[str, Any]] = Field(
        None, 
        description="Configuration for the language model (provider, model name, API key, etc.)"
    )
    search_strategy: Optional[str] = Field(
        "semantic",
        description="Search strategy: 'semantic', 'hybrid', or 'keyword'"
    )
    max_sources: Optional[int] = Field(
        5,
        description="Maximum number of source documents to retrieve"
    )

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer generated from the LLM")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used to generate the answer")
    query_time_ms: int = Field(..., description="Total query time in milliseconds")
    retrieval_time_ms: Optional[int] = Field(None, description="Time taken to retrieve documents in milliseconds")
    llm_time_ms: Optional[int] = Field(None, description="Time taken by the LLM in milliseconds")
    num_sources: int = Field(..., description="Number of source documents used")
    model: Optional[str] = Field(None, description="Information about the model used")
    search_strategy: Optional[str] = Field(None, description="Search strategy used")
    average_relevance: Optional[float] = Field(None, description="Average relevance score of sources")
    top_relevance: Optional[float] = Field(None, description="Highest relevance score")

class LLMStatusResponse(BaseModel):
    is_processing: bool = Field(..., description="Whether the LLM is currently processing a request")
    last_model_used: Optional[str] = Field(None, description="The last model used")
    last_query_time: Optional[float] = Field(None, description="Unix timestamp of the last query")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    cache_size: Optional[int] = Field(None, description="Size of the response cache")
    time_since_last_query: Optional[float] = Field(None, description="Time since the last query in seconds")

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider (ollama, openai, gemini, etc.)")
    description: Optional[str] = Field(None, description="Model description")

class ModelsResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="List of available models")