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
    use_decomposition: Optional[bool] = Field(
        True,
        description="Whether to use query decomposition for complex questions"
    )
    # BGE Reranking parameters (enabled by default for +23.86% MAP improvement)
    use_reranking: Optional[bool] = Field(
        True,
        description="Whether to use BGE reranking for better document relevance scoring (recommended for +23.86% MAP improvement)"
    )
    reranker_model: Optional[str] = Field(
        "BAAI/bge-reranker-base",
        description="BGE reranker model to use: 'BAAI/bge-reranker-base', 'BAAI/bge-reranker-large', or 'BAAI/bge-reranker-v2-m3'"
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
    use_decomposition: Optional[bool] = Field(
        True,
        description="Whether to use query decomposition for complex questions"
    )
    # BGE Reranking parameters (enabled by default for +23.86% MAP improvement)
    use_reranking: Optional[bool] = Field(
        True,
        description="Whether to use BGE reranking for better document relevance scoring (recommended for +23.86% MAP improvement)"
    )
    reranker_model: Optional[str] = Field(
        "BAAI/bge-reranker-base",
        description="BGE reranker model to use: 'BAAI/bge-reranker-base', 'BAAI/bge-reranker-large', or 'BAAI/bge-reranker-v2-m3'"
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
    reranking_used: Optional[bool] = Field(None, description="Whether BGE reranking was applied")
    reranker_model: Optional[str] = Field(None, description="BGE reranker model used, if any")
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

# New models for decomposed queries
class SubQueryResult(BaseModel):
    sub_query: str = Field(..., description="The sub-query that was processed")
    answer: str = Field(..., description="Answer to the sub-query")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used for this sub-query")
    num_sources: int = Field(..., description="Number of sources used")
    relevance_scores: List[float] = Field(..., description="Relevance scores for sources")
    processing_time_ms: int = Field(..., description="Time taken to process this sub-query")
    model_info: Optional[str] = Field(None, description="Model information")

class DecomposedQueryResponse(BaseModel):
    original_query: str = Field(..., description="The original user query")
    is_decomposed: bool = Field(..., description="Whether the query was decomposed")
    sub_queries: List[str] = Field(..., description="List of sub-queries (original query if not decomposed)")
    sub_results: List[SubQueryResult] = Field(..., description="Results for each sub-query")
    final_answer: str = Field(..., description="Final synthesized answer")
    total_query_time_ms: int = Field(..., description="Total time for the entire query process")
    decomposition_time_ms: Optional[int] = Field(None, description="Time taken for decomposition")
    synthesis_time_ms: Optional[int] = Field(None, description="Time taken for answer synthesis")
    model: Optional[str] = Field(None, description="Primary model used")
    search_strategy: Optional[str] = Field(None, description="Search strategy used")
    total_sources: int = Field(..., description="Total number of unique sources across all sub-queries")
    average_relevance: Optional[float] = Field(None, description="Average relevance across all sub-queries")
    top_relevance: Optional[float] = Field(None, description="Highest relevance score across all sub-queries")