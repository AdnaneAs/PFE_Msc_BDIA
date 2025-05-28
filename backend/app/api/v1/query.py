from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from app.models.query_models import QueryRequest, QueryResponse, LLMStatusResponse, StreamingQueryRequest
from app.services.embedding_service import generate_embedding
from app.services.vector_db_service import query_documents, query_documents_advanced, get_query_suggestions
from app.services.llm_service import get_answer_from_llm, get_llm_status, get_available_models
from app.services.metrics_service import QueryMetricsTracker
import time
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Query the documents with a question"
)
async def query(request: QueryRequest):
    """
    Query the documents with a question and get an answer from the LLM
    """
    # Initialize metrics tracker
    metrics = QueryMetricsTracker(request.question, request.config_for_model)
    
    try:
        # Enhanced logging of the query request
        logger.info(f"Received query request - Question: '{request.question}'")
        if request.config_for_model:
            logger.info(f"Model configuration: {json.dumps(request.config_for_model)}")
        
        # Generate embedding for the question
        metrics.mark_retrieval_start()
        query_embedding = generate_embedding(request.question)
        
        # Use advanced query processing
        query_results = query_documents_advanced(
            query_embedding=query_embedding,
            query_text=request.question,
            n_results=request.max_sources or 5,
            search_strategy=request.search_strategy or "semantic"
        )
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        relevance_scores = query_results.get("relevance_scores", [])
        search_strategy = query_results.get("search_strategy", "semantic")
        
        # Mark the end of retrieval and log metrics
        metrics.mark_retrieval_end(len(documents))
        logger.info(f"Document retrieval completed using '{search_strategy}', found {len(documents)} relevant documents")
        
        # Log relevance metrics
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            max_relevance = max(relevance_scores)
            logger.info(f"Relevance scores - Avg: {avg_relevance:.3f}, Max: {max_relevance:.3f}")
        
        if not documents:
            logger.warning(f"No relevant documents found for query: '{request.question}'")
            metrics.complete()
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question. Please try a different question or upload more documents.",
                sources=[],
                query_time_ms=metrics.complete()["total_time_ms"],
                num_sources=0
            )
            
        # Get answer from LLM using the specified model configuration
        metrics.mark_llm_start()
        answer, model_info = get_answer_from_llm(request.question, documents, request.config_for_model)
        
        # Mark the end of LLM processing and log metrics
        metrics.mark_llm_end(model_info, len(answer) if answer else 0)
        
        # Log the complete response for better tracking
        logger.info(f"Query completed - Model: {model_info}")
        
        # Get final metrics and complete the tracking
        final_metrics = metrics.complete()
        
        # Return response with enhanced metrics
        return QueryResponse(
            answer=answer,
            sources=metadatas,
            query_time_ms=final_metrics["total_time_ms"],
            retrieval_time_ms=final_metrics["retrieval_time_ms"],
            llm_time_ms=final_metrics["llm_time_ms"],
            num_sources=len(documents),
            model=model_info,
            search_strategy=search_strategy,
            # Calculate average and top relevance (scores are now 0-100)
            average_relevance=round(sum(relevance_scores) / len(relevance_scores), 1) if relevance_scores else None,
            top_relevance=round(max(relevance_scores), 1) if relevance_scores else None
        )
        
    except Exception as e:
        # Enhanced error logging
        logger.error(f"Error processing query: '{request.question}'", exc_info=True)
        logger.error(f"Error details: {str(e)}")
        
        # Track the error in metrics
        metrics.mark_error(str(e))
        
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post(
    "/stream",
    summary="Stream a response to a query"
)
async def stream_query(request: QueryRequest):
    """
    Query the documents and stream the response from the LLM as it's generated
    """
    # Initialize metrics tracker
    metrics = QueryMetricsTracker(request.question, request.config_for_model)
    
    raise HTTPException(status_code=501, detail="Streaming is no longer supported. Please use the standard query endpoint.")


@router.get(
    "/status",
    response_model=LLMStatusResponse,
    summary="Get the status of the LLM service"
)
async def get_status():
    """
    Get the current status of the LLM service
    """
    try:
        status = get_llm_status()
        return LLMStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting LLM status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting LLM status: {str(e)}")


@router.get(
    "/stream",
    summary="Stream a response to a query (GET method for EventSource)"
)
async def stream_query_get(
    question: str, 
    model_config: str = Query(None, description="JSON string with model configuration")
):
    """
    Query the documents and stream the response from the LLM as it's generated.
    This endpoint uses GET to work with EventSource.
    """
    # Parse model_config if provided
    config = {}
    if model_config:
        try:
            config = json.loads(model_config)
            logger.info(f"Parsed model config: {config}")
        except json.JSONDecodeError:
            logger.error(f"Invalid model config JSON: {model_config}")
    
    # We reuse our POST endpoint by creating a request object
    request = QueryRequest(question=question, model_config=config)
    return await stream_query(request)


@router.get(
    "/models",
    summary="Get available LLM models"
)
async def get_models():
    """
    Get a list of available LLM models
    """
    try:
        models = get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")