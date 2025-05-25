from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from app.models.query_models import QueryRequest, QueryResponse, LLMStatusResponse, StreamingQueryRequest
from app.services.embedding_service import generate_embedding
from app.services.vector_db_service import query_documents
from app.services.llm_service import get_answer_from_llm, get_answer_from_llm_async, get_llm_status, get_streaming_response, get_available_models
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
        
        # Query ChromaDB for relevant documents
        query_results = query_documents(query_embedding)
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        
        # Mark the end of retrieval and log metrics
        metrics.mark_retrieval_end(len(documents))
        logger.info(f"Document retrieval completed, found {len(documents)} relevant documents")
        
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
        
        # Return response with answer, sources, and timing information
        return QueryResponse(
            answer=answer,
            sources=metadatas,
            query_time_ms=final_metrics["total_time_ms"],
            retrieval_time_ms=final_metrics["retrieval_time_ms"],
            llm_time_ms=final_metrics["llm_time_ms"],
            num_sources=len(documents),
            model=model_info
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
    
    try:
        # Enhanced logging of streaming request
        logger.info(f"Received streaming query request - Question: '{request.question}'")
        if request.config_for_model:
            logger.info(f"Model configuration: {json.dumps(request.config_for_model)}")
        
        # Generate embedding for the question
        metrics.mark_retrieval_start()
        query_embedding = generate_embedding(request.question)
        
        # Query ChromaDB for relevant documents
        query_results = query_documents(query_embedding)
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        
        # Mark retrieval completion
        metrics.mark_retrieval_end(len(documents))
        logger.info(f"Document retrieval completed, found {len(documents)} relevant documents")
        
        if not documents:
            error_response = {
                "error": True,
                "message": "No relevant documents found to answer your question."
            }
            logger.warning(f"No relevant documents found for streaming query: '{request.question}'")
            
            # Track completion in metrics
            metrics.complete()
            
            return StreamingResponse(
                iter([json.dumps(error_response)]),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "X-Accel-Buffering": "no"
                }
            )
            
        # Start streaming response
        async def event_generator():
            try:
                # Log streaming start
                logger.info(f"Starting to stream response for query: '{request.question}'")
                
                # First send the metadata about the query
                metadata = {
                    "metadata": True,
                    "num_sources": len(documents),
                    "sources": metadatas,
                    "retrieval_time_ms": metrics.complete()["retrieval_time_ms"]
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                
                # Mark LLM processing start
                metrics.mark_llm_start()
                
                # Track tokens for logging
                token_count = 0
                response_buffer = ""
                
                # Stream the response from the LLM with model configuration
                async for token in get_streaming_response(request.question, documents, request.config_for_model):
                    token_count += 1
                    response_buffer += token
                    
                    # Log progress periodically
                    if token_count % 100 == 0:
                        logger.info(f"Streamed {token_count} tokens so far")
                    
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Mark LLM processing completion
                metrics.mark_llm_end("streaming", len(response_buffer))
                
                # Get final metrics
                final_metrics = metrics.complete()
                
                # Send completion message with timing
                completion = {
                    "complete": True,
                    "query_time_ms": final_metrics["total_time_ms"],
                    "final_token": ""  # Include an empty final token to avoid missing the last chunk
                }
                
                # Log completion of streaming
                logger.info(f"Completed streaming response with {token_count} tokens")
                logger.info(f"First 200 chars of response: '{response_buffer[:200]}...' (truncated)")
                
                yield f"data: {json.dumps(completion)}\n\n"
                
            except Exception as e:
                # Enhanced error logging
                logger.error(f"Error streaming response for query: '{request.question}'", exc_info=True)
                logger.error(f"Error details: {str(e)}")
                
                # Track error in metrics
                metrics.mark_error(str(e))
                
                error_msg = f"Error streaming response: {str(e)}"
                yield f"data: {json.dumps({'error': True, 'message': error_msg})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        # Enhanced error logging
        logger.error(f"Error setting up streaming for query: '{request.question}'", exc_info=True)
        logger.error(f"Error details: {str(e)}")
        
        # Track error in metrics
        metrics.mark_error(str(e))
        
        raise HTTPException(status_code=500, detail=f"Error setting up streaming response: {str(e)}")


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