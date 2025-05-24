from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from app.models import QueryRequest, QueryResponse, LLMStatusResponse, StreamingQueryRequest
from app.services.embedding_service import generate_embedding
from app.services.vector_db_service import query_documents
from app.services.llm_service import get_answer_from_llm, get_answer_from_llm_async, get_llm_status, get_streaming_response, get_available_models
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
    try:
        start_time = time.time()
        
        # Log the request with model configuration
        logger.info(f"Query request: {request.question}")
        if request.model_config:
            logger.info(f"Model config: {request.model_config}")
        
        # Generate embedding for the question
        query_embedding = generate_embedding(request.question)
        
        # Query ChromaDB for relevant documents
        query_results = query_documents(query_embedding)
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        
        retrieval_time = time.time() - start_time
        
        if not documents:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question. Please try a different question or upload more documents.",
                sources=[],
                query_time_ms=int(retrieval_time * 1000),
                num_sources=0
            )
            
        # Get answer from LLM using the specified model configuration
        llm_start_time = time.time()
        answer, model_info = get_answer_from_llm(request.question, documents, request.model_config)
        llm_time = time.time() - llm_start_time
        
        total_time = time.time() - start_time
        
        # Return response with answer, sources, and timing information
        return QueryResponse(
            answer=answer,
            sources=metadatas,
            query_time_ms=int(total_time * 1000),
            retrieval_time_ms=int(retrieval_time * 1000),
            llm_time_ms=int(llm_time * 1000),
            num_sources=len(documents),
            model=model_info
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post(
    "/stream",
    summary="Stream a response to a query"
)
async def stream_query(request: QueryRequest):
    """
    Query the documents and stream the response from the LLM as it's generated
    """
    try:
        start_time = time.time()
        
        # Log the streaming request
        logger.info(f"Streaming query request: {request.question}")
        if request.model_config:
            logger.info(f"Model config: {request.model_config}")
        
        # Generate embedding for the question
        query_embedding = generate_embedding(request.question)
        
        # Query ChromaDB for relevant documents
        query_results = query_documents(query_embedding)
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        
        retrieval_time = time.time() - start_time
        
        if not documents:
            error_response = {
                "error": True,
                "message": "No relevant documents found to answer your question."
            }
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
                # First send the metadata about the query
                metadata = {
                    "metadata": True,
                    "num_sources": len(documents),
                    "sources": metadatas,
                    "retrieval_time_ms": int(retrieval_time * 1000)
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                
                # Stream the response from the LLM with model configuration
                async for token in get_streaming_response(request.question, documents, request.model_config):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Send completion message with timing
                total_time = time.time() - start_time
                completion = {
                    "complete": True,
                    "query_time_ms": int(total_time * 1000),
                    "final_token": ""  # Include an empty final token to avoid missing the last chunk
                }
                yield f"data: {json.dumps(completion)}\n\n"
                
            except Exception as e:
                logger.error(f"Error streaming response: {str(e)}", exc_info=True)
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
        logger.error(f"Error setting up streaming: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


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