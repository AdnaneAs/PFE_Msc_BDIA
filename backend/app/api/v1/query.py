from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from app.models.query_models import QueryRequest, QueryResponse, LLMStatusResponse, StreamingQueryRequest, DecomposedQueryResponse, SubQueryResult
from app.services.embedding_service import generate_embedding
from app.services.vector_db_service import query_documents, query_documents_advanced, get_query_suggestions
from app.services.llm_service import get_answer_from_llm, get_llm_status, get_available_models
from app.services.metrics_service import QueryMetricsTracker
from app.services.query_decomposition_service import query_decomposer
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
        
        # Generate embedding for the question using current model
        metrics.mark_retrieval_start()
        
        # Get current embedding model info
        from app.services.embedding_service import get_current_model_info
        current_model = get_current_model_info()
        model_name = current_model.get("name", "all-MiniLM-L6-v2")
        
        query_embedding = generate_embedding(request.question)
        
        # Use advanced query processing with model-specific collection and enhanced reranking
        # Import configuration for reranking defaults
        from app.config import ENABLE_RERANKING_BY_DEFAULT, DEFAULT_RERANKER_MODEL
        from app.services.settings_service import load_settings
        
        # Load user settings to check for persistent reranking preference
        user_settings = load_settings()
        
        # Apply configuration defaults if not specified, with user settings taking priority
        use_reranking = request.use_reranking
        if use_reranking is None:
            # Check user settings first, then fall back to config default
            use_reranking = user_settings.get("reranking_enabled", ENABLE_RERANKING_BY_DEFAULT)
        
        reranker_model = request.reranker_model or DEFAULT_RERANKER_MODEL
        
        query_results = query_documents_advanced(
            query_embedding=query_embedding,
            query_text=request.question,
            n_results=request.max_sources or 5,
            search_strategy=request.search_strategy or "semantic",
            model_name=model_name,
            use_reranking=use_reranking,
            reranker_model=reranker_model
        )
        
        documents = query_results["documents"]
        metadatas = query_results["metadatas"]
        relevance_scores = query_results.get("relevance_scores", [])
        search_strategy = query_results.get("search_strategy", "semantic")
        reranking_used = query_results.get("reranking_used", False)
        reranker_model_used = query_results.get("reranker_model", None)
        
        # Mark the end of retrieval and log metrics
        metrics.mark_retrieval_end(len(documents))
        reranking_info = f" with BGE reranking ({reranker_model_used})" if reranking_used else ""
        logger.info(f"Document retrieval completed using '{search_strategy}'{reranking_info}, found {len(documents)} relevant documents")
        
        # Log relevance metrics
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            max_relevance = max(relevance_scores)
            logger.info(f"Relevance scores - Avg: {avg_relevance:.3f}, Max: {max_relevance:.3f}")
        
        if not documents:
            logger.warning(f"No relevant documents found for query: '{request.question}'")
            metrics.complete()
            return QueryResponse(
                answer="I don't have any relevant information in my knowledge base to answer your question. Please try rephrasing your question or upload documents that might contain the information you're looking for.",
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
            reranking_used=reranking_used,
            reranker_model=reranker_model_used,
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


@router.post(
    "/decomposed",
    response_model=DecomposedQueryResponse,
    summary="Query with intelligent decomposition for complex questions"
)
async def query_with_decomposition(request: QueryRequest):
    """
    Enhanced query endpoint that automatically decomposes complex questions
    into sub-queries for better accuracy and comprehensive answers.
    """
    start_time = time.time()
    decomposition_time_ms = None
    synthesis_time_ms = None
    
    try:
        logger.info(f"Received decomposed query request - Question: '{request.question}'")
        logger.info(f"Decomposition enabled: {request.use_decomposition}")
        
        if request.config_for_model:
            logger.info(f"Model configuration: {json.dumps(request.config_for_model)}")
        
        # Step 1: Query Decomposition (if enabled)
        if request.use_decomposition:
            decomp_start = time.time()
            is_complex, sub_queries = await query_decomposer.decompose_query(
                request.question, 
                request.config_for_model
            )
            decomposition_time_ms = int((time.time() - decomp_start) * 1000)
            
            logger.info(f"Query decomposition completed in {decomposition_time_ms}ms")
        else:
            is_complex = False
            sub_queries = [request.question]
            logger.info("Query decomposition disabled, treating as simple query")
        
        # Step 2: Process each sub-query with enhanced reranking configuration
        # Import configuration for reranking defaults
        from app.config import ENABLE_RERANKING_BY_DEFAULT, DEFAULT_RERANKER_MODEL
        
        # Apply configuration defaults if not specified
        use_reranking = request.use_reranking if request.use_reranking is not None else ENABLE_RERANKING_BY_DEFAULT
        reranker_model = request.reranker_model or DEFAULT_RERANKER_MODEL
        
        sub_results = []
        for sub_query in sub_queries:
            sub_result = await query_decomposer.process_sub_query(
                sub_query=sub_query,
                model_config=request.config_for_model,
                max_sources=request.max_sources or 5,
                search_strategy=request.search_strategy or "semantic",
                use_reranking=use_reranking,
                reranker_model=reranker_model
            )
            
            # Convert to SubQueryResult model
            sub_query_result = SubQueryResult(
                sub_query=sub_result["sub_query"],
                answer=sub_result["answer"],
                sources=sub_result["sources"],
                num_sources=sub_result["num_sources"],
                relevance_scores=sub_result["relevance_scores"],
                processing_time_ms=sub_result["processing_time_ms"],
                model_info=sub_result.get("model_info")
            )
            sub_results.append(sub_query_result)
        
        # Step 3: Synthesize final answer (if decomposed)
        if is_complex and len(sub_results) > 1:
            synthesis_start = time.time()
            sub_results_dict = [
                {
                    "sub_query": sr.sub_query,
                    "answer": sr.answer,
                    "sources": sr.sources,
                    "num_sources": sr.num_sources,
                    "relevance_scores": sr.relevance_scores
                }
                for sr in sub_results
            ]
            
            final_answer, synthesis_model = await query_decomposer.synthesize_answers(
                request.question,
                sub_results_dict,
                request.config_for_model
            )
            synthesis_time_ms = int((time.time() - synthesis_start) * 1000)
            
            logger.info(f"Answer synthesis completed in {synthesis_time_ms}ms using {synthesis_model}")
        else:
            # For simple queries, use the single answer
            final_answer = sub_results[0].answer if sub_results else "No answer could be generated."
            synthesis_model = sub_results[0].model_info if sub_results else "unknown"
        
        # Calculate aggregate metrics
        total_query_time_ms = int((time.time() - start_time) * 1000)
        
        # Collect all unique sources
        all_sources = []
        all_relevance_scores = []
        for sub_result in sub_results:
            all_sources.extend(sub_result.sources)
            all_relevance_scores.extend(sub_result.relevance_scores)
        
        # Remove duplicate sources based on document ID
        unique_sources = []
        seen_docs = set()
        for source in all_sources:
            doc_id = source.get('document_id', str(source))
            if doc_id not in seen_docs:
                unique_sources.append(source)
                seen_docs.add(doc_id)
        
        # Calculate average relevance
        avg_relevance = round(sum(all_relevance_scores) / len(all_relevance_scores), 1) if all_relevance_scores else None
        top_relevance = round(max(all_relevance_scores), 1) if all_relevance_scores else None
        
        logger.info(f"Decomposed query completed in {total_query_time_ms}ms")
        logger.info(f"Generated {len(sub_results)} sub-queries, final answer length: {len(final_answer)}")
        
        return DecomposedQueryResponse(
            original_query=request.question,
            is_decomposed=is_complex,
            sub_queries=[sr.sub_query for sr in sub_results],
            sub_results=sub_results,
            final_answer=final_answer,
            total_query_time_ms=total_query_time_ms,
            decomposition_time_ms=decomposition_time_ms,
            synthesis_time_ms=synthesis_time_ms,
            model=synthesis_model if is_complex else (sub_results[0].model_info if sub_results else "unknown"),
            search_strategy=request.search_strategy or "semantic",
            total_sources=len(unique_sources),
            average_relevance=avg_relevance,
            top_relevance=top_relevance
        )
        
    except Exception as e:
        logger.error(f"Error in decomposed query processing: {e}")
        
        # Fallback to regular query processing
        try:
            logger.info("Falling back to regular query processing")
            regular_response = await query(request)
            
            # Convert to decomposed format
            return DecomposedQueryResponse(
                original_query=request.question,
                is_decomposed=False,
                sub_queries=[request.question],
                sub_results=[
                    SubQueryResult(
                        sub_query=request.question,
                        answer=regular_response.answer,
                        sources=regular_response.sources,
                        num_sources=regular_response.num_sources,
                        relevance_scores=[],
                        processing_time_ms=regular_response.query_time_ms,
                        model_info=regular_response.model
                    )
                ],
                final_answer=regular_response.answer,
                total_query_time_ms=regular_response.query_time_ms,
                decomposition_time_ms=None,
                synthesis_time_ms=None,
                model=regular_response.model,
                search_strategy=regular_response.search_strategy,
                total_sources=regular_response.num_sources,
                average_relevance=regular_response.average_relevance,
                top_relevance=regular_response.top_relevance
            )
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {fallback_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Query processing failed: {str(e)}. Fallback also failed: {str(fallback_error)}"
            )


@router.get(
    "/reranker/models",
    summary="Get available BGE reranker models"
)
async def get_available_reranker_models():
    """
    Get the list of available BGE reranker models and their status
    """
    try:
        # Import here to avoid circular imports
        from app.services.rerank_service import BGEReranker
        
        # List of supported BGE reranker models
        models = [
            {
                "model_id": "BAAI/bge-reranker-base",
                "name": "BGE Reranker Base",
                "description": "Fast and efficient reranker for general use",
                "size": "Small (~280MB)"
            },
            {
                "model_id": "BAAI/bge-reranker-large", 
                "name": "BGE Reranker Large",
                "description": "Higher accuracy reranker with larger model size",
                "size": "Large (~1.3GB)"
            },
            {
                "model_id": "BAAI/bge-reranker-v2-m3",
                "name": "BGE Reranker v2 M3",
                "description": "Latest multilingual reranker with improved performance",
                "size": "Medium (~560MB)"
            }
        ]
        
        # Check if reranking service is available
        try:
            reranker = BGEReranker()
            service_available = True
        except Exception as e:
            logger.warning(f"BGE reranker service not available: {e}")
            service_available = False
        
        return {
            "available_models": models,
            "service_available": service_available,
            "default_model": "BAAI/bge-reranker-base"
        }
        
    except Exception as e:
        logger.error(f"Error getting reranker models: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting reranker models: {str(e)}")


@router.post(
    "/reranker/test",
    summary="Test BGE reranking with a sample query"
)
async def test_reranking(request: QueryRequest):
    """
    Test BGE reranking functionality with a sample query to compare results
    """
    try:
        # Validate reranking parameters
        if not request.use_reranking:
            raise HTTPException(status_code=400, detail="use_reranking must be set to true for testing")
        
        logger.info(f"Testing BGE reranking for query: '{request.question}'")
        
        # Get current embedding model info
        from app.services.embedding_service import get_current_model_info
        current_model = get_current_model_info()
        model_name = current_model.get("name", "all-MiniLM-L6-v2")
        
        # Generate embedding for the question
        query_embedding = generate_embedding(request.question)
        
        # Get results WITHOUT reranking for comparison
        original_results = query_documents_advanced(
            query_embedding=query_embedding,
            query_text=request.question,
            n_results=request.max_sources or 5,
            search_strategy=request.search_strategy or "semantic",
            model_name=model_name,
            use_reranking=False
        )
        
        # Get results WITH reranking
        reranked_results = query_documents_advanced(
            query_embedding=query_embedding,
            query_text=request.question,
            n_results=request.max_sources or 5,
            search_strategy=request.search_strategy or "semantic", 
            model_name=model_name,
            use_reranking=True,
            reranker_model=request.reranker_model or "BAAI/bge-reranker-base"
        )
        
        # Compare results
        comparison = {
            "query": request.question,
            "reranker_model": request.reranker_model or "BAAI/bge-reranker-base",
            "original_results": {
                "num_documents": len(original_results["documents"]),
                "search_strategy": original_results.get("search_strategy"),
                "relevance_scores": original_results.get("relevance_scores", []),
                "avg_relevance": round(sum(original_results.get("relevance_scores", [])) / max(len(original_results.get("relevance_scores", [])), 1), 3),
                "documents": [
                    {
                        "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        "metadata": meta,
                        "relevance_score": score
                    }
                    for doc, meta, score in zip(
                        original_results["documents"],
                        original_results["metadatas"],
                        original_results.get("relevance_scores", [])
                    )
                ]
            },
            "reranked_results": {
                "num_documents": len(reranked_results["documents"]),
                "search_strategy": reranked_results.get("search_strategy"),
                "relevance_scores": reranked_results.get("relevance_scores", []),
                "avg_relevance": round(sum(reranked_results.get("relevance_scores", [])) / max(len(reranked_results.get("relevance_scores", [])), 1), 3),
                "reranking_applied": reranked_results.get("reranking_used", False),
                "documents": [
                    {
                        "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        "metadata": meta,
                        "relevance_score": score
                    }
                    for doc, meta, score in zip(
                        reranked_results["documents"],
                        reranked_results["metadatas"],
                        reranked_results.get("relevance_scores", [])
                    )
                ]
            }
        }
        
        logger.info(f"Reranking test completed. Original avg: {comparison['original_results']['avg_relevance']}, Reranked avg: {comparison['reranked_results']['avg_relevance']}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error testing reranking: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing reranking: {str(e)}")


@router.get(
    "/reranker/config",
    summary="Get BGE reranker configuration and available models"
)
async def get_reranker_config():
    """
    Get BGE reranker configuration including available models and benchmark results
    """
    try:
        from app.config import (
            ENABLE_RERANKING_BY_DEFAULT, 
            DEFAULT_RERANKER_MODEL, 
            AVAILABLE_RERANKER_MODELS,
            RERANKING_INITIAL_RETRIEVAL_MULTIPLIER,
            RERANKING_MAX_INITIAL_DOCUMENTS
        )
        
        return {
            "reranking_enabled_by_default": ENABLE_RERANKING_BY_DEFAULT,
            "default_reranker_model": DEFAULT_RERANKER_MODEL,
            "available_models": AVAILABLE_RERANKER_MODELS,
            "performance_settings": {
                "initial_retrieval_multiplier": RERANKING_INITIAL_RETRIEVAL_MULTIPLIER,
                "max_initial_documents": RERANKING_MAX_INITIAL_DOCUMENTS
            },
            "benchmark_summary": {
                "base_model_improvements": {
                    "map_improvement_percent": 23.86,
                    "precision_at_5_improvement_percent": 23.08,
                    "ndcg_at_5_improvement_percent": 7.09
                },
                "recommended_model": "BAAI/bge-reranker-base",
                "test_dataset": "HotpotQA academic benchmark (100 samples)"
            }
        }
    except Exception as e:
        logger.error(f"Error getting reranker configuration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting reranker configuration: {str(e)}")


@router.post(
    "/reranker/toggle",
    summary="Toggle BGE reranking on/off for system"
)
async def toggle_reranking(enable: bool):
    """
    Temporarily toggle BGE reranking on/off (runtime setting, not persistent)
    """
    try:
        # This would be a runtime toggle - in a production system, 
        # you might want to store this in a database or configuration service
        import app.config as config
        config.ENABLE_RERANKING_BY_DEFAULT = enable
        
        return {
            "success": True,
            "reranking_enabled": enable,
            "message": f"BGE reranking {'enabled' if enable else 'disabled'} successfully",
            "note": "This is a runtime setting and will reset on server restart"
        }
    except Exception as e:
        logger.error(f"Error toggling reranking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error toggling reranking: {str(e)}")