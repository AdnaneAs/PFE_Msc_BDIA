from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Response, status
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import asyncio
import uuid
import json

from app.models import DocumentUploadResponse
from app.services.llamaparse_service import (
    parse_pdf_document, 
    process_pdf_in_background, 
    get_processing_status,
    get_processing_status_stream
)
from app.services.text_processing_service import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.vector_db_service import add_documents

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and process a document"
)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a PDF document, parse it, chunk it, generate embeddings, and store in vector DB
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Generate a unique ID for this document
        doc_id = f"{file.filename}-{uuid.uuid4()}"
        
        # Parse the document with LlamaParse
        markdown_content = await parse_pdf_document(file, doc_id=doc_id)
        
        # Reset file position for reading again if needed
        await file.seek(0)
        
        # Chunk the text
        chunks = chunk_text(markdown_content)
        
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to extract any text chunks from the document")
        
        # Generate embeddings for each chunk
        chunk_embeddings = generate_embeddings(chunks)
        
        # Create metadata for each chunk
        metadatas = [{
            "filename": file.filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "doc_id": doc_id
        } for i in range(len(chunks))]
        
        # Add documents to ChromaDB
        add_documents(
            texts=chunks,
            embeddings=chunk_embeddings,
            metadatas=metadatas
        )
        
        return DocumentUploadResponse(
            filename=file.filename,
            status="processed",
            chunks_added=len(chunks),
            doc_id=doc_id
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post(
    "/upload/async",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process a document asynchronously"
)
async def upload_document_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a PDF document and process it asynchronously in the background
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Generate a unique ID for this document
        doc_id = f"{file.filename}-{uuid.uuid4()}"
        
        # Read file content once
        file_content = await file.read()
        
        # Define the callback function for when parsing is complete
        async def process_parsed_content(markdown_content: str):
            try:
                # Chunk the text
                chunks = chunk_text(markdown_content)
                if not chunks:
                    print(f"Failed to extract chunks from {file.filename}")
                    return
                
                # Generate embeddings for each chunk
                chunk_embeddings = generate_embeddings(chunks)
                
                # Create metadata for each chunk
                metadatas = [{
                    "filename": file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": doc_id
                } for i in range(len(chunks))]
                
                # Add documents to ChromaDB
                add_documents(
                    texts=chunks,
                    embeddings=chunk_embeddings,
                    metadatas=metadatas
                )
                
                print(f"Successfully processed {file.filename} with {len(chunks)} chunks")
            except Exception as e:
                print(f"Error in background processing: {str(e)}")
        
        # Add background task
        background_tasks.add_task(
            process_pdf_in_background,
            file_content,
            file.filename,
            process_parsed_content,
            doc_id
        )
        
        return {
            "filename": file.filename,
            "status": "accepted",
            "message": "Document processing started in the background",
            "doc_id": doc_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting document processing: {str(e)}")


@router.get(
    "/status/{doc_id}",
    summary="Get processing status for a document"
)
async def get_document_status(doc_id: str):
    """
    Get the current processing status for a document
    """
    status = get_processing_status(doc_id)
    if status["status"] == "unknown":
        raise HTTPException(status_code=404, detail="Document not found or processing hasn't started")
    return status


@router.get(
    "/status/stream/{doc_id}",
    summary="Stream processing status updates for a document"
)
async def stream_document_status(doc_id: str):
    """
    Stream real-time processing status updates for a document
    """
    status = get_processing_status(doc_id)
    if status["status"] == "unknown":
        raise HTTPException(status_code=404, detail="Document not found or processing hasn't started")
    
    async def event_generator():
        try:
            # Initial status
            status = get_processing_status(doc_id)
            yield f"data: {json.dumps(status)}\n\n"
            
            # Stream updates until completed or error
            async for status_update in get_processing_status_stream(doc_id):
                yield f"data: {status_update}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"  # Disable buffering in Nginx if used
        }
    )