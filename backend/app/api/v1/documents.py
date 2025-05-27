from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Path, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import json
import pandas as pd
import io
import os
import logging
from app.models.document_models import DocumentUploadResponse, DocumentResponse, DocumentList
from app.services.document_service import handle_document_upload
from app.database.db_setup import get_all_documents, get_document_by_id, delete_document, init_db
from app.services.vector_db_service import get_all_vectorized_documents, get_collection
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
logger = logging.getLogger(__name__)
# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'doc': 'application/msword',
    'txt': 'text/plain',
    'csv': 'text/csv'
}

def is_supported_file(filename: str) -> bool:
    """Check if the file extension is supported"""
    return any(filename.lower().endswith(f'.{ext}') for ext in SUPPORTED_EXTENSIONS.keys())

async def process_csv_file(file_content: bytes, filename: str) -> str:
    """Process CSV file and store in SQLite database"""
    try:
        # Read CSV content
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Create a table name from the filename
        table_name = f"csv_{filename.replace('.', '_').replace('-', '_')}"
        
        # Store in SQLite database
        conn = await init_db()
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Generate a text representation for vector storage
        text_content = f"CSV File: {filename}\n\n"
        text_content += "Columns:\n"
        text_content += "\n".join(f"- {col}" for col in df.columns)
        text_content += "\n\nSample Data:\n"
        text_content += df.head().to_string()
        
        return text_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")

async def process_text_file(file_content: bytes) -> str:
    """Process text file content"""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid text file encoding. Please use UTF-8.")

@router.post(
    "/upload", 
    response_model=DocumentUploadResponse,
    summary="Upload a document for processing"
)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a document (PDF, DOCX, or TXT) for processing.
    
    The document will be:
    1. Validated for file type
    2. Saved to disk
    3. Processed in the background
    4. Parsed with LlamaParse
    5. Chunked and stored in the vector database
    
    Returns a document ID that can be used to check processing status.
    """
    try:
        result = await handle_document_upload(file, background_tasks)
        return DocumentUploadResponse(**result)
    except ValueError as e:
        # This is for expected errors like unsupported file types
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # This is for unexpected errors
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.get(
    "/", 
    response_model=DocumentList,
    summary="Get all documents"
)
async def get_documents():
    """
    Retrieve a list of all documents and their processing status.
    This includes both documents in the SQLite database and documents in the vector database.
    
    Returns metadata for all documents in the system, ordered by upload time (newest first).
    """
    try:
        # Get documents from SQLite database
        sqlite_documents = await get_all_documents()
        
        # Get documents from vector database
        vector_documents = get_all_vectorized_documents()
        
        # Create a set of filenames from vector database for quick lookup
        vector_filenames = {doc["filename"] for doc in vector_documents}
        
        # Update SQLite documents with vector database status
        for doc in sqlite_documents:
            if doc.filename in vector_filenames:
                doc.status = "vectorized"
        
        # Add any documents that are only in vector database
        for vec_doc in vector_documents:
            if not any(doc.filename == vec_doc["filename"] for doc in sqlite_documents):
                sqlite_documents.append(DocumentResponse(
                    id=vec_doc["doc_id"],
                    filename=vec_doc["filename"],
                    file_type="unknown",  # We don't store this in vector DB
                    status="vectorized",
                    original_name=vec_doc["filename"],
                    file_size=0,  # We don't store this in vector DB
                    uploaded_at=vec_doc["uploaded_at"],
                    chunk_count=vec_doc["chunk_count"]
                ))
        
        return DocumentList(documents=sqlite_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@router.get(
    "/{document_id}", 
    response_model=DocumentResponse,
    summary="Get document by ID"
)
async def get_document(
    document_id: int = Path(..., description="The ID of the document to retrieve")
):
    """
    Retrieve metadata for a specific document by its ID.
    """
    document = await get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
    return document

@router.delete(
    "/{document_id}",
    summary="Delete a document"
)
async def remove_document(
    document_id: int = Path(..., description="The ID of the document to delete")
):
    """
    Delete a document from both SQLite and vector databases.
    Also removes any associated CSV tables if the document was a CSV file.
    """
    try:
        # First try to get the document from SQLite
        document = await get_document_by_id(document_id)
        
        # If not in SQLite, check vector database
        if not document:
            # Get all vectorized documents
            vector_docs = get_all_vectorized_documents()
            vector_doc = next((doc for doc in vector_docs if str(doc["doc_id"]) == str(document_id)), None)
            
            if not vector_doc:
                raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
              # Delete from vector database
            collection = get_collection()
            try:
                # First, find all matching document chunks using the doc_id
                results = collection.get(ids=list([str(document_id)]))
                ids_to_delete = results.get("ids", [])
                
                if not ids_to_delete:
                    logger.warning(f"No vector database documents found with doc_id: {document_id}")
                else:
                    # Delete by specific IDs instead of using 'where' clause
                    status = collection.delete(ids=ids_to_delete)
                    logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id} from vector database status: {status}")
            except Exception as e:
                logger.error(f"Error deleting from vector database: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error deleting from vector database: {str(e)}")
            
            return {"message": f"Document with ID {document_id} successfully deleted from vector database"}
        
        # If document exists in SQLite, handle both databases
        filename = document.filename
          # Delete from vector database if it exists there
        collection = get_collection()
        try:
            # First, find all matching document chunks using the filename
            results = collection.get(where={"filename": filename})
            ids_to_delete = results.get("ids", [])
            
            if not ids_to_delete:
                logger.warning(f"No vector database documents found with filename: {filename}")
            else:
                # Delete by specific IDs instead of using 'where' clause
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document {filename} from vector database")
        except Exception as e:
            logger.error(f"Error deleting from vector database: {str(e)}")
            # Continue with SQLite deletion even if vector DB deletion fails
        
        # If it's a CSV file, delete the associated table
        if filename.lower().endswith('.csv'):
            try:
                table_name = f"csv_{filename.replace('.', '_').replace('-', '_')}"
                conn = await init_db()
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.info(f"Deleted CSV table {table_name}")
            except Exception as e:
                logger.error(f"Error deleting CSV table: {str(e)}")
                # Continue with other deletions even if table deletion fails
        
        # Delete from SQLite
        success = await delete_document(document_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete document with ID {document_id}")
        
        return {"message": f"Document with ID {document_id} successfully deleted from all databases"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

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
    Upload a document and process it asynchronously in the background.
    Supports PDF, DOCX, DOC, TXT, and CSV files.
    """
    if not is_supported_file(file.filename):
        supported_types = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types are: {supported_types}"
        )
    
    try:
        # Generate a unique ID for this document
        doc_id = f"{file.filename}-{uuid.uuid4()}"
        
        # Read file content once
        file_content = await file.read()
        
        # Define the callback function for when parsing is complete
        async def process_parsed_content(content: str):
            try:
                # Chunk the text
                chunks = chunk_text(content)
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
                    "doc_id": doc_id,
                    "file_type": file.filename.split('.')[-1].lower()
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
        
        # Process file based on type
        file_ext = file.filename.split('.')[-1].lower()
        
        if file_ext == 'csv':
            # Process CSV file
            content = await process_csv_file(file_content, file.filename)
            await process_parsed_content(content)
        elif file_ext == 'txt':
            # Process text file
            content = await process_text_file(file_content)
            await process_parsed_content(content)
        else:
            # Process PDF/DOCX/DOC using LlamaParse
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
async def get_document_status(doc_id: int):
    """
    Get the current processing status for a document
    """
    # First check in-memory status for active processing
    status = get_processing_status(str(doc_id))
    
    # If not found in memory, check database for completed/failed documents
    if status["status"] == "unknown":
        try:
            doc = await get_document_by_id(doc_id)
            if doc:
                # Convert database status to the expected format
                status = {
                    "status": doc["status"],
                    "filename": doc["original_name"] or doc["filename"],
                    "message": doc["error_message"] if doc["status"] == "failed" else f"Document {doc['status']}",
                    "progress": 100 if doc["status"] == "completed" else 0,
                    "doc_id": doc["id"]
                }
                if doc["status"] == "completed":
                    status["chunk_count"] = doc["chunk_count"]
                    status["processed_at"] = doc["processed_at"]
            else:
                raise HTTPException(status_code=404, detail="Document not found")
        except Exception as e:
            raise HTTPException(status_code=404, detail="Document not found or processing hasn't started")
    
    return status

@router.get(
    "/status/stream/{doc_id}",
    summary="Stream processing status updates for a document"
)
async def stream_document_status(doc_id: int):
    """
    Stream real-time processing status updates for a document
    """
    # Check if document exists in database
    doc = await get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    async def event_generator():
        try:
            # Get status from database
            current_doc = await get_document_by_id(doc_id)
            status = {
                "status": current_doc["status"] if current_doc else "unknown",
                "progress": 100 if current_doc and current_doc["status"] == "completed" else 50,
                "message": f"Document {current_doc['status']}" if current_doc else "Unknown"
            }
            yield f"data: {json.dumps(status)}\n\n"
            
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