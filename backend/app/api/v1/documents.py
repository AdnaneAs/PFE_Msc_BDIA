from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Path, Query, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
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
from app.services.vector_db_service import get_all_vectorized_documents, get_collection, search_across_all_models, delete_from_all_models
from app.services.llamaparse_service import (
    parse_pdf_document, 
    process_pdf_in_background, 
    get_processing_status,
    get_processing_status_stream,
    get_document_images
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
        sqlite_documents_data = await get_all_documents()
        
        # Convert dictionaries to DocumentResponse objects
        sqlite_documents = []
        for doc_data in sqlite_documents_data:
            sqlite_documents.append(DocumentResponse(
                id=doc_data["id"],
                filename=doc_data["filename"],
                file_type=doc_data["file_type"],
                status=doc_data["status"],
                original_name=doc_data["original_name"],
                file_size=doc_data["file_size"],
                uploaded_at=doc_data["uploaded_at"],
                processed_at=doc_data.get("processed_at"),
                chunk_count=doc_data.get("chunk_count"),
                error_message=doc_data.get("error_message")
            ))
        
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
                # Generate a hash-based integer ID for vector-only documents
                import hashlib
                doc_id_str = vec_doc.get("doc_id", vec_doc["filename"])
                doc_id = abs(hash(doc_id_str)) % (10**9)  # Convert to positive integer
                
                sqlite_documents.append(DocumentResponse(
                    id=doc_id,
                    filename=vec_doc["filename"],
                    file_type="unknown",  # We don't store this in vector DB
                    status="vectorized",
                    original_name=vec_doc["filename"],
                    file_size=0,  # We don't store this in vector DB
                    uploaded_at=vec_doc.get("uploaded_at", ""),
                    chunk_count=vec_doc.get("chunk_count", 0)
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
              # Delete from all embedding model collections
            try:
                success = delete_from_all_models(doc_id=str(document_id))
                if success:
                    logger.info(f"Successfully deleted document {document_id} from vector database")
                else:
                    logger.warning(f"No vector database documents found with doc_id: {document_id}")
            except Exception as e:
                logger.error(f"Error deleting from vector database: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error deleting from vector database: {str(e)}")
            
            return {"message": f"Document with ID {document_id} successfully deleted from vector database"}
        
        # If document exists in SQLite, handle both databases
        filename = document.filename
        # Delete from all embedding model collections
        try:
            success = delete_from_all_models(filename=filename)
            if success:
                logger.info(f"Successfully deleted document {filename} from vector database")
            else:
                logger.warning(f"No vector database documents found with filename: {filename}")
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
            # Get initial status from database
            current_doc = await get_document_by_id(doc_id)
            if current_doc and current_doc["status"] == "completed":
                # If already completed, send completion status and close
                status = {
                    "status": "completed",
                    "progress": 100,
                    "message": "Document processing completed"
                }
                yield f"data: {json.dumps(status)}\n\n"
                return
            
            # For processing documents, send periodic updates but with backoff
            last_status = None
            check_count = 0
            max_checks = 40  # Maximum 2 minutes of polling
            
            while check_count < max_checks:
                current_doc = await get_document_by_id(doc_id)
                status = {
                    "status": current_doc["status"] if current_doc else "unknown",
                    "progress": 100 if current_doc and current_doc["status"] == "completed" else min(50 + check_count * 2, 90),
                    "message": f"Document {current_doc['status']}" if current_doc else "Processing..."
                }
                
                # Only send if status changed or it's the first check
                if status != last_status or check_count == 0:
                    yield f"data: {json.dumps(status)}\n\n"
                    last_status = status
                
                # Exit if completed or error
                if current_doc and current_doc["status"] in ["completed", "error"]:
                    break
                    
                check_count += 1
                
                # Progressive backoff similar to llamaparse_service
                if check_count <= 5:
                    sleep_time = 1.0   # First 5 checks: 1s
                elif check_count <= 15:
                    sleep_time = 2.0   # Next 10 checks: 2s
                else:
                    sleep_time = 3.0   # Remaining checks: 3s
                    
                await asyncio.sleep(sleep_time)
            
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

@router.get("/{doc_id}/images")
async def get_document_images_endpoint(doc_id: str):
    """
    Get list of images extracted from a document.
    
    Args:
        doc_id: The document ID
        
    Returns:
        List of image metadata including filenames and paths
    """
    try:
        # Check if document exists in database
        doc = await get_document_by_id(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get image paths for the document
        image_paths = get_document_images(doc_id)
        
        # Convert paths to relative URLs and metadata
        images = []
        for path in image_paths:
            filename = os.path.basename(path)
            images.append({
                "filename": filename,
                "path": path,
                "url": f"/api/documents/{doc_id}/images/{filename}",
                "size": os.path.getsize(path) if os.path.exists(path) else 0
            })
        
        return {
            "document_id": doc_id,
            "document_filename": doc["filename"],
            "image_count": len(images),
            "images": images
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting images for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document images: {str(e)}")

@router.get("/{doc_id}/images/{image_filename}")
async def get_document_image_file(doc_id: str, image_filename: str):
    """
    Serve a specific image file from a document.
    
    Args:
        doc_id: The document ID
        image_filename: The image filename
        
    Returns:
        The image file
    """
    try:
        # Check if document exists in database
        doc = await get_document_by_id(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get image paths for the document
        image_paths = get_document_images(doc_id)
        
        # Find the specific image
        image_path = None
        for path in image_paths:
            if os.path.basename(path) == image_filename:
                image_path = path
                break
        
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Determine media type based on file extension
        ext = os.path.splitext(image_filename)[1].lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        media_type = media_type_map.get(ext, 'application/octet-stream')
        
        # Return the image file
        return FileResponse(
            path=image_path,
            media_type=media_type,
            filename=image_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {image_filename} for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@router.get(
    "/{doc_id}/chunks",
    summary="Get all chunks for a document"
)
async def get_document_chunks(
    doc_id: str = Path(..., description="The document ID")
):
    """
    Retrieve all chunks (actual content) for a specific document.
    This is used by the frontend modal to display source content.
    """
    try:
        # Search across all embedding model collections
        all_results = search_across_all_models(doc_id=str(doc_id))
        
        if not all_results:
            raise HTTPException(status_code=404, detail=f"No chunks found for document ID {doc_id}")
        
        # Combine results from all collections
        all_chunks = []
        for collection_name, results in all_results.items():
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])
            
            for doc_text, metadata, chunk_id in zip(documents, metadatas, ids):
                chunk = {
                    "chunk_id": chunk_id,
                    "content": doc_text,
                    "metadata": metadata,
                    "doc_id": doc_id,
                    "collection": collection_name  # Track which model collection this came from
                }
                all_chunks.append(chunk)
        
        # Sort chunks by chunk_index if available
        all_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        
        return {"chunks": all_chunks, "total_chunks": len(all_chunks)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunks for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document chunks: {str(e)}")

@router.get(
    "/{doc_id}/chunks/{chunk_index}",
    summary="Get specific chunk content for a document"
)
async def get_document_chunk(
    doc_id: str = Path(..., description="The document ID"),
    chunk_index: int = Path(..., description="The chunk index")
):
    """
    Retrieve a specific chunk's content for a document.
    """
    try:
        # Search across all embedding model collections
        all_results = search_across_all_models(doc_id=str(doc_id))
        
        if not all_results:
            raise HTTPException(
                status_code=404, 
                detail=f"No chunks found for document ID {doc_id}"
            )
        
        # Find the specific chunk by chunk_index across all collections
        target_chunk = None
        for collection_name, results in all_results.items():
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])
            
            for doc_text, metadata, chunk_id in zip(documents, metadatas, ids):
                if metadata.get("chunk_index") == chunk_index:
                    target_chunk = {
                        "chunk_id": chunk_id,
                        "content": doc_text,
                        "metadata": metadata,
                        "chunk_index": chunk_index,
                        "doc_id": doc_id,
                        "collection": collection_name  # Track which model collection this came from
                    }
                    break
            
            if target_chunk:
                break
        
        if target_chunk:
            return target_chunk
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Chunk {chunk_index} not found for document ID {doc_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunk {chunk_index} for document {doc_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving document chunk: {str(e)}"
        )