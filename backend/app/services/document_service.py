import os
import logging
import tempfile
import uuid
import shutil
from datetime import datetime
from fastapi import UploadFile, BackgroundTasks
from typing import Optional, List, Dict, Any, Tuple
import asyncio

from app.database.db_setup import (
    create_document_entry,
    update_document_status,
    get_document_by_id
)
from app.services.embedding_service import generate_embedding, generate_embeddings
from app.services.vector_db_service import add_documents
from app.services.llamaparse_service import parse_document

# For CSV/Excel parsing
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Supported file types
SUPPORTED_FILE_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/csv": "csv",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx"
}

async def save_uploaded_file(file: UploadFile) -> Tuple[str, str, int]:
    """
    Save an uploaded file to disk.
    
    Args:
        file: The uploaded file
        
    Returns:
        Tuple of (saved_path, file_type, file_size)
    """
    # Generate a unique filename
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        shutil.move(temp_path, file_path)
        file_size = os.path.getsize(file_path)
        file_type = SUPPORTED_FILE_TYPES.get(file.content_type, "unknown")
        return file_path, file_type, file_size
    except Exception as e:
        logger.error(f"Error saving file {original_filename}: {str(e)}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise

async def process_document(
    doc_id: int, 
    file_path: str, 
    file_type: str
) -> bool:
    """
    Process a document asynchronously.
    
    Args:
        doc_id: The document ID in the database
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, txt)
        
    Returns:
        Success status
    """
    try:
        await update_document_status(doc_id, "processing")
        # Parse the document using LlamaParse or pandas for CSV/XLSX
        if file_type in ("csv", "xls", "xlsx"):
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                parsed_content = df.to_csv(index=False)
            except Exception as e:
                await update_document_status(
                    doc_id, "failed", error_message=f"Failed to parse spreadsheet: {str(e)}"
                )
                return False
        else:
            parsed_content = await parse_document(file_path, file_type)
        if not parsed_content:
            await update_document_status(
                doc_id, "failed", error_message="Failed to parse document with LlamaParse"
            )
            return False
        chunks = chunk_document(parsed_content)
        # Generate embeddings for all chunks at once
        embeddings = generate_embeddings(chunks)
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "doc_id": str(doc_id),
                "filename": os.path.basename(file_path),
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunk_metadatas.append(metadata)
        # Store in vector database
        add_documents(chunks, embeddings, chunk_metadatas)
        await update_document_status(
            doc_id, "completed", processed_at=datetime.now().isoformat(), chunk_count=len(chunks)
        )
        return True
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        await update_document_status(
            doc_id, "failed", error_message=str(e)
        )
        return False

def chunk_document(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split document text into chunks of approximately equal size.
    
    Args:
        text: The document text to chunk
        chunk_size: Target size for each chunk in characters
        
    Returns:
        List of text chunks
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

async def handle_document_upload(
    file: UploadFile, 
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Handle document upload, validate type, save, and start processing.
    
    Args:
        file: The uploaded file
        background_tasks: FastAPI background tasks
        
    Returns:
        Response data including document ID and status
    """
    if file.content_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {file.content_type}. Supported types are PDF, DOCX, TXT, CSV, XLSX.")
    file_path, file_type, file_size = await save_uploaded_file(file)
    doc_id = await create_document_entry(
        filename=os.path.basename(file_path),
        file_type=file_type,
        original_name=file.filename,
        file_size=file_size
    )
    background_tasks.add_task(
        process_document,
        doc_id,
        file_path, 
        file_type
    )
    return {
        "message": f"File received, processing started.",
        "document_id": doc_id,
        "status": "pending"
    } 