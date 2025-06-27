import aiosqlite
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

# Define database path
DB_PATH = Path("data/documents.db")

# Ensure the data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# SQL to create documents table
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    uploaded_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP,
    chunk_count INTEGER,
    file_size INTEGER,
    original_name TEXT
)
"""

async def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    try:
        logger.info(f"Initializing database at {DB_PATH}")
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(CREATE_DOCUMENTS_TABLE)
            await db.commit()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

async def create_document_entry(
    filename: str, 
    file_type: str, 
    original_name: str,
    file_size: int
) -> int:
    """
    Create a new document entry in the database.
    
    Args:
        filename: The stored filename
        file_type: The document type (pdf, docx, txt)
        original_name: Original filename as uploaded
        file_size: Size of the file in bytes
    
    Returns:
        The ID of the new document
    """
    try:
        current_time = datetime.now().isoformat()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                INSERT INTO documents 
                (filename, file_type, status, uploaded_at, original_name, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename, file_type, "pending", current_time, original_name, file_size)
            )
            await db.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error creating document entry: {str(e)}")
        raise

async def update_document_status(
    doc_id: int,
    status: str,
    error_message: Optional[str] = None,
    processed_at: Optional[str] = None,
    chunk_count: Optional[int] = None
) -> bool:
    """
    Update the status of a document.
    
    Args:
        doc_id: The document ID
        status: The new status (pending, processing, completed, failed)
        error_message: Error message if status is 'failed'
        processed_at: Timestamp when processing completed
        chunk_count: Number of chunks the document was split into
    
    Returns:
        Success status
    """
    try:
        if processed_at is None and status == "completed":
            processed_at = datetime.now().isoformat()
            
        async with aiosqlite.connect(DB_PATH) as db:
            # Create base query and parameters
            query = "UPDATE documents SET status = ?"
            params = [status]
            
            # Add conditional parameters
            if error_message is not None:
                query += ", error_message = ?"
                params.append(error_message)
            
            if processed_at is not None:
                query += ", processed_at = ?"
                params.append(processed_at)
                
            if chunk_count is not None:
                query += ", chunk_count = ?"
                params.append(chunk_count)
                
            # Add WHERE clause
            query += " WHERE id = ?"
            params.append(doc_id)
            
            await db.execute(query, params)
            await db.commit()
            return True
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}")
        return False

async def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a document by its ID."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE id = ?", 
                (doc_id,)
            )
            result = await cursor.fetchone()
            
            if result:
                return dict(result)
            return None
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        return None

async def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieve all documents, sorted by upload time (newest first)."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            )
            results = await cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error retrieving all documents: {str(e)}")
        return []

async def check_duplicate_file(original_filename: str) -> Optional[Dict[str, Any]]:
    """
    Check if a file with the same original name already exists
    
    Args:
        original_filename: The original filename to check
        
    Returns:
        Document info if duplicate found, None otherwise
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE original_name = ? AND status != 'failed'", 
                (original_filename,)
            )
            result = await cursor.fetchone()
            
            if result:
                return dict(result)
            return None
    except Exception as e:
        logger.error(f"Error checking for duplicate file: {str(e)}")
        return None

async def delete_document(doc_id: int) -> bool:
    """Delete a document by its ID."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()
            return True
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False

def check_db_health() -> bool:
    """Check database health for health endpoint"""
    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False