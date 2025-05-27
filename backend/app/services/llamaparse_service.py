import os
import aiofiles
from tempfile import NamedTemporaryFile
from fastapi import UploadFile, HTTPException, BackgroundTasks
from typing import Optional, Dict, List, Any, Callable, AsyncIterable
import asyncio
from functools import partial
import json
import time
import logging
import httpx
from dotenv import load_dotenv
import io

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from app.config import LLAMAPARSE_API_KEY

# Import the LlamaParse client
try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
    logger.info("LlamaParse imported successfully from llama_parse")
except ImportError:
    try:
        from llama_cloud_services import LlamaParse
        LLAMAPARSE_AVAILABLE = True
        logger.info("LlamaParse imported successfully from llama_cloud_services")
    except ImportError:
        LLAMAPARSE_AVAILABLE = False
        logger.warning("LlamaParse is not available. Neither llama_parse nor llama_cloud_services could be imported.")

# Import fallback PDF processing libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 is not available for fallback PDF processing")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf is not available for fallback PDF processing")

# Global parser instance to avoid recreating it for each request
_parser = None

# Progress tracking dict to store processing status
processing_status = {}

# Configuration
# LlamaParse configuration
LLAMAPARSE_API_URL = os.getenv("LLAMAPARSE_API_URL", "https://api.cloud.llamaindex.ai/api/v1/parsing/upload")

def get_parser():
    """Get or create a LlamaParse parser instance"""
    global _parser
    if _parser is None and LLAMAPARSE_AVAILABLE and LLAMAPARSE_API_KEY:
        _parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
            num_workers=8,  # Increased for better parallel processing
            verbose=False,  # Disable verbose to reduce overhead
            language="en"
        )
    return _parser

def update_processing_status(doc_id: str, status: Dict[str, Any]):
    """Update the processing status for a document"""
    processing_status[doc_id] = status

def get_processing_status(doc_id: str) -> Dict[str, Any]:
    """Get the current processing status for a document"""
    return processing_status.get(doc_id, {"status": "unknown"})

async def parse_document(file_path: str, file_type: str) -> Optional[str]:
    """
    Parse a document using LlamaParse API or fallback methods.
    
    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, txt)
    
    Returns:
        Parsed text content or None if parsing failed
    """
    try:
        # Different parsing strategy based on file type
        if file_type == "txt":
            # For text files, just read the content
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # For PDF files, try fallback first if LlamaParse is not available
        if file_type == "pdf" and (not LLAMAPARSE_AVAILABLE or not LLAMAPARSE_API_KEY):
            logger.info(f"Using fallback PDF processing for {file_path}")
            with open(file_path, "rb") as f:
                file_content = f.read()
            return await parse_pdf_fallback(file_content, os.path.basename(file_path))
        
        # For PDF and DOCX, try LlamaParse API first
        if not LLAMAPARSE_API_KEY:
            logger.error("LLAMAPARSE_API_KEY is not set in environment variables")
            if file_type == "pdf":
                # Try fallback for PDF
                with open(file_path, "rb") as f:
                    file_content = f.read()
                return await parse_pdf_fallback(file_content, os.path.basename(file_path))
            return None
        
        logger.info(f"Parsing {file_type} document: {file_path}")
        
        # Use the official LlamaParse library
        parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
            verbose=True
        )
        
        # Parse the document
        documents = parser.load_data(file_path)
        
        # Extract text from the parsed documents
        if documents:
            # Combine all document texts
            text_content = "\n\n".join([doc.text for doc in documents if hasattr(doc, 'text')])
            return text_content
        else:
            logger.warning(f"No content extracted from {file_path}")
            # Try fallback for PDF files
            if file_type == "pdf":
                logger.info("Trying fallback PDF processing due to empty result")
                with open(file_path, "rb") as f:
                    file_content = f.read()
                return await parse_pdf_fallback(file_content, os.path.basename(file_path))
            return None
                
    except Exception as e:
        logger.error(f"Error parsing document {file_path}: {str(e)}")
        # Try fallback for PDF files
        if file_type == "pdf":
            try:
                logger.info("Trying fallback PDF processing after error")
                with open(file_path, "rb") as f:
                    file_content = f.read()
                return await parse_pdf_fallback(file_content, os.path.basename(file_path))
            except Exception as fallback_error:
                logger.error(f"Fallback PDF processing also failed: {str(fallback_error)}")
        return None

async def parse_pdf_document(file: UploadFile, doc_id: str) -> str:
    """
    Parse a PDF document using LlamaParse API
    
    Args:
        file: The uploaded PDF file
        
    Returns:
        str: The extracted text content in markdown format
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if not LLAMAPARSE_AVAILABLE or not LLAMAPARSE_API_KEY:
        logger.info("LlamaParse not available, using fallback PDF processing")
        update_processing_status(doc_id, {
            "status": "fallback",
            "filename": file.filename,
            "message": "Using fallback PDF processing...",
            "progress": 30
        })
        
        try:
            # Use fallback processing directly
            fallback_content = await parse_pdf_fallback(file_content, file.filename)
            
            update_processing_status(doc_id, {
                "status": "completed",
                "filename": file.filename,
                "message": "PDF parsing completed using fallback method.",
                "progress": 100
            })
            
            return fallback_content
            
        except Exception as e:
            update_processing_status(doc_id, {
                "status": "error",
                "filename": file.filename,
                "message": f"Fallback PDF processing failed: {str(e)}",
                "progress": 0
            })
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    update_processing_status(doc_id, {
        "status": "starting",
        "filename": file.filename,
        "message": "Starting PDF parsing...",
        "progress": 0
    })
    
    try:
        # Get file content directly, avoid temporary file if possible
        update_processing_status(doc_id, {
            "status": "reading",
            "filename": file.filename,
            "message": "Reading file content...",
            "progress": 10
        })
        file_content = await file.read()
        file_info = {"file_name": file.filename}
        
        # Get parser instance (reuse the same instance)
        parser = get_parser()
        if not parser:
            raise HTTPException(status_code=500, detail="Failed to initialize LlamaParse parser")
        
        # Parse the document with retry mechanism
        max_retries = 3
        retry_delay = 2  # seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                update_processing_status(doc_id, {
                    "status": "parsing",
                    "filename": file.filename,
                    "message": f"Parsing PDF with LlamaParse (attempt {attempt + 1}/{max_retries})...",
                    "progress": 30 + (attempt * 5)
                })
                
                # Use a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    partial(parser.parse, file_content, extra_info=file_info)
                )
                
                # If we get here, parsing was successful
                update_processing_status(doc_id, {
                    "status": "extracting",
                    "filename": file.filename,
                    "message": "Extracting markdown content...",
                    "progress": 70
                })
                
                # Get markdown content from the result - optimize by requesting all pages at once
                try:
                    # Try to get all pages at once
                    markdown_docs = result.get_markdown_documents(split_by_page=False)
                    if not markdown_docs:
                        raise ValueError("No markdown content extracted")
                    
                    # Combine all markdown content efficiently
                    markdown_content = "\n\n".join([doc.text for doc in markdown_docs])
                    
                    update_processing_status(doc_id, {
                        "status": "completed",
                        "filename": file.filename,
                        "message": "PDF parsing completed successfully.",
                        "progress": 100,
                        "total_chunks": len(markdown_docs)
                    })
                    
                    return markdown_content
                    
                except Exception as e:
                    update_processing_status(doc_id, {
                        "status": "error",
                        "filename": file.filename,
                        "message": f"Error extracting content: {str(e)}",
                        "progress": 0
                    })
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error extracting content from the parsed PDF: {str(e)}"
                    )
                
            except Exception as e:
                last_error = e
                # If not the final attempt, retry
                if attempt < max_retries - 1:
                    update_processing_status(doc_id, {
                        "status": "retrying",
                        "filename": file.filename,
                        "message": f"Retrying after error: {str(e)}",
                        "progress": 20 + (attempt * 5)
                    })
                    # Wait before retrying (with exponential backoff)
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    # This was the last attempt with LlamaParse, try fallback
                    logger.info(f"LlamaParse failed after {max_retries} attempts, trying fallback methods")
                    update_processing_status(doc_id, {
                        "status": "fallback",
                        "filename": file.filename,
                        "message": "LlamaParse failed, trying fallback PDF processing...",
                        "progress": 50
                    })
                    
                    try:
                        # Try fallback PDF processing
                        fallback_content = await parse_pdf_fallback(file_content, file.filename)
                        
                        update_processing_status(doc_id, {
                            "status": "completed",
                            "filename": file.filename,
                            "message": "PDF parsing completed using fallback method.",
                            "progress": 100
                        })
                        
                        return fallback_content
                        
                    except Exception as fallback_error:
                        # Both LlamaParse and fallback failed
                        update_processing_status(doc_id, {
                            "status": "error",
                            "filename": file.filename,
                            "message": f"Both LlamaParse and fallback failed. LlamaParse: {str(last_error)}, Fallback: {str(fallback_error)}",
                            "progress": 0
                        })
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error processing PDF. LlamaParse failed after {max_retries} attempts: {str(last_error)}. Fallback also failed: {str(fallback_error)}"
                        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        update_processing_status(doc_id, {
            "status": "error",
            "filename": file.filename,
            "message": f"Error parsing PDF: {str(e)}",
            "progress": 0
        })
        raise HTTPException(status_code=500, detail=f"Error parsing PDF: {str(e)}")

# Background task for processing PDFs
async def process_pdf_in_background(
    file_content: bytes, 
    filename: str,
    callback: Callable[[str], Any],
    doc_id: str
):
    """Process a PDF in the background and call the callback with the result"""
    update_processing_status(doc_id, {
        "status": "starting",
        "filename": filename,
        "message": "Starting background PDF parsing...",
        "progress": 0
    })
    
    try:
        update_processing_status(doc_id, {
            "status": "parsing",
            "filename": filename,
            "message": "Parsing PDF with LlamaParse...",
            "progress": 30
        })
        
        file_info = {"file_name": filename}
        parser = get_parser()
        
        # Try to parse with retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Record start time for performance tracking
                start_time = time.time()
                
                update_processing_status(doc_id, {
                    "status": "parsing",
                    "filename": filename,
                    "message": f"Parsing PDF with LlamaParse (attempt {attempt + 1}/{max_retries})...",
                    "progress": 30 + (attempt * 5)
                })
                
                # Use synchronous parse for background task
                result = parser.parse(file_content, extra_info=file_info)
                
                # Record parsing time for debugging
                parsing_time = time.time() - start_time
                
                update_processing_status(doc_id, {
                    "status": "extracting",
                    "filename": filename,
                    "message": f"Extracting markdown content (parsed in {parsing_time:.1f}s)...",
                    "progress": 70
                })
                
                # Get markdown content from the result
                markdown_docs = result.get_markdown_documents(split_by_page=False)
                if markdown_docs:
                    markdown_content = "\n\n".join([doc.text for doc in markdown_docs])
                    
                    update_processing_status(doc_id, {
                        "status": "processing",
                        "filename": filename,
                        "message": "Processing extracted content...",
                        "progress": 85,
                        "total_chunks": len(markdown_docs),
                        "parsing_time": f"{parsing_time:.1f}s"
                    })
                    
                    await callback(markdown_content)
                    
                    update_processing_status(doc_id, {
                        "status": "completed",
                        "filename": filename,
                        "message": f"PDF processing completed in {parsing_time:.1f}s.",
                        "progress": 100,
                        "total_chunks": len(markdown_docs)
                    })
                    
                    # Break the retry loop on success
                    break
                    
                else:
                    # If we didn't get markdown content but no exception was raised
                    if attempt < max_retries - 1:
                        update_processing_status(doc_id, {
                            "status": "retrying",
                            "filename": filename,
                            "message": "No content extracted, retrying...",
                            "progress": 20 + (attempt * 5)
                        })
                        # Wait before retrying
                        await asyncio.sleep(2 * (2 ** attempt))
                    else:
                        update_processing_status(doc_id, {
                            "status": "error",
                            "filename": filename,
                            "message": "Failed to extract content after multiple attempts",
                            "progress": 0
                        })
            
            except Exception as e:
                last_error = e
                # If not the final attempt, retry
                if attempt < max_retries - 1:
                    update_processing_status(doc_id, {
                        "status": "retrying",
                        "filename": filename,
                        "message": f"Error: {str(e)}, retrying...",
                        "progress": 20 + (attempt * 5)
                    })
                    # Wait before retrying
                    await asyncio.sleep(2 * (2 ** attempt))
                else:
                    # This was the last attempt
                    update_processing_status(doc_id, {
                        "status": "error",
                        "filename": filename,
                        "message": f"Background PDF processing error after {max_retries} attempts: {str(e)}",
                        "progress": 0
                    })
                    print(f"Background PDF processing error: {str(e)}")
                    
    except Exception as e:
        update_processing_status(doc_id, {
            "status": "error",
            "filename": filename,
            "message": f"Background PDF processing error: {str(e)}",
            "progress": 0
        })
        print(f"Background PDF processing error: {str(e)}")

async def get_processing_status_stream(doc_id: str) -> AsyncIterable[str]:
    """Generate a stream of status updates for the given document ID"""
    prev_status = None
    
    # Check status more frequently for faster updates (0.3 seconds)
    while True:
        current_status = get_processing_status(doc_id)
        
        # Only yield if status changed or we've reached completion
        if current_status != prev_status:
            yield json.dumps(current_status)
            prev_status = current_status.copy()
            
            # Stop streaming if completed or error
            if current_status.get("status") in ["completed", "error"]:
                break
        
        await asyncio.sleep(0.3)  # Faster polling interval for more responsive UI updates

async def parse_pdf_fallback(file_content: bytes, filename: str) -> str:
    """
    Parse PDF using local libraries as fallback when LlamaParse is unavailable
    
    Args:
        file_content: Raw PDF file content
        filename: Name of the file for error reporting
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Using fallback PDF processing for {filename}")
    
    text_content = ""
    
    # Try pypdf first (newer library)
    if PYPDF_AVAILABLE:
        try:
            logger.info("Attempting PDF processing with pypdf library")
            pdf_file = io.BytesIO(file_content)
            reader = PdfReader(pdf_file)
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text_content += page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    
            if text_content.strip():
                logger.info(f"Successfully extracted {len(text_content)} characters using pypdf")
                return text_content
                
        except Exception as e:
            logger.warning(f"pypdf failed: {e}")
    
    # Try PyPDF2 as second fallback
    if PYPDF2_AVAILABLE:
        try:
            logger.info("Attempting PDF processing with PyPDF2 library")
            pdf_file = io.BytesIO(file_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num in range(len(reader.pages)):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text_content += page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    
            if text_content.strip():
                logger.info(f"Successfully extracted {len(text_content)} characters using PyPDF2")
                return text_content
                
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}")
    
    # If all fallback methods failed
    if not text_content.strip():
        error_msg = "Could not extract text from PDF using available fallback methods"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    return text_content