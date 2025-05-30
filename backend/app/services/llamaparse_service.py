import os
import aiofiles
from tempfile import NamedTemporaryFile
from fastapi import UploadFile, HTTPException, BackgroundTasks
from typing import Optional, Dict, List, Any, Callable, AsyncIterable, Tuple
import asyncio
from functools import partial
import json
import time
import logging
import httpx
from dotenv import load_dotenv
import io
import uuid
from pathlib import Path
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

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

# Image storage configuration
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_parser():
    """Get or create a LlamaParse parser instance"""
    global _parser
    if _parser is None and LLAMAPARSE_AVAILABLE and LLAMAPARSE_API_KEY:
        _parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
            num_workers=4,  # Reduced from 8 to decrease load
            verbose=False,  # Disable verbose to reduce overhead
            language="en",
            show_progress=False,  # Disable progress display
            max_timeout=120  # Set maximum timeout to avoid endless polling
        )
    return _parser

def update_processing_status(doc_id: str, status: Dict[str, Any]):
    """Update the processing status for a document"""
    processing_status[doc_id] = status

def get_processing_status(doc_id: str) -> Dict[str, Any]:
    """Get the current processing status for a document"""
    return processing_status.get(doc_id, {"status": "unknown"})

async def should_use_fast_processing(file_content: bytes, filename: str) -> bool:
    """Determine if we should use fast processing for small documents"""
    file_size_mb = len(file_content) / (1024 * 1024)
    
    # Use fast processing for files smaller than 5MB
    if file_size_mb < 5.0:
        logger.info(f"Using fast processing for {filename} ({file_size_mb:.2f}MB)")
        return True
    
    logger.info(f"Using standard processing for {filename} ({file_size_mb:.2f}MB)")
    return False

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
                "progress": 0            })
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
        
        # Check if we should use fast processing path for small files
        use_fast = await should_use_fast_processing(file_content, file.filename)
        if use_fast:
            # For small files, try direct fallback processing first (faster)
            try:
                fallback_content = await parse_pdf_fallback(file_content, file.filename)
                update_processing_status(doc_id, {
                    "status": "completed",
                    "filename": file.filename,
                    "message": "PDF parsing completed with fast processing.",
                    "progress": 100
                })
                return fallback_content
            except Exception as e:
                logger.warning(f"Fast processing failed, falling back to LlamaParse: {e}")
        
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
                    "progress": 30 + (attempt * 5)                })
                
                # Use a thread pool to avoid blocking the event loop with optimized timeout
                loop = asyncio.get_event_loop()
                
                # Set a reasonable timeout for parsing to avoid endless polling
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, 
                            partial(parser.parse, file_content, extra_info=file_info)
                        ),
                        timeout=90  # 90 second timeout to avoid endless polling
                    )
                except asyncio.TimeoutError:
                    raise Exception("PDF parsing timed out after 90 seconds")
                
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
    max_checks = 60  # Maximum number of status checks (2 minutes total)
    check_count = 0
    
    # Use progressive backoff: start fast, then slow down
    while check_count < max_checks:
        current_status = get_processing_status(doc_id)
        
        # Only yield if status changed or we've reached completion
        if current_status != prev_status:
            yield json.dumps(current_status)
            prev_status = current_status.copy()
            
            # Stop streaming if completed or error
            if current_status.get("status") in ["completed", "error"]:
                break
        
        check_count += 1
        
        # Progressive backoff: 0.5s -> 1s -> 2s -> 3s (max)
        if check_count <= 10:
            sleep_time = 0.5  # First 10 checks: 0.5s
        elif check_count <= 20:
            sleep_time = 1.0  # Next 10 checks: 1s  
        elif check_count <= 40:
            sleep_time = 2.0  # Next 20 checks: 2s
        else:
            sleep_time = 3.0  # Remaining checks: 3s
            
        await asyncio.sleep(sleep_time)

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

async def parse_document_with_images(file_path: str, file_type: str, document_id: str) -> Tuple[Optional[str], List[str]]:
    """
    Enhanced document parsing that extracts both text and images using LlamaParse.
    
    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, txt)
        document_id: Unique identifier for the document
    
    Returns:
        Tuple of (parsed_text_content, list_of_saved_image_paths)
    """
    saved_image_paths = []
    
    try:
        # For text files, just read the content (no images)
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), []
        
        # For non-PDF files or when LlamaParse is not available, use regular parsing
        if file_type != "pdf" or not LLAMAPARSE_AVAILABLE or not LLAMAPARSE_API_KEY:
            logger.info(f"Using standard parsing for {file_path} (type: {file_type})")
            text_content = await parse_document(file_path, file_type)
            return text_content, []
        
        logger.info(f"Parsing PDF with image extraction: {file_path}")
        
        # Use LlamaParse with enhanced configuration for image extraction
        parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="en"
        )
        
        # Parse the document to get the JobResult object
        # Run in executor to avoid nested async issues
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, parser.parse, file_path)
        
        if not result:
            logger.warning(f"No result from LlamaParse for {file_path}")
            return None, []
        
        # Extract text content
        text_content = ""
        try:
            # Get markdown documents - run in executor to avoid nested async
            def get_markdown_docs():
                return result.get_markdown_documents(split_by_page=True)
            
            markdown_documents = await loop.run_in_executor(None, get_markdown_docs)
            if markdown_documents:
                text_content = "\n\n".join([doc.text for doc in markdown_documents if hasattr(doc, 'text')])
                logger.info(f"Extracted {len(markdown_documents)} markdown pages")
        except Exception as e:
            logger.warning(f"Error extracting markdown: {e}, trying text documents")
            try:
                def get_text_docs():
                    return result.get_text_documents(split_by_page=False)
                
                text_documents = await loop.run_in_executor(None, get_text_docs)
                if text_documents:
                    text_content = "\n\n".join([doc.text for doc in text_documents if hasattr(doc, 'text')])
                    logger.info(f"Extracted {len(text_documents)} text documents")
            except Exception as e2:
                logger.error(f"Error extracting text documents: {e2}")
        
        # Extract and save images using LlamaParse's built-in image download
        try:
            # Create document-specific image directory
            doc_image_dir = os.path.join(IMAGES_DIR, document_id)
            os.makedirs(doc_image_dir, exist_ok=True)
            
            # Use LlamaParse's get_image_documents method to extract images
            # Run in executor to avoid nested async issues
            def extract_images():
                return result.get_image_documents(
                    include_screenshot_images=True,  # Include page screenshots
                    include_object_images=True,      # Include extracted objects (charts, diagrams)
                    image_download_dir=doc_image_dir  # Download images to our directory
                )
            
            image_documents = await loop.run_in_executor(None, extract_images)
            
            if image_documents:
                logger.info(f"LlamaParse extracted {len(image_documents)} images for document {document_id}")
                
                # Process the downloaded images
                for i, img_doc in enumerate(image_documents):
                    try:
                        # Check if the image document has a local path (already downloaded)
                        if hasattr(img_doc, 'image_path') and img_doc.image_path:
                            if os.path.exists(img_doc.image_path):
                                saved_image_paths.append(img_doc.image_path)
                                logger.info(f"Image downloaded by LlamaParse: {img_doc.image_path}")
                            else:
                                logger.warning(f"Image path reported but file not found: {img_doc.image_path}")
                        
                        # Check if image has binary data that we need to save manually
                        elif hasattr(img_doc, 'image') and img_doc.image:
                            image_filename = f"{document_id}_image_{i+1}_{uuid.uuid4().hex[:8]}.png"
                            image_path = os.path.join(doc_image_dir, image_filename)
                            
                            # Save the image data
                            with open(image_path, 'wb') as img_file:
                                if isinstance(img_doc.image, bytes):
                                    img_file.write(img_doc.image)
                                else:
                                    # If it's a PIL Image or similar, convert to bytes
                                    img_doc.image.save(img_file, format='PNG')
                            
                            saved_image_paths.append(image_path)
                            logger.info(f"Manually saved image: {image_path}")
                        
                        # Check for image_data attribute (alternative name)
                        elif hasattr(img_doc, 'image_data') and img_doc.image_data:
                            image_filename = f"{document_id}_image_{i+1}_{uuid.uuid4().hex[:8]}.png"
                            image_path = os.path.join(doc_image_dir, image_filename)
                            
                            with open(image_path, 'wb') as img_file:
                                img_file.write(img_doc.image_data)
                            
                            saved_image_paths.append(image_path)
                            logger.info(f"Saved image from image_data: {image_path}")
                        
                        else:
                            logger.warning(f"Image document {i+1} has no accessible image data or path")
                            logger.debug(f"Available attributes: {dir(img_doc)}")
                        
                    except Exception as img_error:
                        logger.warning(f"Error processing image {i+1}: {img_error}")
            
            else:
                logger.info(f"No images found in document {document_id}")
                
        except Exception as e:
            logger.warning(f"Error extracting images from {file_path}: {e}")
            # Continue without images if extraction fails
        
        # If no text was extracted, try fallback
        if not text_content and file_type == "pdf":
            logger.info("No text extracted with LlamaParse, trying fallback")
            with open(file_path, "rb") as f:
                file_content = f.read()
            fallback_text = await parse_pdf_fallback(file_content, os.path.basename(file_path))
            return fallback_text, saved_image_paths
        
        return text_content, saved_image_paths
                
    except Exception as e:
        logger.error(f"Error in enhanced parsing for {file_path}: {str(e)}")
        # Try regular parsing as fallback
        try:
            text_content = await parse_document(file_path, file_type)
            return text_content, saved_image_paths
        except Exception as fallback_error:
            logger.error(f"Fallback parsing also failed: {fallback_error}")
            return None, saved_image_paths

def get_document_images(document_id: str) -> List[str]:
    """
    Get all image paths associated with a document.
    
    Args:
        document_id: The document identifier
        
    Returns:
        List of image file paths
    """
    doc_image_dir = os.path.join(IMAGES_DIR, document_id)
    
    if not os.path.exists(doc_image_dir):
        return []
    
    image_paths = []
    try:
        for filename in os.listdir(doc_image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(doc_image_dir, filename))
        
        logger.info(f"Found {len(image_paths)} images for document {document_id}")
        return sorted(image_paths)  # Sort for consistent ordering
        
    except Exception as e:
        logger.error(f"Error listing images for document {document_id}: {e}")
        return []

def delete_document_images(document_id: str) -> bool:
    """
    Delete all images associated with a document.
    
    Args:
        document_id: The document identifier
        
    Returns:
        True if deletion was successful, False otherwise
    """
    doc_image_dir = os.path.join(IMAGES_DIR, document_id)
    
    if not os.path.exists(doc_image_dir):
        return True  # Nothing to delete
    
    try:
        import shutil
        shutil.rmtree(doc_image_dir)
        logger.info(f"Deleted image directory for document {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting images for document {document_id}: {e}")
        return False