from typing import List
import re
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks of specified size with overlap
    
    Args:
        text: The text to chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Simple regex to split by paragraphs or double line breaks
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Initialize chunks
    chunks = []
    current_chunk = ""
    
    # Process each paragraph
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
            if current_chunk:  # Avoid empty chunks
                chunks.append(current_chunk.strip())
            
            # Start a new chunk with overlap
            if len(current_chunk) > CHUNK_OVERLAP:
                # Take last part of previous chunk as overlap
                words = current_chunk.split()
                overlap_words = words[-int(CHUNK_OVERLAP / 10):]  # Approximate number of words for overlap
                current_chunk = ' '.join(overlap_words)
            else:
                current_chunk = ""
                
        # Add paragraph to current chunk
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks 