from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentBase(BaseModel):
    """Base model for document data."""
    filename: str
    file_type: str
    status: str
    original_name: str
    file_size: int


class DocumentCreate(DocumentBase):
    """Model for creating a document entry."""
    pass


class DocumentUpdate(BaseModel):
    """Model for updating a document entry."""
    status: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: Optional[str] = None
    chunk_count: Optional[int] = None


class DocumentResponse(DocumentBase):
    """Model for document response data."""
    id: int
    error_message: Optional[str] = None
    uploaded_at: str
    processed_at: Optional[str] = None
    chunk_count: Optional[int] = None
    
    class Config:
        orm_mode = True


class DocumentList(BaseModel):
    """Model for list of documents."""
    documents: List[DocumentResponse]
    
    
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    document_id: int
    status: str = "pending"


class DocumentMetadata(BaseModel):
    filename: str = Field(..., description="Original filename")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    total_chunks: int = Field(..., description="Total number of chunks in the document")


class UploadResponse(BaseModel):
    filename: str = Field(..., description="Original filename")
    chunks_added: int = Field(..., description="Number of chunks added to the vector store")
    status: str = Field("processed", description="Status of the upload")


class AsyncUploadResponse(BaseModel):
    doc_id: str = Field(..., description="ID of the document being processed")
    filename: str = Field(..., description="Original filename")
    status: str = Field("processing", description="Status of the upload")


class DocumentStatusResponse(BaseModel):
    doc_id: str = Field(..., description="ID of the document being processed")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Status of the processing")
    message: str = Field(..., description="Status message")
    progress: int = Field(..., description="Progress percentage")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks processed")


class DocumentUploadResponse(BaseModel):
    filename: str = Field(..., description="Original filename")
    chunks_added: int = Field(..., description="Number of chunks added to the vector store")
    status: str = Field("processed", description="Status of the upload")
    doc_id: str = Field(..., description="ID of the document") 