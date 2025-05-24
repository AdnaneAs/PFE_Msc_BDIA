from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import documents, query

# Create FastAPI app instance
app = FastAPI(
    title="Audit Report Generation Platform",
    description="Multi-Modal RAG System for Audit Report Generation",
    version="0.1.0",
)

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "Accept", "Content-Length", "Connection"],  # Important for SSE and streaming
)

# Simple API endpoint to test backend-frontend communication
@app.get("/api/hello")
async def hello():
    return {"message": "Backend is running!"}

# Include API routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])

# Add startup event to initialize services
@app.on_event("startup")
async def startup_event():
    # This is where we can initialize services that need to be started when the app starts
    # For example, we could initialize the embedding model here
    pass