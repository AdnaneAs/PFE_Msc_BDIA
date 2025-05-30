from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

from app.api.v1 import documents, query, models, config
from app.database.db_setup import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Audit Report Generation Platform",
    description="Multi-Modal RAG System for Audit Report Generation",
    version="0.2.0",  # Updated version number
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
    logger.info("Hello endpoint called")
    return {"message": "Backend is running!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "message": "Backend server is running"}

# Include API routers with correct prefixes to match frontend expectations
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(models.router, tags=["models"])
app.include_router(config.router, tags=["configuration"])

# Add models endpoint for frontend compatibility
@app.get("/api/query/models") 
async def get_available_models():
    logger.info("Models endpoint called")
    return {
        "available_models": [
            "llama3.2:latest",
            "llama3.2:1b",
            "llama3.2:3b",
            "qwen2.5:latest",
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "qwen2.5:3b",
            "qwen2.5:7b"
        ],
        "default_model": "llama3.2:latest"
    }

# Add startup event to initialize services
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing services...")
        await init_db()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down services...")