from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app instance
app = FastAPI(
    title="Audit Report Generation Platform",
    description="Multi-Modal RAG System for Audit Report Generation",
    version="0.0.1",
)

# Configure CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API endpoint to test backend-frontend communication
@app.get("/api/hello")
async def hello():
    return {"message": "Hello from FastAPI!"} 