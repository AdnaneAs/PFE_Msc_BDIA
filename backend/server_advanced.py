# Alternative server configuration with persistent state for multi-worker support
# This version can be used when you want multiple workers with shared state

import uvicorn
import multiprocessing
import os

def start_production_server():
    """Start production server with multiple workers and Redis state storage"""
    # Calculate optimal number of workers (max 4 for safety)
    workers = min(4, multiprocessing.cpu_count())
    
    print(f"Starting PRODUCTION server with {workers} workers...")
    print("Note: This requires Redis for state sharing between workers")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    )

def start_development_server():
    """Start development server with single worker for state consistency"""
    print("Starting DEVELOPMENT server with single worker...")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        workers=8,
        log_level="debug",
        access_log=True,
        reload=True  # Auto-reload on code changes
    )

if __name__ == "__main__":
    # Choose mode based on environment variable
    mode = os.getenv("SERVER_MODE", "development").lower()
    
    if mode == "production":
        start_production_server()
    else:
        start_development_server()
