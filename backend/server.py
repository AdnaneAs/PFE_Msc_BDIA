# Simple production server using Uvicorn with multiple workers
# This is Windows-compatible alternative to Gunicorn

import uvicorn
import multiprocessing

if __name__ == "__main__":
    # For development/testing: use single worker to avoid state sharing issues
    # For production: implement Redis/database state storage for multi-worker support
    
    print("Starting server in single-worker mode for state consistency...")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker to maintain state consistency
        log_level="info",
        access_log=True,
        reload=False  # Disable reload in production
    )
