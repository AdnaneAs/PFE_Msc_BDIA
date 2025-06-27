# Multi-worker production server with file-based state sharing
# This enables horizontal scaling while maintaining state consistency

import uvicorn
import multiprocessing
import sys
import os

def get_worker_count():
    """Calculate optimal worker count based on CPU cores"""
    cpu_count = multiprocessing.cpu_count()
    # Use (2 * cores) + 1 as recommended for I/O-bound applications
    return min(cpu_count * 2 + 1, 8)  # Cap at 8 workers for stability

if __name__ == "__main__":
    print("Starting multi-worker production server with file-based state sharing...")
    
    # Check if Redis is available (optional, we use file-based by default)
    try:
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()
        print("Redis available - but using file-based state for Windows compatibility")
        workers = get_worker_count()
    except (ImportError, redis.ConnectionError):
        print("Redis not available - using file-based state sharing (recommended for Windows)")
        workers = get_worker_count()
    
    print(f" Starting with {workers} worker(s)")
    print("Background processing will run asynchronously")
    print("State persistence: File-based with cross-process locking")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    )
