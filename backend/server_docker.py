"""
Docker-optimized multi-worker server with Redis state sharing
============================================================

This server is designed for Docker deployments with proper Redis integration
for shared state management across multiple worker processes.
"""

import os
import uvicorn
import logging
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_workers():
    """Get number of workers from environment or default to 4"""
    try:
        workers = int(os.getenv('WORKERS', '4'))
        return max(1, min(workers, 8))  # Between 1 and 8 workers
    except ValueError:
        return 4

def main():
    """Main server entry point"""
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    workers = get_workers()
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    environment = os.getenv('ENVIRONMENT', 'development')
    
    logger.info("🐳 Starting Docker-optimized PFE Audit System Backend")
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Host: {host}")
    logger.info(f"   - Port: {port}")
    logger.info(f"   - Workers: {workers}")
    logger.info(f"   - Redis URL: {redis_url}")
    logger.info(f"   - Environment: {environment}")
    
    # Test Redis connection
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.info("🔄 Falling back to file-based state sharing")
    
    # Start server
    logger.info(f"🚀 Starting server with {workers} worker(s)")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False,
        # Docker-specific optimizations
        loop="asyncio",
        http="httptools"
    )

if __name__ == "__main__":
    main()
