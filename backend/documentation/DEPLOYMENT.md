# Backend Deployment Options

## Development Mode
For development with hot reload:
```bash
# Windows
start_dev.bat

# Linux/Mac  
./start_dev.sh
```

## Production Mode
For production with multiple workers to handle concurrent requests:
```bash
# Windows
start_production.bat

# Linux/Mac
./start_production.sh

# Or directly with Python
python server.py
```

## Server Configuration
- **Workers**: 4 (automatically adjusted based on CPU cores, max 4)
- **Host**: 0.0.0.0 (accepts connections from all interfaces)
- **Port**: 8000
- **Concurrency Limit**: 50 concurrent requests per worker
- **Request Timeout**: 30 seconds

## Performance Features Added
1. **Multiple Workers**: Up to 4 Uvicorn workers for parallel request processing
2. **Concurrency Control**: Middleware to limit concurrent requests and prevent overload
3. **Graceful Degradation**: Returns 503 status when server is at capacity
4. **Request Counting**: Tracks active requests per worker

## Testing
Health check endpoint: `http://localhost:8000/health`

## Notes
- Gunicorn is included for Linux/Mac deployments but not used on Windows
- Windows uses Uvicorn with multiple workers as alternative
- The concurrency middleware prevents server overload during high traffic
