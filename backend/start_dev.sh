#!/bin/bash
# Development start script with single worker for debugging

echo "Starting Audit Backend in development mode..."

# Start with Uvicorn for development (single worker)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
