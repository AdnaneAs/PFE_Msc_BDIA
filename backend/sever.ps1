# Development start script for Windows (PowerShell)

Write-Host "Starting Audit Backend in development mode..."
conda activate pfe

# Start with Uvicorn for development (single worker)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info