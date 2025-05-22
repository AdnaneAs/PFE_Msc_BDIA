# Audit Report Generation Platform

A multi-modal, multi-agent RAG system for audit report generation.

## Project Structure

```
project_root/
├── backend/             # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py      # Main FastAPI app instance
│   │   └── api/         # API routers
│   │       └── __init__.py
│   └── requirements.txt
└── frontend/            # React frontend
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   ├── index.js
    │   ├── index.css
    │   └── services/
    │       └── api.js
    └── package.json
```

## Setup Instructions

### Backend (FastAPI)

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the development server:
   ```
   uvicorn app.main:app --reload
   ```

The API will be available at: http://localhost:8000
The API documentation will be available at: http://localhost:8000/docs

### Frontend (React)

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm start
   ```

The frontend will be available at: http://localhost:3000

## Current Features

- Basic project structure setup
- Backend with a simple "Hello World" API endpoint
- Frontend that communicates with the backend
- Placeholder UI elements for future features 