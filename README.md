# Audit Report Generation Platform

A multi-modal, multi-agent RAG system for audit report generation.

## Project Structure

```
project_root/
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # Main FastAPI app instance
│   │   ├── config.py         # Configuration and environment variables
│   │   ├── models.py         # Pydantic models for API
│   │   ├── api/              # API routers
│   │   │   ├── __init__.py
│   │   │   └── v1/           # API v1 endpoints
│   │   │       ├── __init__.py
│   │   │       ├── documents.py  # Document upload endpoints
│   │   │       └── query.py      # Query endpoints
│   │   └── services/         # Service modules
│   │       ├── __init__.py
│   │       ├── llamaparse_service.py  # PDF parsing
│   │       ├── text_processing_service.py  # Text chunking
│   │       ├── embedding_service.py   # Text embeddings
│   │       ├── vector_db_service.py   # ChromaDB operations
│   │       └── llm_service.py         # LLM integration
│   ├── db_data/              # ChromaDB storage (created at runtime)
│   └── requirements.txt
└── frontend/                 # React frontend
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   ├── index.js
    │   ├── index.css
    │   ├── components/
    │   │   ├── FileUpload.js      # File upload component
    │   │   ├── QueryInput.js      # Query input component
    │   │   └── ResultDisplay.js   # Results display component
    │   └── services/
    │       └── api.js             # API service functions
    ├── tailwind.config.js
    ├── postcss.config.js
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

5. Create a `.env` file with your API keys:
   ```
   LLAMAPARSE_API_KEY=your_llamaparse_api_key_here
   # Optional: OPENAI_API_KEY=your_openai_api_key_here
   # Optional: OLLAMA_ENDPOINT=http://localhost:11434/api/generate
   ```

6. Run the development server:
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

## Features

- **Document Upload:** Upload PDF documents for processing
  - Documents are parsed using LlamaParse
  - Text is chunked and embedded using Sentence Transformers
  - Embeddings are stored in ChromaDB

- **Document Query:** Ask questions about your documents
  - Questions are embedded and used to retrieve relevant document chunks
  - Retrieved chunks are used as context for an LLM to generate answers

## API Endpoints

- `GET /api/hello` - Test endpoint
- `POST /api/documents/upload` - Upload and process a PDF document
- `POST /api/query` - Query the documents with a question 