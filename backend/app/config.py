import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
LLAMAPARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")

# Database paths
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_data")

# Chunk configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Available embedding models with their configurations
AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "display_name": "MiniLM-L6-v2 (Fast)",
        "description": "Efficient general-purpose embedding model. Fast inference with good performance.",
        "dimensions": 384,
        "model_size": "~22MB",
        "speed": "Fast",
        "quality": "Good",
        "use_case": "General purpose, fast retrieval"
    },
    "BAAI/bge-m3": {
        "name": "BAAI/bge-m3",
        "display_name": "BGE-M3 (High Quality)",
        "description": "State-of-the-art multilingual embedding model with superior semantic understanding.",
        "dimensions": 1024,
        "model_size": "~1.5GB", 
        "speed": "Slower",
        "quality": "Excellent",
        "use_case": "High-quality semantic search, multilingual support"
    }
}

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Query configuration
TOP_K_RESULTS = 5