import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
LLAMAPARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")

# Database paths
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_data")

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Query configuration
TOP_K_RESULTS = 5 