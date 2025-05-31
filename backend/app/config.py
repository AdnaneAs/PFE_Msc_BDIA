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

# BGE Reranking configuration (based on academic benchmark results)
# Benchmark showed: MAP +23.86%, Precision@5 +23.08%, NDCG@5 +7.09%
ENABLE_RERANKING_BY_DEFAULT = True
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Available BGE reranking models with their configurations
AVAILABLE_RERANKER_MODELS = {
    "BAAI/bge-reranker-base": {
        "name": "BAAI/bge-reranker-base",
        "display_name": "BGE Reranker Base (Recommended)",
        "description": "High-performance reranker with excellent speed/quality balance. Proven +23.86% MAP improvement.",
        "model_size": "~600MB",
        "speed": "Fast",
        "quality": "Excellent",
        "benchmark_results": {
            "map_improvement": 23.86,
            "precision_at_5_improvement": 23.08,
            "ndcg_at_5_improvement": 7.09
        }
    },
    "BAAI/bge-reranker-large": {
        "name": "BAAI/bge-reranker-large",
        "display_name": "BGE Reranker Large (Max Quality)",
        "description": "Largest BGE reranker model for maximum accuracy. Better for complex queries.",
        "model_size": "~1.3GB",
        "speed": "Slower", 
        "quality": "Best",
        "benchmark_results": {
            "map_improvement": "Estimated 25-30%",
            "precision_at_5_improvement": "Estimated 25-30%",
            "ndcg_at_5_improvement": "Estimated 8-12%"
        }
    },
    "BAAI/bge-reranker-v2-m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "display_name": "BGE Reranker v2-M3 (Multilingual)",
        "description": "Latest BGE reranker with multilingual support and enhanced performance.",
        "model_size": "~600MB",
        "speed": "Fast",
        "quality": "Excellent",
        "benchmark_results": {
            "map_improvement": "Estimated 24-28%",
            "precision_at_5_improvement": "Estimated 24-28%", 
            "ndcg_at_5_improvement": "Estimated 7-10%"
        }
    }
}

# Reranking performance settings
RERANKING_INITIAL_RETRIEVAL_MULTIPLIER = 3  # Retrieve 3x documents for reranking
RERANKING_MAX_INITIAL_DOCUMENTS = 50       # Maximum documents to retrieve for reranking
RERANKING_CACHE_SIZE = 1000                 # Cache size for reranker model