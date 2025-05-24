"""
Query metrics logging module for tracking performance and usage metrics.
"""
import logging
import time
import json
import os
import datetime
from typing import Dict, Any, Optional

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO)

# Create a dedicated logger for query metrics
metrics_logger = logging.getLogger("query_metrics")
metrics_logger.setLevel(logging.INFO)

# Add a file handler to log to a separate metrics file
metrics_file_handler = logging.FileHandler(f'logs/query_metrics_{datetime.datetime.now().strftime("%Y%m%d")}.log')
metrics_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
metrics_logger.addHandler(metrics_file_handler)

class QueryMetricsTracker:
    """
    Class for tracking query metrics and logging them.
    """
    def __init__(self, question: str, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new query metrics tracker.
        
        Args:
            question: The user's question
            model_config: Optional model configuration
        """
        self.question = question
        self.model_config = model_config or {}
        self.start_time = time.time()
        self.retrieval_start_time = None
        self.retrieval_end_time = None
        self.llm_start_time = None
        self.llm_end_time = None
        self.num_documents = 0
        self.model_info = None
        self.answer_length = 0
        self.error = None
        
        # Log the start of the query
        self._log_start()
        
    def _log_start(self):
        """Log the start of a query"""
        metrics_logger.info(json.dumps({
            "event": "query_start",
            "question": self.question,
            "model_config": self.model_config,
            "timestamp": self.start_time
        }))
        
    def mark_retrieval_start(self):
        """Mark the start of document retrieval"""
        self.retrieval_start_time = time.time()
        
    def mark_retrieval_end(self, num_documents: int):
        """
        Mark the end of document retrieval
        
        Args:
            num_documents: Number of documents retrieved
        """
        self.retrieval_end_time = time.time()
        self.num_documents = num_documents
        
        # Log retrieval metrics
        retrieval_time = self.retrieval_end_time - self.retrieval_start_time if self.retrieval_start_time else 0
        metrics_logger.info(json.dumps({
            "event": "retrieval_complete",
            "question": self.question,
            "num_documents": num_documents,
            "retrieval_time_ms": int(retrieval_time * 1000),
            "timestamp": self.retrieval_end_time
        }))
        
    def mark_llm_start(self):
        """Mark the start of LLM processing"""
        self.llm_start_time = time.time()
        
    def mark_llm_end(self, model_info: str, answer_length: int):
        """
        Mark the end of LLM processing
        
        Args:
            model_info: Information about the model used
            answer_length: Length of the generated answer
        """
        self.llm_end_time = time.time()
        self.model_info = model_info
        self.answer_length = answer_length
        
        # Log LLM processing metrics
        llm_time = self.llm_end_time - self.llm_start_time if self.llm_start_time else 0
        metrics_logger.info(json.dumps({
            "event": "llm_complete",
            "question": self.question,
            "model_info": model_info,
            "answer_length": answer_length,
            "llm_time_ms": int(llm_time * 1000),
            "timestamp": self.llm_end_time
        }))
        
    def mark_error(self, error_message: str):
        """
        Mark an error in query processing
        
        Args:
            error_message: Description of the error
        """
        self.error = error_message
        end_time = time.time()
        
        # Log error metrics
        metrics_logger.error(json.dumps({
            "event": "query_error",
            "question": self.question,
            "error": error_message,
            "total_time_ms": int((end_time - self.start_time) * 1000),
            "timestamp": end_time
        }))
        
    def complete(self):
        """Mark the completion of the query and log final metrics"""
        if self.error:
            # Already logged in mark_error
            return
            
        end_time = time.time()
        total_time = end_time - self.start_time
        retrieval_time = (self.retrieval_end_time - self.retrieval_start_time) if self.retrieval_start_time and self.retrieval_end_time else 0
        llm_time = (self.llm_end_time - self.llm_start_time) if self.llm_start_time and self.llm_end_time else 0
        
        # Calculate overhead time (total - retrieval - llm)
        overhead_time = total_time - retrieval_time - llm_time
        
        # Log complete metrics
        metrics_logger.info(json.dumps({
            "event": "query_complete",
            "question": self.question,
            "model_info": self.model_info,
            "num_documents": self.num_documents,
            "answer_length": self.answer_length,
            "total_time_ms": int(total_time * 1000),
            "retrieval_time_ms": int(retrieval_time * 1000),
            "llm_time_ms": int(llm_time * 1000),
            "overhead_time_ms": int(overhead_time * 1000),
            "timestamp": end_time
        }))
        
        return {
            "total_time_ms": int(total_time * 1000),
            "retrieval_time_ms": int(retrieval_time * 1000),
            "llm_time_ms": int(llm_time * 1000)
        }
