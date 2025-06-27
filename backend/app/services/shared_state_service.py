"""
Shared State Manager for multi-worker support (Windows Compatible)
================================================================

This module provides shared state storage using file-based persistence with 
file locking, allowing multiple Uvicorn workers to share workflow state 
without conflicts. Works perfectly on Windows without requiring Redis/WSL2.
"""

import json
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading

try:
    import filelock
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    filelock = None

logger = logging.getLogger(__name__)

class SharedStateManager:
    """File-based shared state manager for multi-worker deployments (Windows Compatible)"""
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize file-based storage
        
        Args:
            storage_dir: Directory to store workflow state files
        """
        if storage_dir is None:
            # Use a fixed directory for multi-worker consistency
            # Place it in the backend app directory to ensure all workers use the same path
            app_dir = Path(__file__).parent.parent  # Go up from services to app
            storage_dir = app_dir / "shared_state_storage"
        
        self.storage_dir = Path(storage_dir) 
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file to track workflows
        self.index_file = self.storage_dir / "workflow_index.json"
        
        # Thread lock for local operations
        self._local_lock = threading.Lock()
        
        logger.info(f"Initialized shared state manager with storage: {self.storage_dir}")
        logger.info(f"File locking available: {FILELOCK_AVAILABLE}")
        
        # Ensure directory permissions are correct
        try:
            # Test write access with a unique file per process
            import os, uuid
            test_file = self.storage_dir / f"test_write_{os.getpid()}_{uuid.uuid4().hex}.tmp"
            test_file.write_text("test")
            try:
                test_file.unlink()
            except (FileNotFoundError, PermissionError):
                pass
            logger.info(" Storage directory write access confirmed")
        except Exception as e:
            logger.error(f" Storage directory write access failed: {e}")
            raise RuntimeError(f"Cannot write to shared state storage: {e}")
    
    def _get_workflow_file(self, workflow_id: str) -> Path:
        """Get the file path for a workflow"""
        return self.storage_dir / f"workflow_{workflow_id}.pkl"
    
    def _get_lock_file(self, workflow_id: str) -> Path:
        """Get the lock file path for a workflow"""
        return self.storage_dir / f"workflow_{workflow_id}.lock"
    
    def _update_index(self, workflow_id: str, action: str = "add"):
        """Update the workflow index file"""
        if not FILELOCK_AVAILABLE:
            # Fallback without file locking
            return self._update_index_fallback(workflow_id, action)
        
        lock_file = self.storage_dir / "index.lock"
        
        try:
            with filelock.FileLock(str(lock_file), timeout=5):
                # Read current index
                index_data = {}
                if self.index_file.exists():
                    try:
                        with open(self.index_file, 'r') as f:
                            index_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        index_data = {}
                
                # Update index
                if action == "add":
                    index_data[workflow_id] = {
                        "created_at": datetime.now().isoformat(),
                        "last_access": datetime.now().isoformat()
                    }
                elif action == "remove" and workflow_id in index_data:
                    del index_data[workflow_id]
                elif action == "update" and workflow_id in index_data:
                    index_data[workflow_id]["last_access"] = datetime.now().isoformat()

                # Write updated index
                with open(self.index_file, 'w') as f:
                    json.dump(index_data, f, indent=2)
                    
        except filelock.Timeout:
            logger.warning(f"Could not acquire lock for index update: {workflow_id}")
        except Exception as e:
            logger.error(f"Error updating index for {workflow_id}: {e}")
    
    def _update_index_fallback(self, workflow_id: str, action: str = "add"):
        """Fallback index update without file locking"""
        try:
            with self._local_lock:
                # Read current index
                index_data = {}
                if self.index_file.exists():
                    try:
                        with open(self.index_file, 'r') as f:
                            index_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        index_data = {}
                
                # Update index
                if action == "add":
                    index_data[workflow_id] = {
                        "created_at": datetime.now().isoformat(),
                        "last_access": datetime.now().isoformat()
                    }
                elif action == "remove" and workflow_id in index_data:
                    del index_data[workflow_id]
                elif action == "update" and workflow_id in index_data:
                    index_data[workflow_id]["last_access"] = datetime.now().isoformat()

                # Write updated index
                with open(self.index_file, 'w') as f:
                    json.dump(index_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error updating index for {workflow_id}: {e}")
    
    def store_workflow_state(self, workflow_id: str, state: Dict[str, Any], ttl_hours: int = 24) -> bool:
        """
        Store workflow state with TTL
        
        Args:
            workflow_id: Workflow identifier
            state: State dictionary to store
            ttl_hours: Time to live in hours
            
        Returns:
            True if stored successfully
        """
        try:
            workflow_file = self._get_workflow_file(workflow_id)
            
            # Add metadata
            state_with_meta = {
                'state': state,
                'stored_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
            }
            
            # Store the state
            if FILELOCK_AVAILABLE:
                lock_file = self._get_lock_file(workflow_id)
                try:
                    with filelock.FileLock(str(lock_file), timeout=5):
                        with open(workflow_file, 'wb') as f:
                            pickle.dump(state_with_meta, f)
                except filelock.Timeout:
                    logger.warning(f"Could not acquire lock for storing {workflow_id}")
                    return False
            else:
                # Fallback without locking
                with open(workflow_file, 'wb') as f:
                    pickle.dump(state_with_meta, f)
            
            # Update index
            self._update_index(workflow_id, "add")
            
            logger.debug(f"Stored workflow {workflow_id} with TTL {ttl_hours}h")
            return True
                
        except Exception as e:
            logger.error(f"Failed to store workflow {workflow_id}: {e}")
            return False
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve workflow state
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            State dictionary or None if not found
        """
        try:
            workflow_file = self._get_workflow_file(workflow_id)
            
            if not workflow_file.exists():
                return None
            
            # Load the state
            if FILELOCK_AVAILABLE:
                lock_file = self._get_lock_file(workflow_id)
                try:
                    with filelock.FileLock(str(lock_file), timeout=5):
                        with open(workflow_file, 'rb') as f:
                            state_with_meta = pickle.load(f)
                except filelock.Timeout:
                    logger.warning(f"Could not acquire lock for reading {workflow_id}")
                    return None
            else:
                # Fallback without locking
                with open(workflow_file, 'rb') as f:
                    state_with_meta = pickle.load(f)
            
            # Check expiration
            expires_at = datetime.fromisoformat(state_with_meta['expires_at'])
            if datetime.now() > expires_at:
                logger.debug(f"Workflow {workflow_id} has expired, removing")
                self.delete_workflow_state(workflow_id)
                return None
            
            # Update last access
            self._update_index(workflow_id, "update")
            
            logger.debug(f"Retrieved workflow {workflow_id}")
            return state_with_meta['state']
            
        except Exception as e:
            logger.error(f"Failed to retrieve workflow {workflow_id}: {e}")
            return None
    
    def list_workflows(self) -> List[str]:
        """
        List all active workflow IDs
        
        Returns:
            List of workflow IDs
        """
        try:
            if not self.index_file.exists():
                return []
            
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
            
            active_workflows = []
            expired_workflows = []
            current_time = datetime.now()
            
            for workflow_id, meta in index_data.items():
                # Check if workflow file still exists
                workflow_file = self._get_workflow_file(workflow_id)
                if workflow_file.exists():
                    try:
                        # Quick check of expiration without full load
                        with open(workflow_file, 'rb') as f:
                            state_with_meta = pickle.load(f)
                        expires_at = datetime.fromisoformat(state_with_meta['expires_at'])
                        
                        if current_time < expires_at:
                            active_workflows.append(workflow_id)
                        else:
                            expired_workflows.append(workflow_id)
                    except:
                        # File corrupted, mark for cleanup
                        expired_workflows.append(workflow_id)
                else:
                    # File doesn't exist, mark for cleanup
                    expired_workflows.append(workflow_id)
            
            # Clean up expired workflows
            for workflow_id in expired_workflows:
                self.delete_workflow_state(workflow_id)
            
            return active_workflows
                
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return []
    
    def delete_workflow_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            workflow_file = self._get_workflow_file(workflow_id)
            lock_file = self._get_lock_file(workflow_id)
            
            # Remove files
            if workflow_file.exists():
                workflow_file.unlink()
            if lock_file.exists():
                lock_file.unlink()
            
            # Update index
            self._update_index(workflow_id, "remove")
            
            logger.debug(f"Deleted workflow {workflow_id}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the state manager
        
        Returns:
            Health status dictionary
        """
        try:
            workflow_count = len(self.list_workflows())
            storage_exists = self.storage_dir.exists()
            
            return {
                "status": "healthy" if storage_exists else "degraded",
                "backend": "file-based",
                "workflow_count": workflow_count,
                "storage_dir": str(self.storage_dir),
                "file_locking": FILELOCK_AVAILABLE,
                "storage_accessible": storage_exists
            }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "file-based",
                "error": str(e),
                "storage_accessible": False
            }


# Global shared state manager instance
shared_state_manager = SharedStateManager()
