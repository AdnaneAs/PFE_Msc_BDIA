"""
Redis-based State Manager for Docker multi-worker support
========================================================

This module provides Redis-based state storage for true multi-worker
support in Docker environments.
"""

import json
import logging
import os
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RedisStateManager:
    """Redis-based state manager for Docker deployments"""
    
    def __init__(self, redis_url: str = None):
        """
        Initialize Redis state manager
        
        Args:
            redis_url: Redis connection URL (default: from environment)
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        self.connected = False
        
        # State management
        self.workflow_states = {}  # Fallback in-memory storage
        
        # Connect to Redis
        self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis with error handling"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary for pickle
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            logger.info(f"✅ Redis connected successfully: {self.redis_url}")
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.info("🔄 Using in-memory fallback storage")
            self.connected = False
            self.redis_client = None
    
    def store_workflow_state(self, workflow_id: str, state: Dict[str, Any]) -> bool:
        """
        Store workflow state in Redis with fallback
        
        Args:
            workflow_id: Unique workflow identifier
            state: Workflow state to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not workflow_id:
            logger.warning("Cannot store state - no workflow_id provided")
            return False
        
        try:
            # Prepare state for storage (handle enum serialization)
            serializable_state = self._prepare_state_for_storage(state)
            
            if self.connected and self.redis_client:
                # Store in Redis
                key = f"workflow:{workflow_id}"
                serialized_state = pickle.dumps(serializable_state)
                
                # Store with 24-hour expiration
                self.redis_client.setex(key, 86400, serialized_state)
                
                # Update workflow index
                self.redis_client.sadd("workflow_ids", workflow_id)
                
                logger.debug(f"📦 Stored workflow {workflow_id} in Redis")
                
            # Always store in memory as fallback
            self.workflow_states[workflow_id] = serializable_state.copy()
            
            logger.debug(f"💾 Stored workflow {workflow_id} (Redis: {self.connected})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store workflow {workflow_id}: {e}")
            
            # Try to store in memory at least
            try:
                serializable_state = self._prepare_state_for_storage(state)
                self.workflow_states[workflow_id] = serializable_state.copy()
                logger.info(f"📝 Stored workflow {workflow_id} in memory fallback")
                return True
            except Exception as fallback_error:
                logger.error(f"❌ Memory fallback also failed: {fallback_error}")
                return False
    
    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve workflow state from Redis with fallback
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            Workflow state if found, None otherwise
        """
        if not workflow_id:
            return None
        
        try:
            if self.connected and self.redis_client:
                # Try Redis first
                key = f"workflow:{workflow_id}"
                serialized_state = self.redis_client.get(key)
                
                if serialized_state:
                    state = pickle.loads(serialized_state)
                    logger.debug(f"📦 Retrieved workflow {workflow_id} from Redis")
                    return state
            
            # Fallback to memory
            if workflow_id in self.workflow_states:
                state = self.workflow_states[workflow_id].copy()
                logger.debug(f"💾 Retrieved workflow {workflow_id} from memory")
                return state
            
            logger.debug(f"❌ Workflow {workflow_id} not found in any store")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve workflow {workflow_id}: {e}")
            return None
    
    def list_workflow_ids(self) -> List[str]:
        """
        List all workflow IDs
        
        Returns:
            List of workflow IDs
        """
        workflow_ids = set()
        
        try:
            if self.connected and self.redis_client:
                # Get from Redis
                redis_ids = self.redis_client.smembers("workflow_ids")
                workflow_ids.update([wid.decode() for wid in redis_ids])
            
            # Add from memory
            workflow_ids.update(self.workflow_states.keys())
            
            return list(workflow_ids)
            
        except Exception as e:
            logger.error(f"❌ Failed to list workflow IDs: {e}")
            return list(self.workflow_states.keys())
    
    def delete_workflow_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            True if deleted successfully
        """
        success = True
        
        try:
            if self.connected and self.redis_client:
                # Delete from Redis
                key = f"workflow:{workflow_id}"
                self.redis_client.delete(key)
                self.redis_client.srem("workflow_ids", workflow_id)
            
            # Delete from memory
            if workflow_id in self.workflow_states:
                del self.workflow_states[workflow_id]
            
            logger.info(f"🗑️ Deleted workflow {workflow_id}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to delete workflow {workflow_id}: {e}")
            return False
    
    def cleanup_expired_workflows(self, max_age_hours: int = 24):
        """
        Clean up old workflows
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for workflow_id in self.list_workflow_ids():
                state = self.get_workflow_state(workflow_id)
                if state and state.get('updated_at'):
                    try:
                        updated_at = datetime.fromisoformat(state['updated_at'])
                        if updated_at < cutoff_time:
                            self.delete_workflow_state(workflow_id)
                            logger.info(f"🧹 Cleaned up expired workflow {workflow_id}")
                    except ValueError:
                        # Invalid date format, skip
                        pass
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired workflows: {e}")
    
    def _prepare_state_for_storage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare state for storage by handling enum serialization
        
        Args:
            state: Raw state dictionary
            
        Returns:
            Serializable state dictionary
        """
        serializable_state = {}
        
        for key, value in state.items():
            if hasattr(value, 'value'):  # Enum
                serializable_state[key] = value.value
            elif hasattr(value, '__dict__'):  # Complex object
                try:
                    serializable_state[key] = value.__dict__
                except:
                    serializable_state[key] = str(value)
            else:
                serializable_state[key] = value
        
        return serializable_state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        return {
            "redis_connected": self.connected,
            "redis_url": self.redis_url,
            "memory_workflows": len(self.workflow_states),
            "total_workflows": len(self.list_workflow_ids())
        }

# Global instance
redis_state_manager = RedisStateManager()
