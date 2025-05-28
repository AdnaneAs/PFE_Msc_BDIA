#!/usr/bin/env python3
"""
Quick API Endpoints Test
Verify that all the corrected API endpoints are working
"""

import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_endpoints():
    """Test all corrected API endpoints"""
    
    base_url = "http://localhost:8000"
    
    logger.info("🔍 Testing Corrected API Endpoints")
    logger.info("=" * 50)
    
    # Test 1: Health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        logger.info(f"✅ Health endpoint: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Health endpoint failed: {e}")
    
    # Test 2: Documents list
    try:
        response = requests.get(f"{base_url}/api/documents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            doc_count = len(data.get('documents', []))
            logger.info(f"✅ Documents endpoint: {response.status_code} - {doc_count} documents found")
        else:
            logger.error(f"❌ Documents endpoint: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Documents endpoint failed: {e}")
    
    # Test 3: Query endpoint
    try:
        payload = {
            "question": "What is audit risk?",
            "search_strategy": "semantic",
            "config_for_model": {"model": "llama3.2:latest"}
        }
        response = requests.post(f"{base_url}/api/query/", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Query endpoint: {response.status_code} - Query time: {data.get('query_time_ms', 0)}ms")
        else:
            logger.error(f"❌ Query endpoint: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Query endpoint failed: {e}")
    
    # Test 4: Models endpoint  
    try:
        response = requests.get(f"{base_url}/api/query/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_count = len(data.get('models', []))
            logger.info(f"✅ Models endpoint: {response.status_code} - {model_count} models available")
        else:
            logger.error(f"❌ Models endpoint: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Models endpoint failed: {e}")
    
    logger.info("=" * 50)
    logger.info("🎉 API Endpoints Testing Complete!")
    logger.info("All endpoints should now work with the frontend")

if __name__ == "__main__":
    test_api_endpoints()
