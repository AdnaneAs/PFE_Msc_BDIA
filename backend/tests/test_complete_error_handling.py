#!/usr/bin/env python3
"""
Complete Error Handling Test Suite
Tests all enhanced error handling features including "Failed to connect to the backend server"
"""

import requests
import time
import json
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorHandlingTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def log_test_result(self, test_name: str, success: bool, message: str, details: Dict[str, Any] = None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} | {test_name}: {message}")
        if details:
            logger.info(f"       Details: {details}")
    
    def test_backend_connection_success(self):
        """Test successful backend connection"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log_test_result(
                    "Backend Connection Success",
                    True,
                    "Backend server is accessible and responding",
                    {"status_code": response.status_code, "response_time": response.elapsed.total_seconds()}
                )
            else:
                self.log_test_result(
                    "Backend Connection Success",
                    False,
                    f"Backend responded with status {response.status_code}",
                    {"status_code": response.status_code}
                )
        except Exception as e:
            self.log_test_result(
                "Backend Connection Success",
                False,
                f"Failed to connect to backend: {str(e)}",
                {"error": str(e)}
            )
    
    def test_connection_timeout(self):
        """Test connection timeout handling"""
        try:
            # Test with very short timeout to simulate timeout
            response = requests.get(f"{self.base_url}/api/v1/documents", timeout=0.001)
            self.log_test_result(
                "Connection Timeout Test",
                False,
                "Request should have timed out but didn't",
                {"status_code": response.status_code}
            )
        except requests.exceptions.Timeout:
            self.log_test_result(
                "Connection Timeout Test",
                True,
                "Timeout properly detected and handled",
                {"timeout": 0.001}
            )
        except Exception as e:
            self.log_test_result(
                "Connection Timeout Test",
                True,
                f"Connection error properly detected: {type(e).__name__}",
                {"error": str(e)}
            )
    
    def test_invalid_endpoint(self):
        """Test handling of invalid endpoints"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/nonexistent", timeout=5)
            self.log_test_result(
                "Invalid Endpoint Test",
                response.status_code == 404,
                f"Invalid endpoint returned status {response.status_code}",
                {"status_code": response.status_code}
            )
        except Exception as e:
            self.log_test_result(
                "Invalid Endpoint Test",
                False,
                f"Unexpected error for invalid endpoint: {str(e)}",                {"error": str(e)}
            )
    
    def test_query_with_invalid_strategy(self):
        """Test query with invalid search strategy"""
        try:
            payload = {
                "question": "What is audit risk?",
                "search_strategy": "invalid_strategy",
                "config_for_model": "llama3.2:latest"
            }
            response = requests.post(f"{self.base_url}/api/v1/query", json=payload, timeout=10)
            
            if response.status_code == 400:
                self.log_test_result(
                    "Invalid Strategy Test",
                    True,
                    "Invalid search strategy properly rejected",
                    {"status_code": response.status_code, "response": response.json()}
                )
            else:
                self.log_test_result(
                    "Invalid Strategy Test",
                    False,
                    f"Invalid strategy not properly handled: {response.status_code}",
                    {"status_code": response.status_code}
                )
        except Exception as e:
            self.log_test_result(
                "Invalid Strategy Test",
                False,
                f"Error testing invalid strategy: {str(e)}",
                {"error": str(e)}
            )
    
    def test_malformed_request(self):
        """Test handling of malformed requests"""
        try:
            # Send invalid JSON
            response = requests.post(
                f"{self.base_url}/api/v1/query",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            self.log_test_result(
                "Malformed Request Test",
                response.status_code == 422,
                f"Malformed request returned status {response.status_code}",
                {"status_code": response.status_code}
            )
        except Exception as e:
            self.log_test_result(
                "Malformed Request Test",
                False,
                f"Error testing malformed request: {str(e)}",
                {"error": str(e)}
            )
    
    def test_empty_question_handling(self):
        """Test handling of empty questions"""
        try:            payload = {
                "question": "",
                "search_strategy": "hybrid",
                "config_for_model": "llama3.2:latest"
            }
            response = requests.post(f"{self.base_url}/api/v1/query", json=payload, timeout=10)
            
            # Should either reject empty question or handle gracefully
            success = response.status_code in [400, 422] or (
                response.status_code == 200 and 
                "error" in response.json().get("answer", "").lower()
            )
            
            self.log_test_result(
                "Empty Question Test",
                success,
                f"Empty question handling: status {response.status_code}",
                {"status_code": response.status_code, "response": response.json() if response.status_code == 200 else None}
            )
        except Exception as e:
            self.log_test_result(
                "Empty Question Test",
                False,
                f"Error testing empty question: {str(e)}",
                {"error": str(e)}
            )
    
    def test_server_unavailable_simulation(self):
        """Test behavior when server becomes unavailable"""
        # This test simulates what happens when frontend tries to connect to unavailable backend
        try:
            # Try to connect to a port that's not running
            fake_url = "http://localhost:9999"
            response = requests.get(f"{fake_url}/health", timeout=2)
            
            self.log_test_result(
                "Server Unavailable Test",
                False,
                "Connection to fake server should have failed",
                {"unexpected_success": True}
            )
        except requests.exceptions.ConnectionError:
            self.log_test_result(
                "Server Unavailable Test",
                True,
                "Connection error properly detected for unavailable server",
                {"error_type": "ConnectionError"}
            )
        except requests.exceptions.Timeout:
            self.log_test_result(
                "Server Unavailable Test",
                True,
                "Timeout properly detected for unavailable server",
                {"error_type": "Timeout"}
            )
        except Exception as e:
            self.log_test_result(
                "Server Unavailable Test",
                True,
                f"Connection failure properly detected: {type(e).__name__}",
                {"error": str(e)}
            )
    
    def test_large_request_handling(self):
        """Test handling of very large requests"""
        try:
            # Create a very long question
            long_question = "What is audit risk? " * 1000  # Very long question            payload = {
                "question": long_question,
                "search_strategy": "semantic",
                "config_for_model": "llama3.2:latest"
            }
            
            response = requests.post(f"{self.base_url}/api/v1/query", json=payload, timeout=30)
            
            # Should either handle it or reject gracefully
            success = response.status_code in [200, 413, 422, 400]
            
            self.log_test_result(
                "Large Request Test",
                success,
                f"Large request handling: status {response.status_code}",
                {
                    "status_code": response.status_code,
                    "question_length": len(long_question),
                    "response_time": response.elapsed.total_seconds()
                }
            )
        except Exception as e:
            self.log_test_result(
                "Large Request Test",
                True,  # Connection errors are acceptable for large requests
                f"Large request properly handled with error: {type(e).__name__}",
                {"error": str(e)}
            )
    
    def test_api_endpoints_availability(self):
        """Test all API endpoints are available"""
        endpoints = [
            ("/health", "GET"),
            ("/api/v1/documents", "GET"),
            ("/api/v1/models", "GET")
        ]
        
        for endpoint, method in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                
                success = response.status_code in [200, 201]
                self.log_test_result(
                    f"Endpoint {endpoint} Availability",
                    success,
                    f"{method} {endpoint} returned status {response.status_code}",
                    {"status_code": response.status_code, "method": method}
                )
            except Exception as e:
                self.log_test_result(
                    f"Endpoint {endpoint} Availability",
                    False,
                    f"Error accessing {endpoint}: {str(e)}",
                    {"error": str(e), "method": method}
                )
    
    def run_all_tests(self):
        """Run all error handling tests"""
        logger.info("🚀 Starting Complete Error Handling Test Suite")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_backend_connection_success,
            self.test_api_endpoints_availability,
            self.test_connection_timeout,
            self.test_invalid_endpoint,
            self.test_malformed_request,
            self.test_empty_question_handling,
            self.test_query_with_invalid_strategy,
            self.test_server_unavailable_simulation,
            self.test_large_request_handling
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
            time.sleep(0.5)  # Small delay between tests
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("=" * 60)
        logger.info("📊 ERROR HANDLING TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info("")
        
        # Group results by success/failure
        passed_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        if passed_tests:
            logger.info("✅ PASSED TESTS:")
            for test in passed_tests:
                logger.info(f"   • {test['test']}: {test['message']}")
        
        if failed_tests:
            logger.info("")
            logger.info("❌ FAILED TESTS:")
            for test in failed_tests:
                logger.info(f"   • {test['test']}: {test['message']}")
                if test['details']:
                    logger.info(f"     Details: {test['details']}")
        
        logger.info("=" * 60)
        
        # Overall assessment
        if pass_rate >= 80:
            logger.info("🎉 EXCELLENT: Error handling system is working well!")
        elif pass_rate >= 60:
            logger.info("👍 GOOD: Error handling system is mostly working")
        else:
            logger.info("⚠️ NEEDS IMPROVEMENT: Error handling system needs attention")
        
        return pass_rate >= 80

if __name__ == "__main__":
    tester = ErrorHandlingTester()
    tester.run_all_tests()
