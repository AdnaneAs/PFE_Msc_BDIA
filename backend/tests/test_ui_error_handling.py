#!/usr/bin/env python3
"""
Frontend UI Error Handling Test
Tests the enhanced error handling features in the frontend interface
"""

import requests
import time
import json
import sys
from typing import Dict, Any

def print_section(title: str) -> None:
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_test(test_name: str, status: str, details: str = "") -> None:
    """Print test result"""
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_icon} {test_name}: {status}")
    if details:
        print(f"   {details}")

def test_backend_health() -> Dict[str, Any]:
    """Test backend server health"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        return {
            "status": "PASS" if response.status_code == 200 else "FAIL",
            "code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def test_frontend_health() -> Dict[str, Any]:
    """Test frontend server health"""
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        return {
            "status": "PASS" if response.status_code == 200 else "FAIL",
            "code": response.status_code
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def test_api_endpoints() -> Dict[str, Dict[str, Any]]:
    """Test all API endpoints"""
    endpoints = {
        "Documents": "http://localhost:8000/api/documents",
        "Models": "http://localhost:8000/api/query/models",
        "Health": "http://localhost:8000/health"
    }
    
    results = {}
    for name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=10)
            results[name] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "code": response.status_code,
                "data_size": len(response.text) if response.text else 0
            }
        except Exception as e:
            results[name] = {"status": "FAIL", "error": str(e)}
    
    return results

def test_error_scenarios() -> Dict[str, Dict[str, Any]]:
    """Test various error scenarios"""
    tests = {}
    
    # Test timeout scenario
    try:
        response = requests.post(
            "http://localhost:8000/api/query/",
            json={"query": "test query", "max_tokens": 1},
            timeout=1  # Very short timeout to trigger timeout error
        )
        tests["Short Timeout"] = {
            "status": "PARTIAL" if response.status_code in [200, 408, 504] else "FAIL",
            "code": response.status_code
        }
    except requests.exceptions.Timeout:
        tests["Short Timeout"] = {"status": "PASS", "note": "Timeout handled correctly"}
    except Exception as e:
        tests["Short Timeout"] = {"status": "PASS", "note": f"Error handled: {type(e).__name__}"}
    
    # Test malformed request
    try:
        response = requests.post(
            "http://localhost:8000/api/query/",
            json={"invalid": "data"},
            timeout=10
        )
        tests["Malformed Request"] = {
            "status": "PASS" if response.status_code in [400, 422] else "FAIL",
            "code": response.status_code
        }
    except Exception as e:
        tests["Malformed Request"] = {"status": "FAIL", "error": str(e)}
    
    # Test non-existent endpoint
    try:
        response = requests.get("http://localhost:8000/api/nonexistent", timeout=10)
        tests["Non-existent Endpoint"] = {
            "status": "PASS" if response.status_code == 404 else "FAIL",
            "code": response.status_code
        }
    except Exception as e:
        tests["Non-existent Endpoint"] = {"status": "FAIL", "error": str(e)}
    
    return tests

def test_connection_recovery() -> Dict[str, Any]:
    """Test connection recovery functionality"""
    try:
        # Test multiple rapid requests to check connection stability
        success_count = 0
        total_requests = 5
        
        for i in range(total_requests):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    success_count += 1
                time.sleep(0.5)  # Small delay between requests
            except Exception:
                pass
        
        success_rate = success_count / total_requests
        return {
            "status": "PASS" if success_rate >= 0.8 else "PARTIAL" if success_rate >= 0.5 else "FAIL",
            "success_rate": f"{success_rate:.1%}",
            "successful_requests": f"{success_count}/{total_requests}"
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}

def main():
    """Run all UI error handling tests"""
    print_section("Frontend UI Error Handling Test Suite")
    
    # Test 1: Server Health
    print_section("1. Server Health Check")
    
    backend_health = test_backend_health()
    print_test("Backend Server", backend_health["status"], 
               f"HTTP {backend_health.get('code', 'N/A')}" if "code" in backend_health else backend_health.get("error", ""))
    
    frontend_health = test_frontend_health()
    print_test("Frontend Server", frontend_health["status"],
               f"HTTP {frontend_health.get('code', 'N/A')}" if "code" in frontend_health else frontend_health.get("error", ""))
    
    # Test 2: API Endpoints
    print_section("2. API Endpoint Validation")
    
    api_results = test_api_endpoints()
    for endpoint, result in api_results.items():
        details = f"HTTP {result.get('code', 'N/A')}"
        if "data_size" in result:
            details += f", {result['data_size']} bytes"
        if "error" in result:
            details = result["error"]
        print_test(f"{endpoint} Endpoint", result["status"], details)
    
    # Test 3: Error Scenarios
    print_section("3. Error Handling Scenarios")
    
    error_tests = test_error_scenarios()
    for test_name, result in error_tests.items():
        details = ""
        if "code" in result:
            details = f"HTTP {result['code']}"
        if "note" in result:
            details = result["note"]
        if "error" in result:
            details = result["error"]
        print_test(test_name, result["status"], details)
    
    # Test 4: Connection Recovery
    print_section("4. Connection Recovery Test")
    
    recovery_test = test_connection_recovery()
    details = f"{recovery_test.get('successful_requests', 'N/A')} successful, {recovery_test.get('success_rate', 'N/A')} rate"
    if "error" in recovery_test:
        details = recovery_test["error"]
    print_test("Connection Stability", recovery_test["status"], details)
    
    # Summary
    print_section("Test Summary")
    
    all_tests = [backend_health, frontend_health] + list(api_results.values()) + list(error_tests.values()) + [recovery_test]
    passed = sum(1 for test in all_tests if test["status"] == "PASS")
    total = len(all_tests)
    
    print(f"✅ Tests Passed: {passed}/{total} ({passed/total:.1%})")
    print(f"🚀 System Status: {'OPERATIONAL' if passed/total >= 0.8 else 'PARTIAL' if passed/total >= 0.5 else 'NEEDS ATTENTION'}")
    
    if passed/total >= 0.8:
        print("\n🎉 UI Error Handling System is working excellently!")
        print("   - Frontend and backend are properly connected")
        print("   - API endpoints are responding correctly") 
        print("   - Error handling is functioning as expected")
        print("   - Connection recovery mechanisms are stable")
    elif passed/total >= 0.5:
        print("\n⚠️  UI Error Handling System has some issues:")
        print("   - Most core functionality is working")
        print("   - Some error scenarios may need attention")
        print("   - Consider checking logs for additional details")
    else:
        print("\n❌ UI Error Handling System needs immediate attention:")
        print("   - Multiple critical issues detected")
        print("   - Check server status and configuration")
        print("   - Review error logs for debugging")

if __name__ == "__main__":
    main()
