#!/usr/bin/env python3
"""
Test script for async workflow processing functionality.
This script verifies that the approval endpoint returns immediately
and that background processing works correctly.
"""

import asyncio
import sys
import os
import logging
import json
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agents.workflow_manager import AuditWorkflowManager
from app.agents.base_agent import AgentState, WorkflowStatus
from app.config import get_llm_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AsyncProcessingTester:
    """Test async processing functionality"""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.manager = AuditWorkflowManager(self.llm_config)
    
    async def test_immediate_approval_response(self) -> bool:
        """Test that approval returns immediately"""
        logger.info("=== Testing Immediate Approval Response ===")
        
        try:
            # Create a workflow in awaiting_human state
            initial_state: AgentState = {
                "workflow_id": "test_async_001",
                "status": WorkflowStatus.PLANNING,
                "enterprise_info": {"name": "Async Test Corp", "id": "atc001"},
                "selected_norms": ["ISO27001"],
                "audit_plan": {
                    "plan_name": "Test Audit Plan",
                    "audit_objective": "Test async processing",
                    "selected_norms": ["ISO27001"],
                    "audit_cycles": [
                        {
                            "cycle_name": "Test Controls",
                            "description": "Testing async functionality",
                            "estimated_hours": 4
                        }
                    ],
                    "created_at": datetime.now().isoformat()
                },
                "awaiting_human": True,
                "plan_approved": False,
                "messages": [],
                "errors": [],
                "react_steps": [],
                "updated_at": datetime.now().isoformat()
            }
            
            # Store the state
            self.manager.workflow_states["test_async_001"] = initial_state.copy()
            
            # Test immediate response time
            start_time = datetime.now()
            
            # Call continue_workflow_async (simulating approval)
            result = await self.manager.continue_workflow_async(
                workflow_id="test_async_001",
                approval=True,
                human_feedback="Approved for async testing"
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Check that response was immediate (less than 1 second)
            if response_time < 1.0:
                logger.info(f"✅ Immediate response test PASSED: {response_time:.3f} seconds")
                
                # Check response content
                if (result.get("background_processing") == True and 
                    result.get("status") == "analyzing"):
                    logger.info("✅ Response content test PASSED")
                    return True
                else:
                    logger.error(f"❌ Response content test FAILED: {result}")
                    return False
            else:
                logger.error(f"❌ Immediate response test FAILED: {response_time:.3f} seconds (too slow)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Immediate approval response test FAILED with exception: {str(e)}")
            return False
    
    async def test_background_processing(self) -> bool:
        """Test that background processing works"""
        logger.info("=== Testing Background Processing ===")
        
        try:
            # Wait a bit to let background processing start
            await asyncio.sleep(2)
            
            # Check if the workflow state has been updated by background processing
            final_state = self.manager.workflow_states.get("test_async_001", {})
            
            # Check if background processing indicators are present
            background_started = final_state.get("background_processing", False)
            status_changed = final_state.get("status") != WorkflowStatus.PLANNING
            has_messages = len(final_state.get("messages", [])) > 1
            
            if background_started or status_changed or has_messages:
                logger.info("✅ Background processing test PASSED")
                logger.info(f"Background processing: {background_started}")
                logger.info(f"Status: {final_state.get('status')}")
                logger.info(f"Messages: {len(final_state.get('messages', []))}")
                return True
            else:
                logger.error("❌ Background processing test FAILED - No signs of background activity")
                logger.error(f"State: {final_state}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Background processing test FAILED with exception: {str(e)}")
            return False
    
    async def test_status_monitoring(self) -> bool:
        """Test status monitoring functionality"""
        logger.info("=== Testing Status Monitoring ===")
        
        try:
            # Get workflow status
            status = await self.manager.get_workflow_status("test_async_001")
            
            # Check if status includes background processing info
            if (status.get("background_processing") is not None and
                "debug_info" in status):
                logger.info("✅ Status monitoring test PASSED")
                logger.info(f"Status: {status.get('status')}")
                logger.info(f"Background processing: {status.get('background_processing')}")
                return True
            else:
                logger.error("❌ Status monitoring test FAILED")
                logger.error(f"Status response: {status}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Status monitoring test FAILED with exception: {str(e)}")
            return False
    
    async def test_rejection_handling(self) -> bool:
        """Test that rejection also works immediately"""
        logger.info("=== Testing Rejection Handling ===")
        
        try:
            # Create another workflow for rejection test
            initial_state: AgentState = {
                "workflow_id": "test_async_002",
                "status": WorkflowStatus.PLANNING,
                "enterprise_info": {"name": "Reject Test Corp", "id": "rtc002"},
                "selected_norms": ["ISO27001"],
                "audit_plan": {"plan_name": "Test Plan for Rejection"},
                "awaiting_human": True,
                "plan_approved": False,
                "messages": [],
                "errors": [],
                "react_steps": [],
                "updated_at": datetime.now().isoformat()
            }
            
            self.manager.workflow_states["test_async_002"] = initial_state.copy()
            
            # Test rejection
            start_time = datetime.now()
            
            result = await self.manager.continue_workflow_async(
                workflow_id="test_async_002",
                approval=False,
                human_feedback="Rejected for testing"
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Check immediate response and correct status
            if (response_time < 1.0 and 
                result.get("background_processing") == False and
                result.get("status") == "planning"):
                logger.info(f"✅ Rejection handling test PASSED: {response_time:.3f} seconds")
                return True
            else:
                logger.error(f"❌ Rejection handling test FAILED")
                logger.error(f"Response time: {response_time:.3f}, Result: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Rejection handling test FAILED with exception: {str(e)}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all async processing tests"""
        logger.info("🚀 Starting Async Processing Tests")
        logger.info("=" * 50)
        
        test_results = []
        
        # Run tests in sequence
        test_results.append(await self.test_immediate_approval_response())
        test_results.append(await self.test_background_processing())
        test_results.append(await self.test_status_monitoring())
        test_results.append(await self.test_rejection_handling())
        
        # Summary
        passed = sum(test_results)
        total = len(test_results)
        
        logger.info("=" * 50)
        logger.info(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL ASYNC PROCESSING TESTS PASSED!")
            logger.info("The system now provides immediate responses and background processing.")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed. Async processing needs improvement.")
            return False

async def main():
    """Main test function"""
    tester = AsyncProcessingTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
