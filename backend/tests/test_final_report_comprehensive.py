#!/usr/bin/env python3
"""
Comprehensive test script for final report generation and access improvements.
Tests the robustness of the agentic audit workflow system.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agents.workflow_manager import WorkflowManager
from app.agents.base_agent import AgentState, WorkflowStatus
from app.config import get_llm_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveFinalReportTester:
    """Comprehensive tester for final report functionality"""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.manager = WorkflowManager(self.llm_config)
    
    async def test_scenario_1_normal_completion(self) -> bool:
        """Test normal workflow completion with sections"""
        logger.info("=== Testing Scenario 1: Normal Completion ===")
        
        try:
            # Create initial state with some generated sections
            initial_state: AgentState = {
                "workflow_id": "test_normal_001",
                "status": WorkflowStatus.CONSOLIDATING,
                "enterprise_info": {"name": "Test Corp", "id": "tc001"},
                "selected_norms": ["ISO27001", "GDPR"],
                "generated_sections": {
                    "security_analysis": {
                        "analysis": "Security framework analysis completed",
                        "findings": ["Encryption properly implemented", "Access controls need improvement"],
                        "recommendations": ["Implement MFA", "Review user permissions"]
                    },
                    "compliance_review": {
                        "analysis": "Compliance review completed",
                        "findings": ["GDPR requirements mostly met", "Documentation gaps found"],
                        "recommendations": ["Update privacy policy", "Conduct staff training"]
                    }
                },
                "messages": [],
                "errors": [],
                "react_steps": [],
                "updated_at": datetime.now().isoformat()
            }
            
            # Store the state
            self.manager.workflow_states["test_normal_001"] = initial_state.copy()
            
            # Progress the workflow (should complete and create final report)
            result = await self.manager.progress_workflow("test_normal_001")
            
            # Check if final report was created
            final_state = self.manager.workflow_states.get("test_normal_001", {})
            
            if final_state.get("final_report"):
                logger.info("✅ Normal completion test PASSED")
                logger.info(f"Final report length: {len(final_state['final_report'])} characters")
                return True
            else:
                logger.error("❌ Normal completion test FAILED - No final report created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Normal completion test FAILED with exception: {str(e)}")
            return False
    
    async def test_scenario_2_empty_sections(self) -> bool:
        """Test workflow completion with no generated sections"""
        logger.info("=== Testing Scenario 2: Empty Sections ===")
        
        try:
            # Create initial state with no generated sections
            initial_state: AgentState = {
                "workflow_id": "test_empty_002",
                "status": WorkflowStatus.CONSOLIDATING,
                "enterprise_info": {"name": "Empty Corp", "id": "ec002"},
                "selected_norms": ["ISO27001"],
                "generated_sections": {},  # Empty sections
                "messages": [],
                "errors": [],
                "react_steps": [],
                "updated_at": datetime.now().isoformat()
            }
            
            # Store the state
            self.manager.workflow_states["test_empty_002"] = initial_state.copy()
            
            # Progress the workflow
            result = await self.manager.progress_workflow("test_empty_002")
            
            # Check if minimal final report was created
            final_state = self.manager.workflow_states.get("test_empty_002", {})
            
            if final_state.get("final_report"):
                logger.info("✅ Empty sections test PASSED")
                logger.info(f"Minimal report created: {final_state['final_report'][:100]}...")
                return True
            else:
                logger.error("❌ Empty sections test FAILED - No final report created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Empty sections test FAILED with exception: {str(e)}")
            return False
    
    async def test_scenario_3_emergency_recovery(self) -> bool:
        """Test emergency recovery when completed workflow lacks final report"""
        logger.info("=== Testing Scenario 3: Emergency Recovery ===")
        
        try:
            # Create a completed state without final report (should not happen but test recovery)
            initial_state: AgentState = {
                "workflow_id": "test_emergency_003",
                "status": WorkflowStatus.COMPLETED,  # Already completed
                "enterprise_info": {"name": "Emergency Corp", "id": "ec003"},
                "selected_norms": ["ISO27001"],
                "generated_sections": {
                    "emergency_section": {
                        "analysis": "Emergency analysis",
                        "findings": ["Some findings"],
                        "recommendations": ["Some recommendations"]
                    }
                },
                "messages": [],
                "errors": [],
                "react_steps": [],
                "updated_at": datetime.now().isoformat()
                # Note: NO final_report field!
            }
            
            # Store the state
            self.manager.workflow_states["test_emergency_003"] = initial_state.copy()
            
            # Get workflow status (should trigger emergency report creation)
            status = await self.manager.get_workflow_status("test_emergency_003")
            
            # Check if emergency final report was created
            if status.get("final_report"):
                logger.info("✅ Emergency recovery test PASSED")
                logger.info(f"Emergency report created: {status['final_report'][:100]}...")
                return True
            else:
                logger.error("❌ Emergency recovery test FAILED - No emergency report created")
                logger.error(f"Status response: {status}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Emergency recovery test FAILED with exception: {str(e)}")
            return False
    
    async def test_scenario_4_api_endpoint_access(self) -> bool:
        """Test final report access through the API endpoint logic"""
        logger.info("=== Testing Scenario 4: API Endpoint Access ===")
        
        try:
            # Create a completed workflow with final report
            initial_state: AgentState = {
                "workflow_id": "test_api_004",
                "status": WorkflowStatus.COMPLETED,
                "enterprise_info": {"name": "API Corp", "id": "ac004"},
                "selected_norms": ["ISO27001"],
                "generated_sections": {
                    "api_section": {
                        "analysis": "API test analysis",
                        "findings": ["API findings"],
                        "recommendations": ["API recommendations"]
                    }
                },
                "final_report": "# Test Final Report\n\nThis is a test final report for API access.",
                "messages": [],
                "errors": [],
                "react_steps": [],
                "updated_at": datetime.now().isoformat()
            }
            
            # Store the state
            self.manager.workflow_states["test_api_004"] = initial_state.copy()
            
            # Simulate API endpoint logic
            status = await self.manager.get_workflow_status("test_api_004")
            
            # Check if final report is accessible
            if status.get("final_report"):
                logger.info("✅ API endpoint access test PASSED")
                logger.info(f"Report accessible via status: {len(status['final_report'])} characters")
                
                # Also test direct state access
                if "test_api_004" in self.manager.workflow_states:
                    direct_state = self.manager.workflow_states["test_api_004"]
                    if direct_state.get("final_report"):
                        logger.info("✅ Direct state access also works")
                        return True
                    else:
                        logger.error("❌ Direct state access failed")
                        return False
                else:
                    logger.error("❌ Workflow not found in state store")
                    return False
            else:
                logger.error("❌ API endpoint access test FAILED - Report not accessible")
                return False
                
        except Exception as e:
            logger.error(f"❌ API endpoint access test FAILED with exception: {str(e)}")
            return False
    
    async def test_scenario_5_react_steps_preservation(self) -> bool:
        """Test that ReAct steps are preserved and accessible"""
        logger.info("=== Testing Scenario 5: ReAct Steps Preservation ===")
        
        try:
            # Create state with ReAct steps
            initial_state: AgentState = {
                "workflow_id": "test_react_005",
                "status": WorkflowStatus.COMPLETED,
                "enterprise_info": {"name": "ReAct Corp", "id": "rc005"},
                "selected_norms": ["ISO27001"],
                "generated_sections": {"test_section": "test content"},
                "final_report": "# Test Report\n\nTest final report.",
                "react_steps": [
                    {"step": 1, "thought": "Starting analysis", "action": "call_analyzer", "observation": "Analysis complete"},
                    {"step": 2, "thought": "Writing report", "action": "call_writer", "observation": "Writing complete"},
                    {"step": 3, "thought": "Consolidating", "action": "call_consolidator", "observation": "Consolidation complete"}
                ],
                "messages": [],
                "errors": [],
                "updated_at": datetime.now().isoformat()
            }
            
            # Store the state
            self.manager.workflow_states["test_react_005"] = initial_state.copy()
            
            # Get status and check ReAct steps
            status = await self.manager.get_workflow_status("test_react_005")
            
            if status.get("react_steps") and len(status["react_steps"]) == 3:
                logger.info("✅ ReAct steps preservation test PASSED")
                logger.info(f"ReAct steps preserved: {len(status['react_steps'])} steps")
                return True
            else:
                logger.error("❌ ReAct steps preservation test FAILED")
                logger.error(f"Expected 3 steps, got: {len(status.get('react_steps', []))}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ReAct steps preservation test FAILED with exception: {str(e)}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all test scenarios"""
        logger.info("🚀 Starting Comprehensive Final Report Tests")
        logger.info("=" * 60)
        
        test_results = []
        
        # Run all test scenarios
        test_results.append(await self.test_scenario_1_normal_completion())
        test_results.append(await self.test_scenario_2_empty_sections())
        test_results.append(await self.test_scenario_3_emergency_recovery())
        test_results.append(await self.test_scenario_4_api_endpoint_access())
        test_results.append(await self.test_scenario_5_react_steps_preservation())
        
        # Summary
        passed = sum(test_results)
        total = len(test_results)
        
        logger.info("=" * 60)
        logger.info(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! The agentic audit workflow is robust.")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed. System needs improvement.")
            return False

async def main():
    """Main test function"""
    tester = ComprehensiveFinalReportTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
