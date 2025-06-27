#!/usr/bin/env python3
"""
Quick verification script for final report generation improvements.
This script performs basic checks to ensure the improvements are working.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.base_agent import AgentState, WorkflowStatus
from app.config import get_llm_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simple_final_report_creation():
    """Test the _create_simple_final_report method directly"""
    logger.info("Testing simple final report creation...")
    
    try:
        # Initialize orchestrator
        llm_config = get_llm_config()
        orchestrator = OrchestratorAgent(llm_config)
        
        # Create test state
        state: AgentState = {
            "workflow_id": "test_001",
            "status": WorkflowStatus.CONSOLIDATING,
            "enterprise_info": {"name": "Test Enterprise", "id": "te001"},
            "selected_norms": ["ISO27001", "GDPR"],
            "generated_sections": {
                "security_assessment": {
                    "analysis": "Security controls evaluation completed",
                    "findings": [
                        "Multi-factor authentication is implemented",
                        "Network segmentation needs improvement",
                        "Data encryption is properly configured"
                    ],
                    "recommendations": [
                        "Implement network micro-segmentation",
                        "Conduct regular penetration testing",
                        "Update security policies"
                    ]
                },
                "privacy_compliance": {
                    "analysis": "GDPR compliance assessment performed",
                    "findings": [
                        "Data processing agreements are in place",
                        "Privacy notice needs updates",
                        "Data retention policies are documented"
                    ],
                    "recommendations": [
                        "Update privacy notices to latest requirements",
                        "Implement automated data retention",
                        "Conduct privacy impact assessments"
                    ]
                }
            },
            "messages": [],
            "errors": [],
            "react_steps": [],
            "updated_at": datetime.now().isoformat()
        }
        
        # Test the simple final report creation
        final_report = orchestrator._create_simple_final_report(state, state["generated_sections"])
        
        # Verify the report
        if final_report and len(final_report) > 100:
            logger.info("✅ Simple final report creation test PASSED")
            logger.info(f"Report length: {len(final_report)} characters")
            logger.info("Report preview:")
            logger.info(final_report[:300] + "...")
            return True
        else:
            logger.error("❌ Simple final report creation test FAILED")
            logger.error(f"Report: {final_report}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple final report creation test FAILED with exception: {str(e)}")
        return False

async def test_complete_workflow_action():
    """Test the complete_workflow action to ensure it creates a final report"""
    logger.info("Testing complete workflow action...")
    
    try:
        # Initialize orchestrator
        llm_config = get_llm_config()
        orchestrator = OrchestratorAgent(llm_config)
        
        # Create test state without final report
        state: AgentState = {
            "workflow_id": "test_002",
            "status": WorkflowStatus.CONSOLIDATING,
            "enterprise_info": {"name": "Complete Test Corp", "id": "ctc002"},
            "selected_norms": ["ISO27001"],
            "generated_sections": {
                "test_section": {
                    "analysis": "Test analysis completed",
                    "findings": ["Test finding 1", "Test finding 2"],
                    "recommendations": ["Test recommendation 1"]
                }
            },
            "messages": [],
            "errors": [],
            "react_steps": [],
            "updated_at": datetime.now().isoformat()
            # Note: NO final_report field initially
        }
        
        # Execute complete_workflow action
        action = {
            "type": "complete_workflow",
            "description": "Test workflow completion",
            "parameters": {}
        }
        
        result_state = await orchestrator._execute_action(state, action)
        
        # Check if final report was created
        if result_state.get("final_report") and result_state.get("status") == WorkflowStatus.COMPLETED:
            logger.info("✅ Complete workflow action test PASSED")
            logger.info(f"Final report created: {len(result_state['final_report'])} characters")
            logger.info(f"Status: {result_state['status']}")
            return True
        else:
            logger.error("❌ Complete workflow action test FAILED")
            logger.error(f"Has final report: {bool(result_state.get('final_report'))}")
            logger.error(f"Status: {result_state.get('status')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Complete workflow action test FAILED with exception: {str(e)}")
        return False

async def test_empty_sections_fallback():
    """Test fallback when no sections are generated"""
    logger.info("Testing empty sections fallback...")
    
    try:
        # Initialize orchestrator
        llm_config = get_llm_config()
        orchestrator = OrchestratorAgent(llm_config)
        
        # Create test state with no generated sections
        state: AgentState = {
            "workflow_id": "test_003",
            "status": WorkflowStatus.CONSOLIDATING,
            "enterprise_info": {"name": "Empty Test Corp", "id": "etc003"},
            "selected_norms": ["ISO27001"],
            "generated_sections": {},  # Empty!
            "messages": [],
            "errors": [],
            "react_steps": [],
            "updated_at": datetime.now().isoformat()
        }
        
        # Execute complete_workflow action
        action = {
            "type": "complete_workflow",
            "description": "Test workflow completion with empty sections",
            "parameters": {}
        }
        
        result_state = await orchestrator._execute_action(state, action)
        
        # Check if minimal final report was created
        if (result_state.get("final_report") and 
            result_state.get("status") == WorkflowStatus.COMPLETED and
            "no sections were generated" in result_state["final_report"]):
            logger.info("✅ Empty sections fallback test PASSED")
            logger.info(f"Minimal report: {result_state['final_report']}")
            return True
        else:
            logger.error("❌ Empty sections fallback test FAILED")
            logger.error(f"Final report: {result_state.get('final_report', 'None')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Empty sections fallback test FAILED with exception: {str(e)}")
        return False

async def main():
    """Run quick verification tests"""
    logger.info("🚀 Running Quick Final Report Verification Tests")
    logger.info("=" * 50)
    
    tests = [
        test_simple_final_report_creation(),
        test_complete_workflow_action(),
        test_empty_sections_fallback()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Count successes
    passed = sum(1 for result in results if result is True)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"📊 QUICK TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All quick tests passed! Core functionality is working.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed.")
        for i, result in enumerate(results):
            if result is not True:
                logger.error(f"Test {i+1} result: {result}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
