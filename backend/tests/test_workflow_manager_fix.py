"""
Test script to verify the workflow manager fixes for recursion limit and human approval handling.
This tests the specific fixes made to prevent infinite loops and recursion limit errors.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.agents.workflow_manager import AuditWorkflowManager
from app.agents.audit_types import WorkflowStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_workflow_manager_fixes():
    """Test the workflow manager fixes"""
    logger.info("=== Testing Workflow Manager Fixes ===")
    
    # Test configuration
    llm_config = {
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp",
        "api_key": "test-key",  # Using test key for this test
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        # Initialize workflow manager
        logger.info("1. Initializing AuditWorkflowManager...")
        workflow_manager = AuditWorkflowManager(llm_config)
        logger.info("✓ AuditWorkflowManager initialized successfully")
        
        # Test 1: Verify routing logic with awaiting_human=True
        logger.info("\n2. Testing routing logic for awaiting_human=True...")
        test_state = {
            "workflow_id": "test-123",
            "status": WorkflowStatus.AWAITING_APPROVAL,
            "awaiting_human": True,
            "enterprise_report_id": "test-report",
            "selected_norms": ["ISO 27001"],
            "plan_approved": False
        }
        
        route = workflow_manager._route_from_orchestrator(test_state)
        if route == "end":
            logger.info("✓ Routing correctly stops execution when awaiting_human=True")
        else:
            logger.error(f"✗ Routing should return 'end' for awaiting_human=True, got: {route}")
        
        # Test 2: Verify routing logic with awaiting_human=False
        logger.info("\n3. Testing routing logic for awaiting_human=False...")
        test_state["awaiting_human"] = False
        route = workflow_manager._route_from_orchestrator(test_state)
        if route != "end":  # Should route to appropriate agent, not end
            logger.info(f"✓ Routing continues execution when awaiting_human=False: {route}")
        else:
            logger.error("✗ Routing should not end when awaiting_human=False")
        
        # Test 3: Verify different status routing
        logger.info("\n4. Testing routing for different workflow statuses...")
        test_cases = [
            (WorkflowStatus.PLANNING, "planner"),
            (WorkflowStatus.ANALYZING, "analyzer"),
            (WorkflowStatus.WRITING, "writer"),
            (WorkflowStatus.CONSOLIDATING, "consolidator"),
            (WorkflowStatus.COMPLETED, "end"),
            (WorkflowStatus.FAILED, "end")
        ]
        
        for status, expected_route in test_cases:
            test_state["status"] = status
            test_state["awaiting_human"] = False
            route = workflow_manager._route_from_orchestrator(test_state)
            if route == expected_route:
                logger.info(f"✓ Status {status.value} routes to {route}")
            else:
                logger.error(f"✗ Status {status.value} should route to {expected_route}, got {route}")
        
        # Test 4: Verify graph compilation with recursion protection
        logger.info("\n5. Testing graph compilation...")
        if workflow_manager.graph is not None:
            logger.info("✓ LangGraph workflow compiled successfully")
        else:
            logger.warning("! LangGraph not available, using fallback workflow")
        
        # Test 5: Test continue_workflow_async logic
        logger.info("\n6. Testing continue_workflow_async...")
        
        # First create a workflow state
        workflow_id = f"test-workflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow_manager.workflow_states[workflow_id] = {
            "workflow_id": workflow_id,
            "status": WorkflowStatus.AWAITING_APPROVAL,
            "awaiting_human": True,
            "plan_approved": False,
            "human_feedback": None,
            "messages": []
        }
        
        # Test approval
        result = await workflow_manager.continue_workflow_async(workflow_id, approval=True)
        if result.get("plan_approved") and not result.get("awaiting_human"):
            logger.info("✓ continue_workflow_async correctly handles approval")
        else:
            logger.error(f"✗ continue_workflow_async approval handling failed: {result}")
        
        # Test rejection
        result = await workflow_manager.continue_workflow_async(workflow_id, approval=False)
        if not result.get("plan_approved"):
            logger.info("✓ continue_workflow_async correctly handles rejection")
        else:
            logger.error(f"✗ continue_workflow_async rejection handling failed: {result}")
        
        logger.info("\n=== Workflow Manager Fix Tests Completed ===")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow_manager_fixes())
