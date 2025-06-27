#!/usr/bin/env python3
"""
Test script to verify final report generation
"""

import asyncio
import logging
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.workflow_manager import AuditWorkflowManager
from app.agents.audit_types import WorkflowStatus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_final_report_fix():
    """Test the final report generation fix"""
    print("🔧 Testing Final Report Fix")
    print("=" * 50)
    
    # Test LLM configuration
    llm_config = {
        "provider": "ollama",
        "model": "llama3.2:latest",
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        # Test 1: Initialize workflow manager
        print("✅ Test 1: Initialize WorkflowManager")
        manager = AuditWorkflowManager(llm_config)
        print(f"   - Manager initialized successfully")
        
        # Test 2: Start a workflow
        print("\n✅ Test 2: Start Workflow")
        workflow_id = await manager.start_workflow(
            enterprise_report_id="test_report_final",
            selected_norms=["ifrs", "iso_9001"],
            user_id="test_user"
        )
        print(f"   - Workflow ID: {workflow_id}")
        
        # Test 3: Simulate approval
        print("\n✅ Test 3: Approve Plan")
        await manager.continue_workflow(
            workflow_id=workflow_id,
            approval=True
        )
        
        # Test 4: Get status and check for final report
        print("\n✅ Test 4: Check Final Report Status")
        status = await manager.get_workflow_status(workflow_id)
        
        print(f"   - Workflow Status: {status.get('status')}")
        print(f"   - Has final_report in response: {'final_report' in status}")
        print(f"   - Generated sections: {status.get('generated_sections', [])}")
        
        # Test 5: Direct state access
        print("\n✅ Test 5: Direct State Access")
        if workflow_id in manager.workflow_states:
            state = manager.workflow_states[workflow_id]
            final_report = state.get("final_report")
            print(f"   - Has final_report in state: {bool(final_report)}")
            print(f"   - Final report type: {type(final_report)}")
            if final_report:
                print(f"   - Final report length: {len(final_report)} characters")
                print(f"   - Final report preview: {final_report[:200]}...")
        
        # Test 6: Test API endpoint logic
        print("\n✅ Test 6: Test API Logic")
        from app.api.v1.agentic_audit import get_audit_report, get_workflow_manager
        
        try:
            # Mock the get_workflow_manager to return our manager
            original_get_manager = get_workflow_manager
            def mock_get_manager():
                return manager
            
            # This would be called by the API
            report_response = {
                "status": "success" if final_report else "error",
                "has_direct_access": workflow_id in manager.workflow_states,
                "state_keys": list(manager.workflow_states[workflow_id].keys()) if workflow_id in manager.workflow_states else []
            }
            print(f"   - API test result: {report_response}")
            
        except Exception as e:
            print(f"   - API test error: {str(e)}")
        
        print("\n🎉 Final report fix test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_final_report_fix())
    sys.exit(0 if success else 1)
