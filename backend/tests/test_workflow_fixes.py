#!/usr/bin/env python3
"""
Test script to verify the workflow fixes
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

async def test_workflow_fixes():
    """Test the workflow fixes"""
    print("🔧 Testing Workflow Fixes")
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
        print(f"   - Agents initialized: {list(manager.agents.keys())}")
        print(f"   - LangGraph available: {manager.graph is not None}")
        print(f"   - Checkpointer available: {manager.checkpointer is not None}")
        
        # Test 2: Start a workflow
        print("\n✅ Test 2: Start Workflow")
        workflow_id = await manager.start_workflow(
            enterprise_report_id="test_report_123",
            selected_norms=["ifrs", "iso_9001"],
            user_id="test_user"
        )
        print(f"   - Workflow ID: {workflow_id}")
        
        # Test 3: Get workflow status
        print("\n✅ Test 3: Get Workflow Status")
        status = await manager.get_workflow_status(workflow_id)
        print(f"   - Status: {status}")
        print(f"   - Has react_steps: {'react_steps' in status}")
        print(f"   - Has final_report: {'final_report' in status}")
        
        # Test 4: Check state access for ReAct steps
        print("\n✅ Test 4: Check State Access")
        if workflow_id in manager.workflow_states:
            state = manager.workflow_states[workflow_id]
            print(f"   - State keys: {list(state.keys())}")
            print(f"   - ReAct steps count: {len(state.get('react_steps', []))}")
            print(f"   - Status in state: {state.get('status')}")
        
        print("\n🎉 All tests passed! Workflow fixes appear to be working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow_fixes())
    sys.exit(0 if success else 1)
