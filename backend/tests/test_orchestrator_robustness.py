#!/usr/bin/env python3
"""
Test script to verify orchestrator robustness and action handling
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.audit_types import AgentState, WorkflowStatus, AuditCycle

async def test_invalid_action_handling():
    """Test that orchestrator handles invalid actions gracefully"""
    print("Testing orchestrator robustness...")
    
    # Create test state
    state = AgentState(
        workflow_id="test-robustness",
        status=WorkflowStatus.PLANNING,
        user_query="Test robustness",
        messages=[],
        errors=[],
        findings=[],
        current_section=AuditCycle.VENTES_CLIENTS.value,
        generated_sections={},
        react_steps=[]
    )
      # Create orchestrator with basic config
    llm_config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 2000
    }
    orchestrator = OrchestratorAgent(llm_config)
    
    # Test various scenarios
    test_cases = [
        {
            "name": "Invalid action type",
            "action": {"type": "N/A", "description": "Invalid action", "parameters": {}},
            "expected": "Should handle gracefully and not get stuck"
        },
        {
            "name": "None action type", 
            "action": {"type": None, "description": "None action", "parameters": {}},
            "expected": "Should handle gracefully"
        },
        {
            "name": "Missing action type",
            "action": {"description": "Missing type", "parameters": {}},
            "expected": "Should handle gracefully"
        },
        {
            "name": "Empty action",
            "action": {},
            "expected": "Should handle gracefully"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        try:
            result_state = await orchestrator._execute_action(state.copy(), test_case["action"])
            
            # Check that workflow didn't get stuck
            if result_state["status"] == WorkflowStatus.ERROR:
                print("❌ Workflow ended in ERROR status")
            elif result_state["status"] == WorkflowStatus.COMPLETED:
                print("✅ Workflow completed gracefully")
            else:
                print(f"⚠️  Workflow in status: {result_state['status']}")
                
            # Check for error messages
            if result_state.get("errors"):
                print(f"   Errors logged: {len(result_state['errors'])}")
                print(f"   Last error: {result_state['errors'][-1]['message']}")
            
            # Check for final report
            if result_state.get("final_report"):
                print(f"   Final report created: {len(result_state['final_report'])} characters")
            
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
    
    # Test action decision with invalid responses
    print(f"\n--- Testing action decision robustness ---")
    try:
        # Test fallback action selection
        fallback_action = orchestrator._get_fallback_action(state)
        print(f"✅ Fallback action: {fallback_action['type']}")
        
        # Test possible actions
        possible_actions = orchestrator._get_possible_actions(state)
        print(f"✅ Possible actions for PLANNING: {[a['type'] for a in possible_actions]}")
        
        # Test with different statuses
        for status in [WorkflowStatus.PROCESSING, WorkflowStatus.APPROVED, WorkflowStatus.COMPLETED]:
            test_state = state.copy()
            test_state["status"] = status
            actions = orchestrator._get_possible_actions(test_state)
            print(f"✅ Possible actions for {status.value}: {[a['type'] for a in actions]}")
            
    except Exception as e:
        print(f"❌ Error in action decision testing: {str(e)}")
    
    print("\n--- Test Summary ---")
    print("✅ Orchestrator robustness test completed")
    print("✅ All invalid actions should be handled gracefully")
    print("✅ Workflow should never get stuck in infinite loops")

if __name__ == "__main__":
    asyncio.run(test_invalid_action_handling())
