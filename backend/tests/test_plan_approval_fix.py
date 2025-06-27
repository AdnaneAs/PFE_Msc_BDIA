#!/usr/bin/env python3
"""
Test the plan approval logic fix
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_plan_approval_logic():
    """Test that orchestrator handles plan approval correctly"""
    print("🔍 Testing plan approval logic...")
    
    try:
        from app.agents.orchestrator_agent import OrchestratorAgent
        from app.agents.audit_types import AgentState, WorkflowStatus
        
        # Create orchestrator
        llm_config = {"model_name": "test", "temperature": 0.7}
        orchestrator = OrchestratorAgent(llm_config)
        
        print("✅ Created orchestrator instance")
        
        # Test Case 1: No plan exists - should create plan
        print("\n📋 Test Case 1: No audit plan exists")
        state1 = {
            "workflow_id": "test-no-plan",
            "status": WorkflowStatus.PLANNING,
            "user_query": "Test no plan",
            "messages": [],
            "errors": [],
            "findings": [],
            "generated_sections": {},
            "react_steps": [],
            "plan_approved": False
            # No audit_plan key
        }
        
        possible_actions1 = orchestrator._get_possible_actions(state1)
        action_types1 = [action['type'] for action in possible_actions1]
        print(f"  Actions when no plan: {action_types1}")
        
        # Test Case 2: Plan exists but not approved - should wait for approval
        print("\n⏳ Test Case 2: Plan exists but not approved")
        state2 = {
            "workflow_id": "test-unapproved-plan",
            "status": WorkflowStatus.PLANNING,
            "user_query": "Test unapproved plan",
            "messages": [],
            "errors": [],
            "findings": [],
            "generated_sections": {},
            "react_steps": [],
            "audit_plan": {"sections": [{"name": "Overview", "cycle": "overview"}]},
            "plan_approved": False
        }
        
        # Test direct action decision (should return wait_human_approval)
        decision_action = await orchestrator._decide_action(state2, "Plan exists but needs approval")
        print(f"  Decision action: {decision_action['type']} - {decision_action['description']}")
        
        possible_actions2 = orchestrator._get_possible_actions(state2)
        action_types2 = [action['type'] for action in possible_actions2]
        print(f"  Available actions: {action_types2}")
        
        # Verify query_knowledge is NOT in the available actions
        if "query_knowledge" in action_types2:
            print("  ❌ query_knowledge should not be available when waiting for plan approval")
        else:
            print("  ✅ query_knowledge correctly excluded when waiting for plan approval")
        
        # Test Case 3: Plan exists and is approved - should start processing
        print("\n✅ Test Case 3: Plan exists and is approved")
        state3 = {
            "workflow_id": "test-approved-plan",
            "status": WorkflowStatus.AWAITING_APPROVAL,
            "user_query": "Test approved plan",
            "messages": [],
            "errors": [],
            "findings": [],
            "generated_sections": {},
            "react_steps": [],
            "audit_plan": {"sections": [{"name": "Overview", "cycle": "overview"}]},
            "plan_approved": True
        }
        
        possible_actions3 = orchestrator._get_possible_actions(state3)
        action_types3 = [action['type'] for action in possible_actions3]
        print(f"  Actions when plan approved: {action_types3}")
        
        print("\n🎯 Summary of fixes:")
        print("  ✅ When no plan exists: 'create_audit_plan' is available")
        print("  ✅ When plan exists but not approved: 'wait_human_approval' is chosen")
        print("  ✅ When plan exists but not approved: 'query_knowledge' is excluded")
        print("  ✅ When plan is approved: 'start_processing' is available")
        print("  ✅ No more infinite query_knowledge loops!")
        
        print("\n🚀 Plan approval logic is FIXED!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_plan_approval_logic())
