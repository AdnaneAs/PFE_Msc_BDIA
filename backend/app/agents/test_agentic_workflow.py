import asyncio
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_dir)

from app.agents.workflow_manager import AuditWorkflowManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test configuration - using Gemini 2.0 Flash
llm_config = {
    "provider": "gemini",
    "model": "gemini-2.0-flash",
    "temperature": 0.0,
    "top_p": 0.8,
    "max_tokens": 2048,
    "api_key": os.getenv("GEMINI_API_KEY")
}

async def main():
    manager = AuditWorkflowManager(llm_config=llm_config)
    print("Starting agentic audit workflow test...")
    print(f"Using LLM: {llm_config['provider']}/{llm_config['model']}")
    print(f"Gemini API Key configured: {llm_config.get('api_key') is not None}")
    print(f"API Key (first 10 chars): {llm_config.get('api_key', 'None')[:10]}...")    
    # Test LLM connectivity first
    print("\n=== Testing LLM Connectivity ===")
    try:
        from app.services.llm_service import query_gemini_llm
        test_response, model_info = query_gemini_llm("Hello, can you respond with 'LLM is working'?", llm_config)
        print(f"LLM Test Response: {test_response}")
        print(f"Model Info: {model_info}")
    except Exception as e:
        print(f"LLM Connectivity Test FAILED: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    print("\n=== Starting Workflow ===")
    # Dummy test data - using an existing document ID from the database
    enterprise_report_id = "6"  # Use existing doc_id from the database
    selected_norms = ["CGI_Maroc"]
    user_id = "test_user"

    # Start workflow
    workflow_id = await manager.start_workflow(enterprise_report_id, selected_norms, user_id)
    print(f"Workflow started with ID: {workflow_id}")

    # Check status until plan is ready for approval
    while True:
        status = await manager.get_workflow_status(workflow_id)
        print(f"Current status: {status['status']}")
        if status.get("awaiting_human"):
            print("Audit plan is ready for approval.")
            print("Plan preview:", status.get("audit_plan"))
            answer = input("Approve plan? (y/n): ").strip().lower()
            approved = answer == 'y'
            result = await manager.continue_workflow(workflow_id, approval=approved)
            print(f"Plan {'approved' if approved else 'rejected'}. Workflow status: {result['status']}")
            if not approved:
                print("Plan was rejected. Exiting test.")
                return
            break
        await asyncio.sleep(1)    # Progress through all sections
    iteration_count = 0
    max_iterations = 20  # Prevent infinite loops
    
    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n--- Iteration {iteration_count} ---")
        
        status = await manager.get_workflow_status(workflow_id)
        print(f"Current status: {status['status']}, Current section: {status.get('current_section')}")
        
        # Print detailed state information
        print(f"Awaiting human: {status.get('awaiting_human')}")
        print(f"Generated sections: {list(status.get('generated_sections', []))}")
        print(f"Available audit sections: {status.get('audit_sections', [])}")
        
        # Print errors if any exist
        if status.get("errors"):
            print(f"ERRORS: {status.get('errors')}")
        
        # Print recent messages for debugging
        messages = status.get('messages', [])
        if messages:
            print(f"Last message: {messages[-1] if messages else 'None'}")
        
        if status['status'] in ["COMPLETED", "FAILED", "CANCELLED"]:
            print(f"\nWorkflow finished with status: {status['status']}")
            print("Final Errors:", status.get("errors"))
            print("Final Messages:", status.get("messages", [])[-3:] if status.get("messages") else "None")
            break
            
        # Progress workflow (simulate orchestrator step)
        try:
            print("Attempting to progress workflow...")
            result = await manager.progress_workflow(workflow_id)
            print(f"Progress result: {result['status']}, Section: {result.get('current_section')}")
            
            # Check if we're stuck in the same state
            if result['status'] == 'FAILED':
                print("ERROR: Workflow entered FAILED state during progression")
                print(f"Errors from progression: {result.get('errors')}")
                break
                
        except Exception as e:
            print(f"EXCEPTION during workflow progression: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            break
            
        await asyncio.sleep(2)  # Increased sleep time for better observation
    
    if iteration_count >= max_iterations:
        print(f"\nStopped after {max_iterations} iterations to prevent infinite loop")
        final_status = await manager.get_workflow_status(workflow_id)
        print(f"Final status: {final_status['status']}")
        print(f"Final errors: {final_status.get('errors')}")

if __name__ == "__main__":
    asyncio.run(main())
