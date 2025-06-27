#!/usr/bin/env python3

import sys
sys.path.append('.')

print("Testing workflow manager import...")

try:
    from app.agents.workflow_manager import AuditWorkflowManager
    print("✅ AuditWorkflowManager imported successfully")
    
    # Test initialization
    config = {"provider": "ollama", "model": "llama3.2"}
    manager = AuditWorkflowManager(config)
    print("✅ Manager initialized successfully")
    print(f"Graph available: {manager.graph is not None}")
    print(f"Checkpointer available: {manager.checkpointer is not None}")
    print(f"Number of agents: {len(manager.agents)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
