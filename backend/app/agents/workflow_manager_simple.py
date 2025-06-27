"""
Simple Workflow Manager for Agentic Audit System
================================================

A simplified workflow manager that works without LangGraph dependencies.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .audit_types import AgentState, WorkflowStatus, AgentType
from .orchestrator_agent import OrchestratorAgent
from .planner_agent import PlannerAgent
from .analyzer_agent import AnalyzerAgent
from .writer_agent import WriterAgent
from .consolidator_agent import ConsolidatorAgent

logger = logging.getLogger(__name__)

class SimpleAuditWorkflowManager:
    """Simple workflow manager without LangGraph dependency"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize the workflow manager"""
        self.llm_config = llm_config
        self.agents = {}
        self.workflow_states = {}  # Simple in-memory state store
        
        # Initialize agents
        self._initialize_agents()
        logger.info("Simple workflow manager initialized")
    
    def _initialize_agents(self):
        """Initialize all agents with shared LLM config"""
        self.agents = {
            AgentType.ORCHESTRATOR: OrchestratorAgent(self.llm_config),
            AgentType.PLANNER: PlannerAgent(self.llm_config),
            AgentType.ANALYZER: AnalyzerAgent(self.llm_config),
            AgentType.WRITER: WriterAgent(self.llm_config),
            AgentType.CONSOLIDATOR: ConsolidatorAgent(self.llm_config)
        }
        logger.info(f"Initialized {len(self.agents)} agents for audit workflow")
    
    async def start_workflow(self, enterprise_report_id: str, selected_norms: List[str], 
                           user_id: str) -> str:
        """Start a new audit workflow"""
        workflow_id = f"audit_{uuid.uuid4().hex[:12]}"
        
        # Initialize state
        initial_state: AgentState = {
            "workflow_id": workflow_id,
            "status": WorkflowStatus.PLANNING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            
            "enterprise_report_id": enterprise_report_id,
            "selected_norms": selected_norms,
            "user_id": user_id,
            
            "audit_plan": None,
            "plan_approved": False,
            "approval_feedback": None,
            
            "current_section": None,
            "audit_sections": [],
            
            "retrieved_documents": [],
            "analysis_context": {},
            
            "generated_sections": {},
            "final_report": None,
            
            "messages": [],
            "errors": [],
            
            "human_feedback": None,
            "awaiting_human": False
        }
        
        # Store state
        self.workflow_states[workflow_id] = initial_state
        
        logger.info(f"Starting audit workflow {workflow_id}")
        
        # Start with planning phase
        try:
            # Execute planner to create audit plan
            planner_agent = self.agents[AgentType.PLANNER]
            updated_state = await planner_agent.execute(initial_state)
            
            # Mark as waiting for human approval
            updated_state["awaiting_human"] = True
            updated_state["updated_at"] = datetime.now().isoformat()
            
            # Update stored state
            self.workflow_states[workflow_id] = updated_state
            
            logger.info(f"Audit plan created for workflow {workflow_id}, awaiting approval")
            
        except Exception as e:
            logger.error(f"Error starting workflow {workflow_id}: {str(e)}")
            # Update state with error
            initial_state["status"] = WorkflowStatus.FAILED
            initial_state["errors"].append(f"Startup error: {str(e)}")
            self.workflow_states[workflow_id] = initial_state
        
        return workflow_id
    
    async def continue_workflow(self, workflow_id: str, human_feedback: Optional[str] = None,
                              approval: Optional[bool] = None) -> Dict[str, Any]:
        """Continue a workflow after human input"""
        if workflow_id not in self.workflow_states:
            return {"error": "Workflow not found"}
        
        state = self.workflow_states[workflow_id].copy()
        
        # Update with human input
        if human_feedback:
            state["human_feedback"] = human_feedback
        
        if approval is not None:
            state["plan_approved"] = approval
            state["awaiting_human"] = False
            state["updated_at"] = datetime.now().isoformat()
        
        if approval is False:
            # Plan rejected, mark as cancelled
            state["status"] = WorkflowStatus.CANCELLED
            self.workflow_states[workflow_id] = state
            return {"status": "cancelled", "message": "Plan rejected by user"}
        
        if approval is True:
            # Plan approved, continue with analysis
            state["status"] = WorkflowStatus.ANALYZING
            
            try:
                # Start analysis phase
                orchestrator = self.agents[AgentType.ORCHESTRATOR]
                updated_state = await orchestrator.execute(state)
                self.workflow_states[workflow_id] = updated_state
                
                return {"status": "continued", "message": "Workflow continued after approval"}
                
            except Exception as e:
                logger.error(f"Error continuing workflow {workflow_id}: {str(e)}")
                state["status"] = WorkflowStatus.FAILED
                state["errors"].append(f"Continuation error: {str(e)}")
                self.workflow_states[workflow_id] = state
                return {"error": f"Failed to continue workflow: {str(e)}"}
        
        # Just update state without changing status
        self.workflow_states[workflow_id] = state
        return {"status": "updated", "message": "Workflow state updated"}
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflow_states:
            return {"error": "Workflow not found"}
        
        state = self.workflow_states[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": state.get("status", WorkflowStatus.PENDING).value if hasattr(state.get("status"), "value") else state.get("status"),
            "updated_at": state.get("updated_at"),
            "awaiting_human": state.get("awaiting_human", False),
            "audit_plan": state.get("audit_plan"),
            "generated_sections": list(state.get("generated_sections", {}).keys()),
            "current_section": state.get("current_section"),
            "errors": state.get("errors", [])
        }
