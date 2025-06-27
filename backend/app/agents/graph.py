"""
LangGraph Studio Integration
============================

This module exports the audit workflow graph for LangGraph Studio visualization and debugging.
"""

import os
from typing import Dict, Any, TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from .audit_types import AgentState, WorkflowStatus, AgentType
from .workflow_manager import AuditWorkflowManager


class GraphState(TypedDict):
    """State definition for LangGraph Studio"""
    workflow_id: str
    status: WorkflowStatus
    enterprise_report_id: str
    selected_norms: List[str]
    audit_plan: Dict[str, Any]
    plan_approved: bool
    current_section: str
    analysis_context: Dict[str, Any]
    generated_sections: Dict[str, Any]
    final_report: str
    messages: Annotated[List[Dict[str, Any]], add_messages]
    errors: List[str]
    awaiting_human: bool
    created_at: str
    updated_at: str


def create_audit_graph() -> StateGraph:
    """Create and return the audit workflow graph for LangGraph Studio"""
    
    # Default LLM configuration for the graph
    llm_config = {
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    # Create workflow manager
    workflow_manager = AuditWorkflowManager(llm_config)
    
    # Build the graph
    builder = StateGraph(GraphState)
    
    # Add nodes for each agent
    builder.add_node("orchestrator", workflow_manager._call_orchestrator)
    builder.add_node("planner", workflow_manager._call_planner)
    builder.add_node("analyzer", workflow_manager._call_analyzer)
    builder.add_node("writer", workflow_manager._call_writer)
    builder.add_node("consolidator", workflow_manager._call_consolidator)
    
    # Set entry point
    builder.set_entry_point("orchestrator")
    
    # Add conditional edges based on workflow status
    def route_workflow(state: GraphState) -> str:
        """Route the workflow based on current status"""
        status = state["status"]
        
        if status == WorkflowStatus.PENDING:
            return "planner"
        elif status == WorkflowStatus.PLANNING:
            return "planner"
        elif status == WorkflowStatus.AWAITING_APPROVAL:
            if state.get("awaiting_human", False):
                return END
            else:
                return "orchestrator"
        elif status == WorkflowStatus.ANALYZING:
            return "analyzer"
        elif status == WorkflowStatus.WRITING:
            return "writer"
        elif status == WorkflowStatus.CONSOLIDATING:
            return "consolidator"
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            return END
        else:
            return "orchestrator"
    
    # Add conditional edges
    builder.add_conditional_edges("orchestrator", route_workflow)
    builder.add_conditional_edges("planner", route_workflow)
    builder.add_conditional_edges("analyzer", route_workflow)
    builder.add_conditional_edges("writer", route_workflow)
    builder.add_conditional_edges("consolidator", route_workflow)
    
    # Add checkpointer for persistence
    checkpointer = SqliteSaver.from_conn_string("./audit_workflow_checkpoints.db")
    
    # Compile the graph
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph


# Export the graph for LangGraph Studio
graph = create_audit_graph()
