"""
LangGraph Workflow for Agentic Audit System
===========================================

Implements the LangGraph workflow that orchestrates the multi-agent audit 
conformity report generation system. Manages state transitions and agent
coordination using a graph-based approach.
"""

import uuid
from typing import Dict, Any, List, Optional, Annotated
from datetime import datetime
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    # Try different import paths for SqliteSaver based on LangGraph version
    SqliteSaver = None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            try:
                from langgraph.checkpoint.memory import MemorySaver as SqliteSaver
            except ImportError:
                pass
except ImportError:
    # Fallback if LangGraph is not installed
    StateGraph = None
    END = "END"
    add_messages = None
    SqliteSaver = None

import os
import logging
from .audit_types import AgentState, WorkflowStatus, AgentType
from .orchestrator_agent import OrchestratorAgent
from .planner_agent import PlannerAgent
from .analyzer_agent import AnalyzerAgent
from .writer_agent import WriterAgent
from .consolidator_agent import ConsolidatorAgent

logger = logging.getLogger(__name__)

# Choose state manager based on environment
if os.getenv('REDIS_URL'):
    from ..services.redis_state_service import redis_state_manager as state_manager
    logger.info(" Using Redis state manager for Docker deployment")
else:
    from ..services.shared_state_service import shared_state_manager as state_manager
    logger.info(" Using file-based state manager for local deployment")

logger = logging.getLogger(__name__)

class AuditWorkflowManager:
    """Manages the LangGraph workflow for audit report generation"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize the workflow manager
        
        Args:
            llm_config: LLM configuration for all agents
        """
        self.llm_config = llm_config
        self.graph = None
        self.checkpointer = None
        self.agents = {}
        
        # Simple fallback state store when LangGraph is not available
        self.workflow_states = {}
        
        # Background task executor for async workflow processing
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="workflow-")
        
        # Track background processing tasks
        self.background_tasks = {}        
        # Initialize agents
        self._initialize_agents()
        
        # Build workflow graph
        self._build_workflow_graph()
    
    def _initialize_agents(self):
        """Initialize all agents with shared LLM config"""
        logger.info("Starting agent initialization...")
        self.agents = {}
        
        # Initialize each agent with error handling
        try:
            logger.info("Creating OrchestratorAgent...")
            self.agents[AgentType.ORCHESTRATOR] = OrchestratorAgent(self.llm_config)
            logger.info("OrchestratorAgent created successfully")
        except Exception as e:
            logger.error(f"Failed to create OrchestratorAgent: {str(e)}")
        
        try:
            logger.info("Creating PlannerAgent...")
            self.agents[AgentType.PLANNER] = PlannerAgent(self.llm_config)
            logger.info("PlannerAgent created successfully")
        except Exception as e:
            logger.error(f"Failed to create PlannerAgent: {str(e)}")
        
        try:
            logger.info("Creating AnalyzerAgent...")
            self.agents[AgentType.ANALYZER] = AnalyzerAgent(self.llm_config)
            logger.info("AnalyzerAgent created successfully")
        except Exception as e:
            logger.error(f"Failed to create AnalyzerAgent: {str(e)}")
        
        try:
            logger.info("Creating WriterAgent...")
            self.agents[AgentType.WRITER] = WriterAgent(self.llm_config)
            logger.info("WriterAgent created successfully")
        except Exception as e:
            logger.error(f"Failed to create WriterAgent: {str(e)}")
        
        try:
            logger.info("Creating ConsolidatorAgent...")
            self.agents[AgentType.CONSOLIDATOR] = ConsolidatorAgent(self.llm_config)
            logger.info("ConsolidatorAgent created successfully")
        except Exception as e:
            logger.error(f"Failed to create ConsolidatorAgent: {str(e)}")
        logger.info(f"Initialized {len(self.agents)} agents for audit workflow")
        logger.info(f"Available agents: {list(self.agents.keys())}")
    
    def _build_workflow_graph(self):
        """Build the LangGraph workflow"""
        if StateGraph is None:
            logger.warning("LangGraph not available, using fallback workflow")
            return
        
        try:
            # Initialize checkpointer for state persistence
            self.checkpointer = None
            
            # Try to use SqliteSaver first
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                self.checkpointer = SqliteSaver.from_conn_string(":memory:")
                logger.info("SQLite checkpointer initialized")
            except ImportError:
                # Fallback to MemorySaver
                try:
                    from langgraph.checkpoint.memory import MemorySaver
                    self.checkpointer = MemorySaver()
                    logger.info("Memory checkpointer initialized")
                except ImportError:
                    logger.warning("No checkpointer available, state will not persist")
                    self.checkpointer = None
            
            # Create state graph
            workflow = StateGraph(AgentState)
            
            # Add nodes for each agent
            workflow.add_node("orchestrator", self._orchestrator_node)
            workflow.add_node("planner", self._planner_node)
            workflow.add_node("analyzer", self._analyzer_node)
            workflow.add_node("writer", self._writer_node)
            workflow.add_node("consolidator", self._consolidator_node)
            workflow.add_node("human_approval", self._human_approval_node)
            
            # Define edges based on workflow status
            workflow.add_conditional_edges(
                "orchestrator",
                self._route_from_orchestrator,
                {
                    "planner": "planner",
                    "analyzer": "analyzer", 
                    "writer": "writer",
                    "consolidator": "consolidator",
                    "human_approval": "human_approval",
                    "end": END
                }
            )            # Add edges from other nodes back to orchestrator
            workflow.add_edge("planner", "orchestrator")
            workflow.add_edge("analyzer", "orchestrator")
            workflow.add_edge("writer", "orchestrator")
            workflow.add_edge("consolidator", "orchestrator")
            # NOTE: human_approval does NOT go back to orchestrator automatically
            # It should wait for external input and only continue when explicitly resumed
            
            # Set entry point
            workflow.set_entry_point("orchestrator")
            
            # Compile with checkpointer
            if self.checkpointer:
                self.graph = workflow.compile(checkpointer=self.checkpointer)
            else:
                self.graph = workflow.compile()
            
            logger.info("LangGraph workflow compiled successfully")
        except Exception as e:
            logger.error(f"Failed to build LangGraph workflow: {str(e)}")
            self.graph = None
    
    async def start_workflow(self, enterprise_report_id: str, selected_norms: List[str], 
                           user_id: str) -> str:
        """
        Start a new audit workflow
        
        Args:
            enterprise_report_id: ID of the uploaded enterprise report
            selected_norms: List of conformity norms to check against
            user_id: ID of the user starting the workflow
            
        Returns:
            Workflow ID for tracking
        """
        workflow_id = f"audit_{uuid.uuid4().hex[:12]}"
        
        # Initialize state
        initial_state: AgentState = {
            "workflow_id": workflow_id,
            "status": WorkflowStatus.PENDING,
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
            "awaiting_human": False        }
        logger.info(f"Starting audit workflow {workflow_id}") 
        # Store state in fallback store with robust persistence
        self._ensure_state_persistence(workflow_id, initial_state)
        
        if self.graph:
            # Use LangGraph if available
            try:
                config = {"configurable": {"thread_id": workflow_id}}
                logger.info(f"WORKFLOW_START: Using LangGraph for workflow {workflow_id}")
                result = await self.graph.ainvoke(initial_state, config)
                logger.info(f"WORKFLOW_START: LangGraph completed for workflow {workflow_id}")
                return workflow_id
            except AttributeError as e:
                if "'_GeneratorContextManager' object has no attribute" in str(e):
                    logger.error(f"LangGraph version compatibility issue: {str(e)}")
                    logger.info("Falling back to manual orchestration due to LangGraph compatibility issue")
                else:
                    logger.error(f"LangGraph AttributeError: {str(e)}")
                # Fall back to manual orchestration
            except Exception as e:
                logger.error(f"LangGraph execution failed: {str(e)}")
                logger.info(f"WORKFLOW_START: LangGraph failed for workflow {workflow_id}, falling back to manual orchestration")
                # Fall back to manual orchestration
        else:
            logger.info(f"WORKFLOW_START: LangGraph not available for workflow {workflow_id}, using manual orchestration")
        
        # Fallback to manual orchestration
        logger.info(f"WORKFLOW_START: Starting manual orchestration for workflow {workflow_id}")
        await self._manual_orchestration(initial_state)
        logger.info(f"WORKFLOW_START: Manual orchestration completed for workflow {workflow_id}")
        return workflow_id
    async def continue_workflow(self, workflow_id: str, human_feedback: Optional[str] = None,
                              approval: Optional[bool] = None) -> Dict[str, Any]:
        """
        Continue a workflow after human input
        
        This method now delegates to continue_workflow_async for better performance
        
        Args:
            workflow_id: ID of the workflow to continue
            human_feedback: Optional human feedback
            approval: Optional approval decision
            
        Returns:
            Updated workflow state
        """
        # Use the new async processing method for better performance
        return await self.continue_workflow_async(workflow_id, human_feedback, approval)
    
    async def continue_workflow_async(self, workflow_id: str, human_feedback: Optional[str] = None,
                                    approval: Optional[bool] = None) -> Dict[str, Any]:
        """
        Continue a workflow after human input with immediate response and async processing
        
        Args:
            workflow_id: ID of the workflow to continue
            human_feedback: Optional human feedback
            approval: Optional approval decision
              Returns:
            Immediate response dict, actual processing happens in background
        """
        logger.info(f"Starting async workflow continuation for {workflow_id}")
        
        # First, try to load state from state manager (Redis or file-based)
        logger.info(f" Attempting to load workflow {workflow_id} from state manager...")
        shared_state = state_manager.get_workflow_state(workflow_id)
        if shared_state:
            state = shared_state.copy()
            logger.info(f" Loaded workflow {workflow_id} from state manager")
            # Also update local cache
            self.workflow_states[workflow_id] = state.copy()
        elif workflow_id in self.workflow_states:
            # Fallback to local state
            state = self.workflow_states[workflow_id].copy()
            logger.info(f" Loaded workflow {workflow_id} from local state (fallback)")
        else:
            logger.error(f" Workflow {workflow_id} not found in state manager OR local state")
            return {"error": "Workflow not found"}
        
        if not state:
            logger.error(f" Failed to load state for workflow {workflow_id}")
            return {"error": "Workflow not found"}
        
        # Update with human input immediately
        if human_feedback:
            state["human_feedback"] = human_feedback
            logger.info(f"Added human feedback: {human_feedback}")
        
        if approval is not None:
            state["plan_approved"] = approval
            state["awaiting_human"] = False
            state["updated_at"] = datetime.now().isoformat()
            logger.info(f"Plan approval set to: {approval}")
            
            if approval:
                # Set to processing status immediately
                state["status"] = WorkflowStatus.ANALYZING
                state["messages"].append(f"Plan approved at {datetime.now().isoformat()}, starting analysis...")
                logger.info("Workflow approved, setting status to ANALYZING")
            else:
                # Back to planning
                state["status"] = WorkflowStatus.PLANNING
                state["messages"].append(f"Plan rejected at {datetime.now().isoformat()}, back to planning")
                logger.info("Workflow rejected, back to planning phase")
          # Update stored state immediately (both local and state manager)
        self.workflow_states[workflow_id] = state.copy()
        
        # Also persist to state manager for multi-worker support
        try:
            success = state_manager.store_workflow_state(workflow_id, state.copy())
            if success:
                logger.info(f" Persisted workflow {workflow_id} to state manager")
            else:
                logger.warning(f" Failed to persist workflow {workflow_id} to state manager")
        except Exception as e:
            logger.error(f" Error persisting to state manager: {str(e)}")
        
        # Schedule background processing if approved
        if approval:
            logger.info("Scheduling background workflow processing...")
            # Use asyncio.create_task to run in background without blocking
            task = asyncio.create_task(self._background_workflow_processing(workflow_id))
            self.background_tasks[workflow_id] = task
        
        # Return immediate response with transition signals
        response = {
            "status": "success",
            "workflow_id": workflow_id,
            "approved": approval,
            "message": "Audit plan approved successfully" if approval else "Audit plan processing updated",
            "feedback_provided": bool(human_feedback),
            "immediate_response": True,
            "background_processing_started": bool(approval),
            "workflow_status": state.get("status").value if hasattr(state.get("status"), "value") else state.get("status"),
            "updated_at": state.get("updated_at"),
            "transition_to_processing": bool(approval),
            "processing_message": "Workflow processing has started in the background. You can monitor progress via the status endpoint." if approval else None,
            "redirect_to": "processing" if approval else None        }
        
        return response

    async def _background_workflow_processing(self, workflow_id: str):
        """        Background workflow processing after approval
        
        Args:
            workflow_id: ID of the workflow to process
        """
        logger.info(f"Starting background processing for workflow {workflow_id}")
        
        try:
            # Load state from state manager first (for multi-worker support)
            state = None
            
            shared_state = state_manager.get_workflow_state(workflow_id)
            if shared_state:
                state = shared_state.copy()
                logger.info(f" Loaded workflow {workflow_id} from state manager for background processing")
                # Update local cache
                self.workflow_states[workflow_id] = state.copy()
            elif workflow_id in self.workflow_states:
                state = self.workflow_states[workflow_id].copy()
                logger.info(f" Using local state for background processing of {workflow_id}")
            else:
                logger.error(f" Workflow {workflow_id} not found for background processing")
                return
            
            # Add processing indicator
            state["messages"].append(f"Background processing started at {datetime.now().isoformat()}")
            state["background_processing"] = True
            
            # Update both local and state manager
            self.workflow_states[workflow_id] = state.copy()
            state_manager.store_workflow_state(workflow_id, state.copy())
            
            # Use orchestrator agent to make ReAct-based decisions
            logger.info("Executing orchestrator agent in background...")
            
            # Run multiple orchestrator cycles to complete the workflow
            max_cycles = 15
            cycle_count = 0
            
            while (cycle_count < max_cycles and 
                   state.get("status") not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]):
                
                cycle_count += 1
                logger.info(f"Background processing cycle {cycle_count} for workflow {workflow_id}")
                
                try:
                    orchestrator_agent = self.agents[AgentType.ORCHESTRATOR]
                    state = await orchestrator_agent.execute(state)
                      # Update stored state after each cycle
                    state["updated_at"] = datetime.now().isoformat()
                    state["messages"].append(f"Cycle {cycle_count} completed at {datetime.now().isoformat()}")
                    state["background_processing"] = True
                    
                    # CRITICAL: Persist to both local cache and state manager after every cycle
                    self.workflow_states[workflow_id] = state.copy()
                    try:
                        success = state_manager.store_workflow_state(workflow_id, state.copy())
                        if success:
                            logger.debug(f"✓ Cycle {cycle_count}: State persisted to shared manager")
                        else:
                            logger.warning(f"⚠ Cycle {cycle_count}: Failed to persist to shared manager")
                    except Exception as e:
                        logger.error(f"❌ Cycle {cycle_count}: Error persisting to state manager: {str(e)}")
                    
                    logger.info(f"Cycle {cycle_count} completed. Status: {state.get('status')}, Current section: {state.get('current_section')}")
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(1)
                    
                    # Check if we're waiting for human input (shouldn't happen in background)
                    if state.get("awaiting_human", False):
                        logger.warning(f"Background processing stopped - workflow {workflow_id} is awaiting human input")
                        break
                    
                except Exception as e:
                    logger.error(f"Error in background processing cycle {cycle_count}: {str(e)}")
                    state["status"] = WorkflowStatus.FAILED
                    state["errors"].append(f"Background processing error in cycle {cycle_count}: {str(e)}")
                    break
              # Final state update with complete synchronization
            if cycle_count >= max_cycles and state.get("status") not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                logger.warning(f"Background processing reached max cycles ({max_cycles}) for workflow {workflow_id}")
                state["status"] = WorkflowStatus.FAILED
                state["errors"].append(f"Background processing timed out after {max_cycles} cycles")
            
            state["updated_at"] = datetime.now().isoformat()
            state["messages"].append(f"Background processing completed at {datetime.now().isoformat()}")
            state["background_processing"] = False
            
            # CRITICAL: Final state persistence to both local cache and state manager
            self.workflow_states[workflow_id] = state.copy()
            try:
                success = state_manager.store_workflow_state(workflow_id, state.copy())
                if success:
                    logger.info(f"✓ Final state persisted to shared manager for workflow {workflow_id}")
                else:
                    logger.error(f"❌ Failed to persist final state to shared manager for workflow {workflow_id}")
            except Exception as e:
                logger.error(f"❌ Error persisting final state to state manager: {str(e)}")
            
            logger.info(f"Background processing completed for workflow {workflow_id}. Final status: {state.get('status')}")
            
            # Clean up background task reference
            if workflow_id in self.background_tasks:
                del self.background_tasks[workflow_id]
            
        except Exception as e:
            logger.error(f"Critical error in background workflow processing: {str(e)}")
            logger.exception("Full traceback:")
              # Update state with error and ensure persistence
            if workflow_id in self.workflow_states:
                state = self.workflow_states[workflow_id].copy()
                state["status"] = WorkflowStatus.FAILED
                state["errors"].append(f"Critical background processing error: {str(e)}")
                state["updated_at"] = datetime.now().isoformat()
                state["background_processing"] = False
                
                # Persist error state to both local cache and state manager
                self.workflow_states[workflow_id] = state.copy()
                try:
                    state_manager.store_workflow_state(workflow_id, state.copy())
                    logger.info(f"✓ Error state persisted to shared manager for workflow {workflow_id}")
                except Exception as persist_error:
                    logger.error(f"❌ Failed to persist error state: {str(persist_error)}")
            
            # Clean up background task reference
            if workflow_id in self.background_tasks:
                del self.background_tasks[workflow_id]

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get current status of a workflow
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Current workflow status and state
        """
        logger.info(f"Getting workflow status for {workflow_id}")
        logger.info(f"Available workflows in memory: {list(self.workflow_states.keys())}")
        
        # Try LangGraph first
        if self.graph and self.checkpointer:
            try:
                config = {"configurable": {"thread_id": workflow_id}}
                current_state = await self.graph.aget_state(config)
                
                if current_state and current_state.values:
                    state = current_state.values
                    logger.info(f"Found LangGraph state for {workflow_id}: status={state.get('status')}")
                    return self._format_workflow_status_response(workflow_id, state)
                    
            except Exception as e:
                logger.error(f"Failed to get workflow status from LangGraph: {str(e)}")
          # Check state manager first (Redis + fallback)
        shared_state = state_manager.get_workflow_state(workflow_id)
        if shared_state:
            logger.info(f"Found shared state for {workflow_id}: status={shared_state.get('status')}, awaiting_human={shared_state.get('awaiting_human', False)}")
            return self._format_workflow_status_response(workflow_id, shared_state)
        
        # Fallback to simple state store
        if workflow_id in self.workflow_states:
            state = self.workflow_states[workflow_id]
            logger.info(f"Found fallback state for {workflow_id}: status={state.get('status')}, awaiting_human={state.get('awaiting_human', False)}")
            logger.info(f"State keys: {list(state.keys())}")
            
            # Safety check: if workflow is completed but no final report exists, create one
            if (state.get("status") == WorkflowStatus.COMPLETED and 
                not state.get("final_report")):
                logger.warning(f"Workflow {workflow_id} is completed but missing final report. Creating emergency fallback report.")
                
                # Create emergency final report
                generated_sections = state.get("generated_sections", {})
                if generated_sections:
                    # Use orchestrator's method to create a simple report
                    if AgentType.ORCHESTRATOR in self.agents:
                        try:
                            orchestrator = self.agents[AgentType.ORCHESTRATOR]
                            state["final_report"] = orchestrator._create_simple_final_report(state, generated_sections)
                            logger.info("Emergency final report created successfully")
                        except Exception as e:
                            logger.error(f"Failed to create emergency report: {str(e)}")
                            state["final_report"] = "# Emergency Audit Report\n\nWorkflow completed but final report generation failed. Please contact system administrator."
                else:
                    state["final_report"] = "# Audit Report\n\nWorkflow completed but no analysis sections were generated."
                  # Update the stored state
                self.workflow_states[workflow_id] = state.copy()
            
            return self._format_workflow_status_response(workflow_id, state)
        
        logger.warning(f"Workflow {workflow_id} not found in any state store")
        return {"error": "Workflow not found"}

    def _format_workflow_status_response(self, workflow_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to format workflow status response consistently with enhanced messaging"""
        
        # Get current status
        status = state.get("status", WorkflowStatus.PENDING)
        status_value = status.value if hasattr(status, "value") else status
        
        # Enhanced status messaging for frontend
        status_messages = {
            "pending": "Initializing audit workflow...",
            "planning": "AI agents are analyzing your requirements and creating audit plan...",
            "awaiting_approval": "Audit plan ready for your review and approval",
            "processing": "AI agents are conducting comprehensive audit analysis...",
            "completed": "Audit report generated successfully",
            "failed": "Audit workflow encountered an error"
        }
        
        # Determine more specific status message based on state
        status_message = status_messages.get(status_value, f"Status: {status_value}")
        
        # Add processing details
        if status_value == "processing":
            current_section = state.get("current_section")
            if current_section:
                status_message = f"Analyzing {current_section} section..."
            elif state.get("background_processing"):
                status_message = "Deep analysis in progress - AI agents working..."
        
        # Add completion details
        if status_value == "completed":
            final_report = state.get("final_report")
            if final_report and len(final_report) > 100:
                status_message = "Comprehensive audit report ready for download"
            else:
                status_message = "Audit completed - finalizing report..."
        
        # Progress indicator
        generated_sections = state.get("generated_sections", {})
        total_sections = len(state.get("audit_plan", {}).get("sections", [])) if state.get("audit_plan") else 0
        progress_percentage = 0
        
        if total_sections > 0:
            progress_percentage = min(100, (len(generated_sections) / total_sections) * 100)
        elif status_value == "completed":
            progress_percentage = 100
        elif status_value == "processing":
            progress_percentage = max(10, min(95, 20 + len(generated_sections) * 15))
        
        return {
            "workflow_id": workflow_id,
            "status": status_value,
            "status_message": status_message,
            "progress_percentage": round(progress_percentage),
            "updated_at": state.get("updated_at"),
            "awaiting_human": state.get("awaiting_human", False),
            "audit_plan": state.get("audit_plan"),
            "generated_sections": list(generated_sections.keys()),
            "total_sections": total_sections,
            "current_section": state.get("current_section"),
            "errors": state.get("errors", []),
            "messages": state.get("messages", [])[-5:] if state.get("messages") else [],
            "final_report": state.get("final_report"),
            "react_steps": state.get("react_steps", []),
            "background_processing": state.get("background_processing", False),
            "processing_details": {
                "cycles_completed": len([msg for msg in state.get("messages", []) if "Cycle" in msg and "completed" in msg]),
                "last_activity": state.get("updated_at"),
                "has_errors": len(state.get("errors", [])) > 0
            },
            "debug_info": {
                "has_audit_plan": bool(state.get("audit_plan")),
                "plan_approved": state.get("plan_approved", False),
                "enterprise_report_id": state.get("enterprise_report_id"),
                "selected_norms": state.get("selected_norms", []),
                "has_final_report": bool(state.get("final_report")),
                "state_manager_sync": True  # Always true since we ensure sync now
            }
        }

    # Node functions for LangGraph
    async def _orchestrator_node(self, state: AgentState) -> AgentState:
        """Orchestrator node function"""
        return await self.agents[AgentType.ORCHESTRATOR].execute(state)
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node function"""
        return await self.agents[AgentType.PLANNER].execute(state)
    
    async def _analyzer_node(self, state: AgentState) -> AgentState:
        """Analyzer node function"""
        return await self.agents[AgentType.ANALYZER].execute(state)
    
    async def _writer_node(self, state: AgentState) -> AgentState:
        """Writer node function"""
        return await self.agents[AgentType.WRITER].execute(state)
    
    async def _consolidator_node(self, state: AgentState) -> AgentState:
        """Consolidator node function"""
        return await self.agents[AgentType.CONSOLIDATOR].execute(state)
    
    async def _human_approval_node(self, state: AgentState) -> AgentState:
        """Human approval node - marks as awaiting human input and prevents further processing"""
        logger.info(f"Human approval node reached for workflow {state.get('workflow_id')}")
        state["awaiting_human"] = True
        state["status"] = WorkflowStatus.AWAITING_APPROVAL
        state["updated_at"] = datetime.now().isoformat()
        # Store the state to ensure persistence
        workflow_id = state.get("workflow_id")
        if workflow_id and hasattr(self, 'workflow_states'):
            self._ensure_state_persistence(workflow_id, state)
        return state
    
    def _route_from_orchestrator(self, state: AgentState) -> str:
        """Route from orchestrator based on current status with recursion prevention"""
        status = state.get("status")
        awaiting_human = state.get("awaiting_human", False)
        
        # CRITICAL: If awaiting human input, always route to human_approval and log this
        if awaiting_human:
            logger.info(f"ROUTING: Workflow {state.get('workflow_id')} is awaiting human input - routing to human_approval")
            return "human_approval"
        
        # Route based on status
        if status == WorkflowStatus.PLANNING:
            logger.info(f"ROUTING: Status is PLANNING - routing to planner")
            return "planner"
        elif status == WorkflowStatus.ANALYZING:
            logger.info(f"ROUTING: Status is ANALYZING - routing to analyzer")
            return "analyzer"
        elif status == WorkflowStatus.WRITING:
            logger.info(f"ROUTING: Status is WRITING - routing to writer")
            return "writer"        
        elif status == WorkflowStatus.CONSOLIDATING:
            logger.info(f"ROUTING: Status is CONSOLIDATING - routing to consolidator")
            return "consolidator"
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            logger.info(f"ROUTING: Status is {status} - ending workflow")
            return "end"
        else:              
            logger.warning(f"ROUTING: Unknown status {status} - ending workflow")
            return "end"
    
    async def _manual_orchestration(self, state: AgentState):
        """Fallback manual orchestration when LangGraph is not available"""
        logger.info("Starting manual orchestration fallback")
        
        workflow_id = state.get("workflow_id")
        logger.info(f"Manual orchestration for workflow {workflow_id}")
        logger.info(f"Initial state: status={state.get('status')}, enterprise_report_id={state.get('enterprise_report_id')}")
        # CRITICAL: Always ensure the state is stored immediately
        if workflow_id:
            self._ensure_state_persistence(workflow_id, state)
        
        try:
            # Start with planning phase
            logger.info("Setting status to PLANNING and calling planner agent")
            state["status"] = WorkflowStatus.PLANNING
            state["updated_at"] = datetime.now().isoformat()
            # CRITICAL: Update stored state immediately after status change
            if workflow_id:
                self._ensure_state_persistence(workflow_id, state)
            
            # Check if planner agent exists
            if AgentType.PLANNER not in self.agents:
                logger.error(f"Planner agent not found in agents dict. Available: {list(self.agents.keys())}")
                raise ValueError("Planner agent not initialized")
            
            planner_agent = self.agents[AgentType.PLANNER]
            logger.info(f"Planner agent retrieved: {planner_agent}")
            
            logger.info("Executing planner agent...")
            state = await planner_agent.execute(state)
            logger.info(f"Planner execution completed. Has audit_plan: {bool(state.get('audit_plan'))}")
            # CRITICAL: Update stored state after planner execution
            if workflow_id:
                self._ensure_state_persistence(workflow_id, state)
            
            # Check if planner succeeded
            if state.get("audit_plan"):
                # Plan created successfully, wait for human approval
                state["awaiting_human"] = True
                state["status"] = WorkflowStatus.AWAITING_APPROVAL
                state["updated_at"] = datetime.now().isoformat()
                logger.info("Audit plan created successfully, awaiting human approval")
                logger.info(f"Audit plan preview: {str(state.get('audit_plan'))[:200]}...")
            else:
                # Planner failed, mark as failed
                state["status"] = WorkflowStatus.FAILED
                state["updated_at"] = datetime.now().isoformat()
                error_msg = "Planner failed to create audit plan"
                if "errors" not in state:
                    state["errors"] = []
                state["errors"].append(error_msg)
                logger.error(error_msg)
                logger.error(f"Planner errors: {state.get('errors', [])}")
            # CRITICAL: Final state update - this is crucial for persistence
            if workflow_id:
                self._ensure_state_persistence(workflow_id, state)
                
        except Exception as e:
            logger.error(f"Manual orchestration error: {str(e)}")
            logger.exception("Full traceback:")
            state["status"] = WorkflowStatus.FAILED
            state["updated_at"] = datetime.now().isoformat()
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Orchestration error: {str(e)}")
            # CRITICAL: Store state even on error
            if workflow_id:
                self._ensure_state_persistence(workflow_id, state)
    
    async def progress_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Progress a workflow using orchestrator agent's ReAct architecture
        
        Args:
            workflow_id: ID of the workflow to progress
            
        Returns:
            Updated workflow state
        """
        if workflow_id not in self.workflow_states:
            logger.error(f"Workflow {workflow_id} not found for progression")
            return {"error": "Workflow not found"}
        
        state = self.workflow_states[workflow_id].copy()
        logger.info(f"Progressing workflow {workflow_id} using orchestrator ReAct loop")
        logger.info(f"Current status: {state.get('status')}, awaiting_human: {state.get('awaiting_human', False)}")
        
        # Don't progress if waiting for human input
        if state.get("awaiting_human", False):
            logger.info("Workflow is awaiting human input, not progressing")
            return self._format_workflow_response(workflow_id, state)
        
        # Don't progress if in terminal states
        status = state.get("status")
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            logger.info(f"Workflow is in terminal state {status}, not progressing")
            return self._format_workflow_response(workflow_id, state)
        
        try:
            # Use orchestrator agent to make ReAct-based decisions about next steps
            logger.info("Executing orchestrator agent for workflow progression...")
            orchestrator_agent = self.agents[AgentType.ORCHESTRATOR]
            state = await orchestrator_agent.execute(state)
            logger.info(f"Orchestrator execution completed. Status: {state.get('status')}, Current section: {state.get('current_section')}")
            
            # Update stored state
            self.workflow_states[workflow_id] = state.copy()
            
        except Exception as e:
            logger.error(f"Orchestrator execution failed during workflow progression: {str(e)}")
            state["status"] = WorkflowStatus.FAILED
            state["errors"].append(f"Workflow progression error: {str(e)}")
            self.workflow_states[workflow_id] = state.copy()
        
        return self._format_workflow_response(workflow_id, state)
    
    def _format_workflow_response(self, workflow_id: str, state: AgentState) -> Dict[str, Any]:
        """Format workflow state for API response"""
        return {
            "workflow_id": workflow_id,
            "status": state.get("status").value if hasattr(state.get("status"), "value") else state.get("status"),
            "awaiting_human": state.get("awaiting_human", False),
            "plan_approved": state.get("plan_approved", False),
            "updated_at": state.get("updated_at"),
            "current_section": state.get("current_section"),            "generated_sections": list(state.get("generated_sections", {}).keys()),
            "errors": state.get("errors", [])
        }

    def _ensure_state_persistence(self, workflow_id: str, state: AgentState) -> None:
        """
        Ensure workflow state is always persisted using shared state manager with robust error handling
        
        Args:
            workflow_id: ID of the workflow
            state: State to persist
        """
        if not workflow_id:
            logger.warning("Cannot persist state - no workflow_id provided")
            return
        
        try:
            # Store in shared state manager (Redis + fallback to memory)
            success = state_manager.store_workflow_state(workflow_id, state)
            
            # Also store in local memory for backwards compatibility
            self.workflow_states[workflow_id] = state.copy()
            
            # Enhanced logging for debugging multi-worker issues
            if success:
                logger.info(f"✓ PERSISTENCE: Successfully stored state for workflow {workflow_id}")
            else:
                logger.warning(f"⚠ PERSISTENCE: Failed to store in shared manager for workflow {workflow_id}")
            
            logger.debug(f"PERSISTENCE: Status={state.get('status')}, awaiting_human={state.get('awaiting_human')}")
            logger.debug(f"PERSISTENCE: Has final_report={bool(state.get('final_report'))}")
            logger.debug(f"PERSISTENCE: Total workflows in local store: {len(self.workflow_states)}")
            
            # Verify persistence by attempting to retrieve
            if success:
                try:
                    verification_state = state_manager.get_workflow_state(workflow_id)
                    if verification_state:
                        logger.debug(f"✓ PERSISTENCE: Verified state retrieval for workflow {workflow_id}")
                    else:
                        logger.warning(f"⚠ PERSISTENCE: Verification failed - could not retrieve workflow {workflow_id}")
                except Exception as verify_error:
                    logger.warning(f"⚠ PERSISTENCE: Verification check failed: {str(verify_error)}")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to persist state for workflow {workflow_id}: {str(e)}")
            logger.exception("PERSISTENCE ERROR traceback:")
            
            # Ensure at least local persistence works as fallback
            try:
                self.workflow_states[workflow_id] = state.copy()
                logger.info(f"✓ FALLBACK: Ensured local persistence for workflow {workflow_id}")
            except Exception as fallback_error:
                logger.critical(f"❌ CRITICAL: Even local persistence failed: {str(fallback_error)}")
