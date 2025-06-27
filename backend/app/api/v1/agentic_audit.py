"""
Agentic Audit API Endpoints
===========================

FastAPI endpoints for the agentic audit conformity report generation system.
Provides interfaces for starting workflows, managing human-in-the-loop interactions,
and monitoring workflow progress.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

from app.agents.workflow_manager import AuditWorkflowManager
from app.agents.audit_types import WorkflowStatus
from app.services.settings_service import load_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic-audit"])

# Global workflow manager instance
workflow_manager: Optional[AuditWorkflowManager] = None

def get_workflow_manager() -> AuditWorkflowManager:
    """Get or create workflow manager instance"""
    global workflow_manager
    
    if workflow_manager is None:
        # Get LLM configuration from settings
        settings = load_settings()
        llm_config = {
            "provider": settings.get("llm_provider", "ollama"),
            "model": settings.get("llm_model", "llama3.2:latest"),
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        workflow_manager = AuditWorkflowManager(llm_config)
        logger.info("Initialized audit workflow manager")
    
    return workflow_manager

@router.get(
    "/norms",
    summary="Get available conformity norms",
    response_description="List of available conformity norms for audit"
)
async def get_available_norms() -> Dict[str, Any]:
    """
    Get list of available conformity norms that can be selected for audit
    
    Returns:
        Dict containing available norms organized by category
    """
    try:
        # Standard conformity norms
        norms = {
            "financial_standards": {
                "name": "Financial Standards",
                "norms": [
                    {
                        "id": "ifrs",
                        "name": "IFRS (International Financial Reporting Standards)",
                        "description": "International accounting standards for financial reporting",
                        "applicable_cycles": ["ventes_clients", "achats_fournisseurs", "immobilisations", "stocks"]
                    },
                    {
                        "id": "gaap",
                        "name": "GAAP (Generally Accepted Accounting Principles)",
                        "description": "Standard accounting principles and procedures",
                        "applicable_cycles": ["all"]
                    },                    {
                        "id": "cgi_maroc",
                        "name": "Code Général des Impôts (CGI) - Maroc",
                        "description": "Code fiscal marocain pour la comptabilité et les obligations fiscales",
                        "applicable_cycles": ["all"]
                    }
                ]
            },
            "quality_standards": {
                "name": "Quality Management Standards",
                "norms": [
                    {
                        "id": "iso_9001",
                        "name": "ISO 9001:2015",
                        "description": "Quality management systems standard",
                        "applicable_cycles": ["stocks", "achats_fournisseurs", "immobilisations"]
                    },
                    {
                        "id": "iso_14001",
                        "name": "ISO 14001:2015",
                        "description": "Environmental management systems",
                        "applicable_cycles": ["achats_fournisseurs", "immobilisations"]
                    }
                ]
            },
            "security_standards": {
                "name": "Security and Risk Standards",
                "norms": [
                    {
                        "id": "iso_27001",
                        "name": "ISO 27001:2022",
                        "description": "Information security management",
                        "applicable_cycles": ["tresorerie", "paie_personnel"]
                    },
                    {
                        "id": "coso",
                        "name": "COSO Framework",
                        "description": "Internal control framework",
                        "applicable_cycles": ["all"]
                    }
                ]
            },            "regulatory_compliance": {
                "name": "Regulatory Compliance",
                "norms": [
                    {
                        "id": "sarbanes_oxley",
                        "name": "Sarbanes-Oxley Act",
                        "description": "US federal law for financial reporting",
                        "applicable_cycles": ["ventes_clients", "tresorerie", "impots_taxes"]
                    },
                    {
                        "id": "gdpr",
                        "name": "GDPR",
                        "description": "General Data Protection Regulation",
                        "applicable_cycles": ["paie_personnel"]
                    }
                ]
            },
            "moroccan_compliance": {
                "name": "Conformité Réglementaire Marocaine",
                "norms": [
                    {
                        "id": "loi_comptable_maroc",
                        "name": "Loi Comptable Marocaine (Loi 9-88)",
                        "description": "Obligations comptables des commerçants au Maroc",
                        "applicable_cycles": ["all"]
                    },
                    {
                        "id": "cgnc",
                        "name": "Code Général de Normalisation Comptable (CGNC)",
                        "description": "Norme comptable marocaine pour l'établissement des états financiers",
                        "applicable_cycles": ["all"]
                    },
                    {
                        "id": "tva_maroc",
                        "name": "TVA Maroc",
                        "description": "Taxe sur la Valeur Ajoutée selon le CGI marocain",
                        "applicable_cycles": ["ventes_clients", "achats_fournisseurs", "impots_taxes"]
                    },
                    {
                        "id": "ir_maroc",
                        "name": "Impôt sur le Revenu (IR) - Maroc",
                        "description": "Obligations fiscales liées à l'IR selon le CGI marocain",
                        "applicable_cycles": ["paie_personnel", "impots_taxes"]
                    },
                    {
                        "id": "is_maroc",
                        "name": "Impôt sur les Sociétés (IS) - Maroc",
                        "description": "Obligations fiscales des sociétés selon le CGI marocain",
                        "applicable_cycles": ["all"]
                    }
                ]
            }
        }
        
        return {
            "status": "success",
            "norms": norms,
            "total_categories": len(norms),
            "total_norms": sum(len(category["norms"]) for category in norms.values())
        }
        
    except Exception as e:
        logger.error(f"Error getting available norms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available norms: {str(e)}")

@router.post(
    "/workflows",
    summary="Start new audit workflow",
    response_description="Started workflow information"
)
async def start_audit_workflow(request: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Start a new agentic audit conformity workflow
    
    Args:
        request: Request containing enterprise_report_id, selected_norms, and user_id
        
    Returns:
        Dict containing workflow ID and initial status
    """
    try:
        # Validate request
        enterprise_report_id = request.get("enterprise_report_id")
        selected_norms = request.get("selected_norms", [])
        user_id = request.get("user_id", "anonymous")
        
        if not enterprise_report_id:
            raise HTTPException(status_code=400, detail="enterprise_report_id is required")
        
        if not selected_norms:
            raise HTTPException(status_code=400, detail="At least one conformity norm must be selected")
        
        # Get workflow manager
        manager = get_workflow_manager()
        
        # Start workflow in background
        workflow_id = await manager.start_workflow(
            enterprise_report_id=enterprise_report_id,
            selected_norms=selected_norms,
            user_id=user_id
        )
        
        logger.info(f"Started audit workflow {workflow_id} for user {user_id}")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "message": "Audit workflow started successfully",
            "started_at": datetime.now().isoformat(),
            "enterprise_report_id": enterprise_report_id,
            "selected_norms": selected_norms
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting audit workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start audit workflow: {str(e)}")

@router.get(
    "/workflows/{workflow_id}",
    summary="Get workflow status",
    response_description="Current workflow status and details"
)
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get the current status and details of an audit workflow
    
    Args:
        workflow_id: ID of the workflow
        
    Returns:
        Dict containing workflow status and details
    """
    logger.info(f"API_DEBUG: get_workflow_status called for {workflow_id}")
    try:
        manager = get_workflow_manager()
        logger.info(f"API_DEBUG: Retrieved workflow manager, calling get_workflow_status")
        status = await manager.get_workflow_status(workflow_id)
        # logger.info(f"API_DEBUG: Manager returned status: {status}")
        
        if "error" in status:
            logger.warning(f"API_DEBUG: Status contains error, returning 404: {status['error']}")
            raise HTTPException(status_code=404, detail=status["error"])
        
        logger.info(f"API_DEBUG: Returning successful status for {workflow_id}")
        return {
            "status": "success",
            "workflow_id": workflow_id,
            **status
        }
        
    except HTTPException:
        logger.error(f"API_DEBUG: HTTPException raised for {workflow_id}")
        raise
    except Exception as e:
        logger.error(f"API_DEBUG: Unexpected error getting workflow status for {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")

@router.post(
    "/workflows/{workflow_id}/approve",
    summary="Approve audit plan (Human-in-the-Loop)",
    response_description="Immediate workflow continuation result with background processing"
)
async def approve_audit_plan(workflow_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Approve or reject the audit plan (Human-in-the-Loop interaction)
    
    This endpoint returns immediately after updating the workflow state,
    allowing the frontend to switch to processing view while the actual
    workflow execution happens asynchronously in the background.
    
    Args:
        workflow_id: ID of the workflow
        request: Request containing approval decision and optional feedback
        
    Returns:
        Dict containing immediate workflow continuation result    """
    try:
        approval = request.get("approved", False)
        feedback = request.get("feedback", "")
        
        logger.info(f"Processing approval request for workflow {workflow_id}: approved={approval}")
        
        manager = get_workflow_manager()
        # Use the new async processing method
        result = await manager.continue_workflow_async(
            workflow_id=workflow_id,
            human_feedback=feedback,
            approval=approval
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        logger.info(f"Audit plan {'approved' if approval else 'rejected'} for workflow {workflow_id}")
        
        # Return immediate response
        response = {
            "status": "success",
            "workflow_id": workflow_id,
            "approved": approval,
            "message": f"Audit plan {'approved' if approval else 'rejected'} successfully",
            "feedback_provided": bool(feedback),
            "immediate_response": True,
            "background_processing_started": approval,  # True if processing started in background
            "workflow_status": result.get("status"),
            "updated_at": result.get("updated_at"),
            "transition_to_processing": approval  # Explicit signal for frontend
        }
        
        if approval:
            response["processing_message"] = "Workflow processing has started in the background. You can monitor progress via the status endpoint."
            response["redirect_to"] = "processing"  # Explicit redirect instruction
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving audit plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to approve audit plan: {str(e)}")

# Add a new endpoint for real-time status monitoring
@router.get(
    "/workflows/{workflow_id}/status/live",
    summary="Get real-time workflow status",
    response_description="Current workflow status with live updates"
)
async def get_live_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get real-time workflow status including background processing updates
    
    Args:
        workflow_id: ID of the workflow
        
    Returns:
        Dict containing current workflow status with live updates
    """
    try:
        manager = get_workflow_manager()
        status = await manager.get_workflow_status(workflow_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
          # Add live status indicators with detailed progress tracking
        current_status = status.get("status")
        is_processing = current_status in ["analyzing", "writing", "consolidating"]
        is_completed = current_status in ["completed", "failed", "cancelled"]
        is_background_processing = status.get("background_processing", False)
        
        # Calculate detailed progress
        audit_plan = status.get("audit_plan", {})
        planned_sections = audit_plan.get("sections", [])
        generated_sections = status.get("generated_sections", [])
        current_section = status.get("current_section")
        
        # Build section progress data
        section_progress = []
        for i, section in enumerate(planned_sections):
            section_id = section.get("id", f"section_{i}")
            section_name = section.get("name", f"Section {i+1}")
            
            # Determine section status
            if section_id in generated_sections:
                section_status = "completed"
            elif section_id == current_section:
                section_status = "in_progress"
            else:
                section_status = "pending"
            
            section_progress.append({
                "id": section_id,
                "name": section_name,
                "status": section_status,
                "order": i + 1
            })
        
        # Calculate completion percentage
        total_sections = len(planned_sections)
        completed_sections = len(generated_sections)
        completion_percentage = round((completed_sections / total_sections * 100) if total_sections > 0 else 0)
        
        live_status = {
            **status,
            "live_indicators": {
                "is_processing": is_processing,
                "is_completed": is_completed,
                "is_waiting": status.get("awaiting_human", False),
                "is_background_processing": is_background_processing,
                "last_update": status.get("updated_at"),
                "processing_stage": current_status
            },
            "progress_tracking": {
                "total_sections": total_sections,
                "completed_sections": completed_sections,
                "completion_percentage": completion_percentage,
                "current_section_name": next((s["name"] for s in section_progress if s["status"] == "in_progress"), None),
                "section_progress": section_progress
            },
            "progress_info": {
                "total_sections": len(status.get("generated_sections", [])),
                "current_section": status.get("current_section"),
                "estimated_completion": (
                    "Processing in background..." if is_background_processing 
                    else ("Completed" if is_completed 
                    else ("Processing..." if is_processing 
                    else "Pending"))
                ),
                "recent_messages": status.get("messages", [])[-3:] if status.get("messages") else []
            }
        }
        
        return live_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting live workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get live workflow status: {str(e)}")

@router.post(
    "/workflows/{workflow_id}/progress",
    summary="Progress workflow using ReAct orchestration",
    response_description="Workflow progression result"
)
async def progress_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Progress the workflow to the next step using orchestrator's ReAct architecture
    
    Args:
        workflow_id: ID of the workflow to progress
        
    Returns:
        Dict containing workflow progression result
    """
    try:
        manager = get_workflow_manager()
        result = await manager.progress_workflow(workflow_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        logger.info(f"Workflow {workflow_id} progressed to status: {result.get('status')}")
        
        return {
            "status": "success",
            "message": "Workflow progressed successfully",
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error progressing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to progress workflow: {str(e)}")

@router.post(
    "/workflows/{workflow_id}/feedback",
    summary="Provide human feedback",
    response_description="Feedback processing result"
)
async def provide_human_feedback(workflow_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide human feedback to the workflow
    
    Args:
        workflow_id: ID of the workflow
        request: Request containing feedback message
        
    Returns:
        Dict containing feedback processing result
    """
    try:
        feedback = request.get("feedback", "")
        
        if not feedback:
            raise HTTPException(status_code=400, detail="Feedback message is required")
        
        manager = get_workflow_manager()
        result = await manager.continue_workflow(
            workflow_id=workflow_id,
            human_feedback=feedback
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        logger.info(f"Human feedback provided for workflow {workflow_id}")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "feedback_received": feedback,
            "message": "Feedback provided successfully",
            "workflow_updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to provide feedback: {str(e)}")

@router.get(
    "/workflows/{workflow_id}/plan",
    summary="Get audit plan for approval",
    response_description="Audit plan details for human review"
)
async def get_audit_plan(workflow_id: str) -> Dict[str, Any]:
    """
    Get the audit plan for human review and approval
    
    Args:
        workflow_id: ID of the workflow
        
    Returns:
        Dict containing audit plan details
    """
    try:
        manager = get_workflow_manager()
        status = await manager.get_workflow_status(workflow_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        audit_plan = status.get("audit_plan")
        if not audit_plan:
            raise HTTPException(status_code=404, detail="Audit plan not yet available")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "audit_plan": audit_plan,
            "awaiting_approval": status.get("awaiting_human", False),
            "plan_approved": audit_plan.get("plan_approved", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audit plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit plan: {str(e)}")

@router.get(
    "/workflows/{workflow_id}/report",
    summary="Get final audit report",
    response_description="Generated audit conformity report"
)
async def get_audit_report(workflow_id: str) -> Dict[str, Any]:
    """
    Get the final generated audit conformity report
    
    Args:
        workflow_id: ID of the workflow
        
    Returns:
        Dict containing the final audit report
    """
    try:
        manager = get_workflow_manager()
        
        # First, try to get status from workflow manager (includes fallback state)
        status = await manager.get_workflow_status(workflow_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        # Try to get final report from the status response first (fallback mode)
        final_report = status.get("final_report")
        
        if final_report:
            return {
                "status": "success",
                "workflow_id": workflow_id,
                "final_report": final_report,
                "generated_at": status.get("updated_at"),
                "report_format": "markdown"
            }
        
        # If not found in status response, try direct access to workflow state
        if workflow_id in manager.workflow_states:
            state = manager.workflow_states[workflow_id]
            final_report = state.get("final_report")
            
            if final_report:
                return {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "final_report": final_report,
                    "generated_at": state.get("updated_at"),
                    "report_format": "markdown"
                }
        
        # Only try LangGraph as last resort (and handle its errors)
        if manager.graph and manager.checkpointer:
            try:
                config = {"configurable": {"thread_id": workflow_id}}
                current_state = await manager.graph.aget_state(config)
                
                if current_state and current_state.values:
                    final_report = current_state.values.get("final_report")
                    
                    if final_report:
                        return {
                            "status": "success",
                            "workflow_id": workflow_id,
                            "final_report": final_report,
                            "generated_at": current_state.values.get("updated_at"),
                            "report_format": "markdown"
                        }
                    
            except Exception as e:
                logger.warning(f"Could not access LangGraph state for final report: {str(e)}")
                # Don't raise here, continue to check workflow status
        
        # Check workflow status to give appropriate error
        workflow_status = status.get("status")
        if workflow_status == "completed":
            # Log state keys for debugging
            if workflow_id in manager.workflow_states:
                state_keys = list(manager.workflow_states[workflow_id].keys())
                logger.error(f"Workflow {workflow_id} is completed but no final_report found. State keys: {state_keys}")
                has_final_report = "final_report" in manager.workflow_states[workflow_id]
                final_report_value = manager.workflow_states[workflow_id].get("final_report")
                logger.error(f"Has final_report key: {has_final_report}, Value type: {type(final_report_value)}, Value length: {len(str(final_report_value)) if final_report_value else 0}")
            
            raise HTTPException(status_code=404, detail="Final report not found despite completed status")
        else:
            raise HTTPException(
                status_code=202, 
                detail=f"Report not ready yet. Workflow status: {workflow_status}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audit report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit report: {str(e)}")

@router.delete(
    "/workflows/{workflow_id}",
    summary="Cancel audit workflow",
    response_description="Cancellation result"
)
async def cancel_audit_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Cancel a running audit workflow
    
    Args:
        workflow_id: ID of the workflow to cancel
        
    Returns:
        Dict containing cancellation result
    """
    try:
        manager = get_workflow_manager()
        
        # For now, just return success as cancellation logic would need
        # more complex state management
        logger.info(f"Workflow {workflow_id} marked for cancellation")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "message": "Workflow cancellation requested",
            "cancelled_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cancelling workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")

@router.get(
    "/audit-cycles",
    summary="Get audit cycle templates",
    response_description="Available audit cycle templates"
)
async def get_audit_cycles() -> Dict[str, Any]:
    """
    Get available audit cycle templates with their objectives and controls
    
    Returns:
        Dict containing audit cycle templates
    """
    try:
        from app.agents.audit_types import AUDIT_CYCLE_TEMPLATES, AuditCycle
        
        # Convert enum keys to strings for JSON serialization
        cycles = {}
        for cycle_enum, template in AUDIT_CYCLE_TEMPLATES.items():
            cycles[cycle_enum.value] = {
                **template,
                "cycle_id": cycle_enum.value
            }
        
        return {
            "status": "success",
            "audit_cycles": cycles,
            "total_cycles": len(cycles)
        }
        
    except Exception as e:
        logger.error(f"Error getting audit cycles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit cycles: {str(e)}")

@router.post(
    "/workflows/{workflow_id}/progress",
    summary="Progress workflow using ReAct orchestration",
    response_description="Workflow progression result"
)
async def progress_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Progress the workflow to the next step using orchestrator's ReAct architecture
    
    Args:
        workflow_id: ID of the workflow to progress
        
    Returns:
        Dict containing workflow progression result
    """
    try:
        manager = get_workflow_manager()
        result = await manager.progress_workflow(workflow_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        logger.info(f"Workflow {workflow_id} progressed to status: {result.get('status')}")
        
        return {
            "status": "success",
            "message": "Workflow progressed successfully",
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error progressing workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to progress workflow: {str(e)}")

@router.get(
    "/workflows/{workflow_id}/react_steps",
    summary="Get ReAct steps for a workflow",
    response_description="List of ReAct steps (thought, action, observation, agent, timestamp)"
)
async def get_react_steps(workflow_id: str) -> Dict[str, Any]:
    """
    Get the ReAct steps for a workflow for frontend visualization.
    """
    try:
        manager = get_workflow_manager()
        
        # Try to get from LangGraph state first
        if manager.graph and manager.checkpointer:
            try:
                config = {"configurable": {"thread_id": workflow_id}}
                current_state = await manager.graph.aget_state(config)
                if current_state and current_state.values and "react_steps" in current_state.values:
                    return {"steps": current_state.values["react_steps"]}
            except Exception as e:
                logger.warning(f"Could not access LangGraph state for ReAct steps: {str(e)}")
        
        # Fallback to workflow manager state
        state = manager.workflow_states.get(workflow_id)
        if not state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        react_steps = state.get("react_steps", [])
        return {
            "steps": react_steps,
            "total_steps": len(react_steps),
            "workflow_status": state.get("status", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ReAct steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get ReAct steps: {str(e)}")
