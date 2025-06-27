"""
Orchestrator Agent - ReAct Architecture
=======================================

The orchestrator agent manages the overall workflow using ReAct (Reasoning + Acting)
architecture. It coordinates all other agents and makes decisions about the next steps
in the audit conformity report generation process.

ReAct Flow:
1. Thought: Analyze current state and determine what needs to be done
2. Action: Choose which agent to call or what action to take
3. Observation: Analyze the results and update the state
4. Repeat until completion
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from .base_agent import BaseAgent
from .audit_types import AgentState, AgentType, WorkflowStatus, AuditCycle

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent using ReAct architecture"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(AgentType.ORCHESTRATOR, llm_config)
        self.max_iterations = 30  # Prevent infinite loops

    def get_available_actions(self, state: AgentState) -> List[str]:
        """List all available actions for the orchestrator agent"""
        return [
            "create_audit_plan",
            "start_processing", 
            "start_next_section",
            "query_knowledge",
            "call_analyzer",
            "call_writer",
            "call_consolidator",
            "wait_human_approval",
            "finish_workflow"
        ]

    def _append_react_step(self, state, thought, action, observation, agent):
        """Append a ReAct step to the state for frontend visualization"""
        if "react_steps" not in state:
            state["react_steps"] = []
        state["react_steps"].append({
            "thought": thought,
            "action": action,
            "observation": observation,
            "agent": agent,
            "timestamp": datetime.now().isoformat()        })
        return state
        
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute the ReAct orchestration loop
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after orchestration        """
        if not self.validate_state(state):
            return self.log_error(state, "Invalid state received by orchestrator")
        
        # Check if workflow is already awaiting human input - don't restart orchestration
        if state.get("awaiting_human", False):
            state = self.log_message(state, "Workflow is awaiting human input - orchestration paused")
            # Important: Return immediately without entering the loop to prevent recursion
            return state
        
        # Check if workflow is already completed - don't restart
        if state.get("status") in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            state = self.log_message(state, f"Workflow already in final state: {state.get('status')} - orchestration stopped")
            return state
        
        state = self.log_message(state, "Orchestrator starting ReAct workflow")
        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while iteration < self.max_iterations:
            iteration += 1

            # ReAct Step 1: Thought - Analyze current situation
            thought = await self._think(state)
            state = self.log_message(state, f"Thought: {thought}")

            # Check for early completion conditions before deciding action
            if state["status"] == WorkflowStatus.COMPLETED:
                state = self.log_message(state, "Workflow already completed, stopping orchestration")
                break
            
            # Check if we have a final report and should complete
            if state.get("final_report") and state["status"] == WorkflowStatus.CONSOLIDATING:
                state = self.update_status(state, WorkflowStatus.COMPLETED)
                state = self.log_message(state, "Final report generated, marking workflow as completed")
                # Append final ReAct step
                state = self._append_react_step(state, thought, {"type": "complete_workflow", "description": "Final report ready, workflow completed"}, "Workflow completed successfully", self.agent_id)
                break

            # ReAct Step 2: Action - Decide what to do next
            try:
                action = await self._decide_action(state, thought)
                state = self.log_message(state, f"Action: {action['type']} - {action['description']}")
                consecutive_errors = 0  # Reset error count on successful action decision
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error deciding action (attempt {consecutive_errors}): {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    state = self.log_error(state, f"Too many consecutive errors ({consecutive_errors}), forcing finalization")
                    action = {"type": "finalize_workflow", "description": "Force completion due to errors", "parameters": {}}
                else:
                    # Use fallback action
                    action = self._get_fallback_action(state)
                    state = self.log_message(state, f"Using fallback action: {action['type']}")

            # ReAct Step 3: Act - Execute the chosen action
            try:
                state = await self._execute_action(state, action)
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error executing action {action.get('type')}: {str(e)}")
                state = self.log_error(state, f"Failed to execute action: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    # Force finalization
                    state["final_report"] = f"# Error Report\n\nWorkflow failed after {consecutive_errors} consecutive errors.\nLast error: {str(e)}"
                    state = self.update_status(state, WorkflowStatus.COMPLETED)
                    break

            # ReAct Step 4: Observation - Analyze results
            observation = await self._observe(state)
            state = self.log_message(state, f"Observation: {observation}")

            # Append ReAct step for frontend visualization
            state = self._append_react_step(state, thought, action, observation, self.agent_id)

            # Check if workflow is complete after action
            if state["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                break

            # Check if waiting for human input
            if state["awaiting_human"]:
                state = self.log_message(state, "Pausing workflow - awaiting human input")
                break
                
            # Additional safety check: if we're in ERROR status for too long, force completion
            if state["status"] == WorkflowStatus.FAILED and iteration > 5:
                state = self.log_message(state, "Workflow in ERROR status too long, forcing completion")
                state["final_report"] = "# Error Recovery Report\n\nWorkflow completed after error recovery."
                state = self.update_status(state, WorkflowStatus.COMPLETED)
                break
        
        if iteration >= self.max_iterations:
            state = self.log_error(state, "Orchestrator reached maximum iterations - forcing completion")
            # Force completion instead of failure
            if not state.get("final_report"):
                state["final_report"] = f"# Timeout Report\n\nWorkflow completed after {self.max_iterations} iterations."
            state = self.update_status(state, WorkflowStatus.COMPLETED)
        
        return state
    
    async def _think(self, state: AgentState) -> str:
        """
        ReAct Thought step - analyze current situation
        
        Args:
            state: Current workflow state
            
        Returns:
            Thought about current situation and next steps
        """
        status = state["status"]
        generated_sections = state.get("generated_sections", {})
        audit_plan = state.get("audit_plan", {})
        final_report = state.get("final_report")
        
        # Get planned sections for progress assessment
        planned_sections = []
        if audit_plan and "sections" in audit_plan:
            planned_sections = [section.get("cycle") for section in audit_plan["sections"] if section.get("cycle")]
        
        context = f"""
        Current workflow status: {status.value}
        Enterprise report ID: {state.get('enterprise_report_id', 'None')}
        Selected norms: {state.get('selected_norms', [])}
        Plan approved: {state.get('plan_approved', False)}
        Audit plan exists: {bool(state.get('audit_plan'))}
        Current section: {state.get('current_section', 'None')}
        Awaiting human: {state.get('awaiting_human', False)}
        Generated sections: {list(generated_sections.keys())} ({len(generated_sections)}/{len(planned_sections)})
        Has final report: {bool(final_report)}
        
        Recent messages: {json.dumps(state['messages'][-3:], indent=2) if state['messages'] else 'None'}
        Errors: {json.dumps(state['errors'][-2:], indent=2) if state['errors'] else 'None'}
        """
        
        prompt = f"""
        You are an orchestrator for an audit conformity report generation system.
        Analyze the current situation and think about what should happen next.
        
        Current Context:
        {context}
        
        Think step by step about:
        1. What has been completed so far?
        2. What is the current status/issue?
        3. What should be the next logical step?
        4. Are we waiting for something (human input, agent completion)?
        5. Are there any errors or issues to address?
        6. If audit plan exists but is not approved, should we wait for human approval?
        7. If all planned sections are done, should we consolidate?
        8. If final report exists, should we complete the workflow?
        
        IMPORTANT: If an audit plan exists but is not approved, the workflow should wait for human approval, not continuously query knowledge.
        
        Provide your analytical thought in 2-3 sentences focusing on workflow progression.
        """
        
        thought = await self.call_llm(prompt, "You are a strategic orchestrator that thinks systematically about workflow progression.")
        return thought.strip()
    
    async def _decide_action(self, state: AgentState, thought: str) -> Dict[str, Any]:
        """
        ReAct Action step - decide what action to take next
        
        Args:
            state: Current workflow state
            thought: Previous thought about the situation
            
        Returns:
            Action to take with type and parameters        """
        status = state["status"]        # Check if we should prioritize waiting for approval
        has_plan = bool(state.get("audit_plan"))
        plan_approved = state.get("plan_approved", False)
        
        # If we have a plan but it's not approved, prioritize waiting for approval
        if has_plan and not plan_approved and status in [WorkflowStatus.PLANNING, WorkflowStatus.PENDING]:
            # Return wait_human_approval action directly to prevent knowledge query loops
            return {
                "type": "wait_human_approval",
                "description": "Audit plan exists but needs human approval before proceeding",
                "parameters": {}
            }
        
        # If we're consolidating, prioritize calling the consolidator
        if status == WorkflowStatus.CONSOLIDATING:
            return {
                "type": "call_consolidator", 
                "description": "All sections completed, generating final consolidated report",
                "parameters": {}
            }
          # If all sections are done and we're analyzing, move to consolidation
        generated_sections = state.get("generated_sections", {})
        audit_plan = state.get("audit_plan", {})
        plan_sections = audit_plan.get("sections", []) if audit_plan else []
        planned_cycles = [section.get("cycle") for section in plan_sections if section and section.get("cycle")]
        all_sections_complete = all(cycle in generated_sections for cycle in planned_cycles) if planned_cycles else False
        
        if status == WorkflowStatus.ANALYZING and all_sections_complete:
            return {
                "type": "call_consolidator",
                "description": "All planned sections completed, starting consolidation",
                "parameters": {}
            }
        
        # Define possible actions based on current state
        possible_actions = self._get_possible_actions(state)
        
        prompt = f"""
        Based on your thought: "{thought}"
        
        Current status: {status.value}
        
        Choose the most appropriate action from these options:
        {json.dumps(possible_actions, indent=2)}
        
        Respond with a JSON object containing:
        {{
            "type": "action_type",
            "description": "brief description of why this action",
            "parameters": {{}}
        }}
        
        IMPORTANT: 
        - The "type" field MUST be exactly one of the action types listed above
        - Do NOT use "N/A", "none", or any other invalid values
        - Choose only ONE action that makes the most sense given the current situation
        - Always respond with valid JSON
        """
        
        try:
            response = await self.call_llm(prompt, "You are a decision-making orchestrator. Always respond with valid JSON using only the provided action types.")
            
            # Try to extract JSON from response
            response = response.strip()
            
            # Look for JSON in the response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                response = response[start:end]
            
            action = json.loads(response)
            
            # Validate action structure
            if "type" not in action:
                raise ValueError("Action must have 'type' field")
            
            # Validate action type is in possible actions
            valid_types = [act["type"] for act in possible_actions]
            if action["type"] not in valid_types:
                logger.warning(f"Invalid action type '{action['type']}' not in {valid_types}")
                raise ValueError(f"Invalid action type: {action['type']}")
            
            # Ensure required fields exist
            if "description" not in action:
                action["description"] = f"Execute {action['type']} action"
            if "parameters" not in action:
                action["parameters"] = {}
            
            logger.info(f"[{self.agent_id}] Selected action: {action['type']} - {action['description']}")
            return action
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in action decision: {str(e)}")
            logger.error(f"Response was: {response}")
            # Fallback based on current status
            fallback_action = self._get_fallback_action(state)
            logger.warning(f"Using fallback action: {fallback_action['type']}")
            return fallback_action
            
        except Exception as e:
            logger.error(f"Failed to decide action: {str(e)}")
            # Fallback based on current status
            fallback_action = self._get_fallback_action(state)
            logger.warning(f"Using fallback action due to error: {fallback_action['type']}")
            return fallback_action
    
    def _get_possible_actions(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Get list of possible actions based on current state
        
        Args:
            state: Current workflow state
            
        Returns:
            List of possible actions with type and description
        """
        status = state["status"]
        has_audit_plan = bool(state.get("audit_plan"))
        plan_approved = state.get("plan_approved", False)
        generated_sections = state.get("generated_sections", {})
        current_section = state.get("current_section")        
        possible_actions = []
          # Handle initial planning phase - only PENDING status should create plan
        if status == WorkflowStatus.PENDING:
            if not has_audit_plan:
                possible_actions.extend([
                    {
                        "type": "create_audit_plan",
                        "description": "Create comprehensive audit plan"
                    }
                ])
            elif has_audit_plan and not plan_approved:
                # Plan exists but not approved - move to awaiting approval
                possible_actions.extend([
                    {
                        "type": "wait_human_approval", 
                        "description": "Wait for human approval of the audit plan"
                    }
                ])
        elif status == WorkflowStatus.PLANNING:
            # Should not happen if status transitions are correct, but handle it
            if has_audit_plan and not plan_approved:
                possible_actions.extend([
                    {
                        "type": "wait_human_approval", 
                        "description": "Wait for human approval of the audit plan"
                    }
                ])
            elif has_audit_plan and plan_approved:
                possible_actions.extend([
                    {
                        "type": "start_processing",
                        "description": "Begin processing first audit section"
                    }
                ])
        elif status == WorkflowStatus.AWAITING_APPROVAL:
            # Only allow processing if plan is actually approved
            if state.get("plan_approved", False):                possible_actions.extend([
                    {
                        "type": "start_processing",
                        "description": "Begin the audit processing phase"
                    }
                ])
            else:
                # Still waiting for human approval
                possible_actions.extend([
                    {
                        "type": "wait_human_approval",
                        "description": "Continue waiting for human approval of the audit plan"
                    }
                ])
            
        elif status == WorkflowStatus.ANALYZING:
            # Check if we need to start a specific section or continue processing
            audit_plan = state.get("audit_plan", {})
            plan_sections = audit_plan.get("sections", []) if audit_plan else []
            planned_cycles = [section.get("cycle") for section in plan_sections if section and section.get("cycle")]
            
            # Find next section that needs processing
            next_section_needed = None
            for cycle in planned_cycles:
                if cycle not in generated_sections:
                    next_section_needed = cycle
                    break
            
            if next_section_needed:
                if not current_section:
                    possible_actions.extend([
                        {
                            "type": "start_next_section",
                            "description": f"Start processing section: {next_section_needed}"
                        }
                    ])
                else:
                    # We have a current section - analyze it
                    possible_actions.extend([
                        {
                            "type": "call_analyzer",
                            "description": f"Analyze current section: {current_section}"
                        }
                    ])
            else:
                # All sections completed, move to consolidation
                possible_actions.extend([
                    {
                        "type": "call_consolidator",
                        "description": "All sections analyzed, consolidate findings into final report"
                    }
                ])
                
        elif status == WorkflowStatus.WRITING:
            # When in WRITING status, we should generate content and then move to next section
            current_section = state.get("current_section")
            if current_section and current_section not in generated_sections:                # Generate content for current section
                possible_actions.extend([
                    {
                        "type": "call_writer",
                        "description": f"Generate content for section: {current_section}"
                    }
                ])
            else:
                # Current section is done or no current section - move to next
                possible_actions.extend([
                    {
                        "type": "start_next_section",
                        "description": "Move to next audit section"
                    }
                ])
                
        elif status == WorkflowStatus.CONSOLIDATING:
            # Use deterministic action selection to prevent loops
            possible_actions.extend([
                {
                    "type": "call_consolidator",
                    "description": "Generate final consolidated report"
                }
            ])
            
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            # These are final states - no actions needed
            possible_actions.extend([
                {
                    "type": "finish_workflow",
                    "description": "Workflow is in final state"
                }
            ])# Always include some universal actions, but be smart about it
        # Don't offer query_knowledge if we're just waiting for plan approval or already awaiting approval
        has_plan = bool(state.get("audit_plan"))
        plan_approved_check = state.get("plan_approved", False)
        
        # Don't offer knowledge query if:
        # 1. We're waiting for plan approval 
        # 2. We're already in AWAITING_APPROVAL status
        # 3. We're already completed/failed/cancelled
        # 4. We're in CONSOLIDATING status (to prevent loops at the end)
        # 5. All sections are complete and we should be consolidating        generated_sections = state.get("generated_sections", {})
        audit_plan = state.get("audit_plan", {})
        plan_sections = audit_plan.get("sections", []) if audit_plan else []
        planned_cycles = [section.get("cycle") for section in plan_sections if section and section.get("cycle")]
        all_sections_complete = all(cycle in generated_sections for cycle in planned_cycles) if planned_cycles else False
        
        should_skip_knowledge = (
            (has_plan and not plan_approved_check and status == WorkflowStatus.PLANNING) or
            status == WorkflowStatus.AWAITING_APPROVAL or
            status == WorkflowStatus.CONSOLIDATING or
            status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] or
            all_sections_complete  # Don't query knowledge when all sections are done
        )
        
        if not should_skip_knowledge and status in [WorkflowStatus.PENDING, WorkflowStatus.ANALYZING]:
            # Only offer knowledge query during active phases
            possible_actions.extend([
                {
                    "type": "query_knowledge",
                    "description": "Query knowledge base for additional information"
                }
            ])
        
        return possible_actions
    
    def _get_fallback_action(self, state: AgentState) -> Dict[str, Any]:
        """
        Get a safe fallback action based on current state
        
        Args:
            state: Current workflow state
            
        Returns:
            Safe fallback action that will progress the workflow
        """
        status = state["status"]
        
        if status == WorkflowStatus.PLANNING:
            return {
                "type": "create_audit_plan",
                "description": "Fallback: Create basic audit plan",
                "parameters": {}
            }
        elif status == WorkflowStatus.AWAITING_APPROVAL:
            # Check if plan is actually approved before starting processing
            if state.get("plan_approved", False):
                return {
                    "type": "start_processing",
                    "description": "Fallback: Start processing phase",
                    "parameters": {}
                }
            else:
                return {
                    "type": "wait_human_approval",
                    "description": "Fallback: Wait for human approval",
                    "parameters": {}
                }
        elif status == WorkflowStatus.ANALYZING:
            return {
                "type": "consolidate_findings",
                "description": "Fallback: Consolidate current findings",
                "parameters": {}
            }
        else:
            # For any other status (including ERROR, COMPLETED)
            return {
                "type": "finalize_workflow",
                "description": "Fallback: Generate final report",
                "parameters": {}
            }    
    async def _execute_action(self, state: AgentState, action: Dict[str, Any]) -> AgentState:
        """
        Execute the chosen action
        
        Args:
            state: Current workflow state
            action: Action to execute
            
        Returns:
            Updated state after action execution        """
        action_type = action.get("type")
        parameters = action.get("parameters", {})
        
        logger.info(f"[{self.agent_id}] Executing action: {action_type}")
        
        try:
            if action_type == "create_audit_plan":
                from .planner_agent import PlannerAgent
                planner = PlannerAgent(self.llm_config)
                state = await planner.execute(state)
                # After creating plan, ALWAYS move to awaiting approval if plan was created
                if state.get("audit_plan"):
                    state["awaiting_human"] = True  # Set the flag for frontend
                    state["plan_approved"] = False  # Ensure it's not approved yet
                    state = self.update_status(state, WorkflowStatus.AWAITING_APPROVAL)
                    state = self.log_message(state, "Audit plan created, awaiting human approval")
                else:
                    state = self.log_error(state, "Failed to create audit plan")
                    state = self.update_status(state, WorkflowStatus.FAILED)
            elif action_type == "start_processing":
                # Only start processing if plan is approved
                if not state.get("plan_approved", False):
                    state = self.log_message(state, "Cannot start processing - audit plan not yet approved")
                    state["awaiting_human"] = True
                    state = self.update_status(state, WorkflowStatus.AWAITING_APPROVAL)
                else:
                    state["awaiting_human"] = False  # Clear the flag since we're proceeding
                    state = self.update_status(state, WorkflowStatus.ANALYZING)
                    state = self.log_message(state, "Started processing phase - now analyzing")
                    # Automatically start the first section
                    state = self._start_next_section(state)
                
            elif action_type == "collect_evidence":
                from .analyzer_agent import AnalyzerAgent
                analyzer = AnalyzerAgent(self.llm_config)
                state = await analyzer.execute(state)
                
            elif action_type == "assess_risks":
                # Assess risks based on current findings
                findings = state.get("findings", [])
                if findings:
                    risk_assessment = await self._assess_workflow_risks(findings)
                    state["risk_assessment"] = risk_assessment
                    state = self.log_message(state, f"Assessed {len(findings)} findings for risks")
                else:
                    state = self.log_message(state, "No findings available for risk assessment")
                    
            elif action_type == "consolidate_findings":
                from .consolidator_agent import ConsolidatorAgent
                consolidator = ConsolidatorAgent(self.llm_config)
                state = await consolidator.execute(state)
                
            elif action_type == "query_knowledge":
                query = parameters.get("query", "general audit information")
                knowledge_result = await self._query_knowledge_base(query)
                state.setdefault("knowledge_queries", []).append({
                    "query": query,
                    "result": knowledge_result,
                    "timestamp": datetime.now().isoformat()                })
                state = self.log_message(state, f"Queried knowledge base: {query}")
                
            elif action_type == "finalize_workflow":
                # Only finalize if we have processed all sections OR if we're forcing completion
                generated_sections = state.get("generated_sections", {})
                audit_plan = state.get("audit_plan", {})
                
                # Safe access to plan sections - handle case where audit_plan is None
                plan_sections = audit_plan.get("sections", []) if audit_plan else []
                planned_cycles = [section.get("cycle") for section in plan_sections if section and section.get("cycle")]
                
                # Check if all planned sections are completed
                all_sections_done = all(cycle in generated_sections for cycle in planned_cycles) if planned_cycles else True
                
                if not all_sections_done and not parameters.get("force", False):
                    # Don't finalize yet, still have sections to process
                    state = self.log_message(state, f"Cannot finalize yet - {len(generated_sections)}/{len(planned_cycles)} sections completed")
                    # Try to start next section instead
                    state = self._start_next_section(state)
                else:# Ensure we have a final report before completing
                    if not state.get("final_report"):
                        if generated_sections:
                            state["final_report"] = self._create_simple_final_report(state, generated_sections)
                            state = self.log_message(state, "Created simple final report for workflow completion")
                        else:
                            state["final_report"] = "# Audit Report\n\nWorkflow completed but no sections were generated."
                            state = self.log_message(state, "Created minimal final report for workflow completion")
                    
                    state = self.update_status(state, WorkflowStatus.COMPLETED)
                    state = self.log_message(state, f"Workflow completed with final report ({len(state.get('final_report', ''))} characters)")
                
            elif action_type == "finish_workflow":
                # Workflow is already in final state, just log and return
                state = self.log_message(state, f"Workflow is already in final state: {state.get('status')}")
                
            elif action_type == "start_next_section":
                state = self._start_next_section(state)
                
            elif action_type == "call_analyzer":
                from .analyzer_agent import AnalyzerAgent
                analyzer = AnalyzerAgent(self.llm_config)
                state = await analyzer.execute(state)
                
            elif action_type == "call_writer":
                from .writer_agent import WriterAgent
                writer = WriterAgent(self.llm_config)
                state = await writer.execute(state)
                
            elif action_type == "call_consolidator":
                from .consolidator_agent import ConsolidatorAgent
                consolidator = ConsolidatorAgent(self.llm_config)
                state = await consolidator.execute(state)
                # After consolidation, check if final report was generated and mark complete
                if state.get("final_report"):
                    state = self.update_status(state, WorkflowStatus.COMPLETED)
                    state = self.log_message(state, "Final report generated, workflow completed")
                else:
                    state = self.log_message(state, "Consolidator executed but no final report generated")
                
            elif action_type == "wait_human_approval":
                state["awaiting_human"] = True
                state = self.update_status(state, WorkflowStatus.AWAITING_APPROVAL)
                
            elif action_type == "complete_workflow":
                # Redirect to finalize_workflow with force flag
                return await self._execute_action(state, {"type": "finalize_workflow", "parameters": {"force": True}})
                
            elif action_type == "cancel_workflow":
                state = self.update_status(state, WorkflowStatus.CANCELLED)
                
            elif action_type == "error_recovery":
                state = self._handle_error_recovery(state, parameters)
                
            else:
                # Unknown action type - log error and try to finalize
                logger.error(f"Unknown action type: {action_type}")
                state = self.log_error(state, f"Unknown action type: {action_type}")
                
                # Force finalization without recursion to prevent infinite loops
                if not state.get("final_report"):
                    state["final_report"] = f"# Error Report\n\nWorkflow failed due to unknown action type: {action_type}"
                state = self.update_status(state, WorkflowStatus.COMPLETED)
                state = self.log_message(state, "Workflow forcefully completed due to unknown action")
        
        except Exception as e:
            logger.error(f"Failed to execute action {action_type}: {str(e)}")
            state = self.log_error(state, f"Failed to execute action {action_type}: {str(e)}")
            
            # Force completion without recursion
            if not state.get("final_report"):
                state["final_report"] = f"# Error Report\n\nWorkflow failed with error: {str(e)}"
            state = self.update_status(state, WorkflowStatus.COMPLETED)
            state = self.log_message(state, "Workflow forcefully completed due to execution error")        
        return state
    
    def _start_next_section(self, state: AgentState) -> AgentState:
        """Determine and start the next audit section"""
        # Get completed sections and audit plan
        generated_sections = state.get("generated_sections", {})
        audit_plan = state.get("audit_plan", {})
        plan_sections = audit_plan.get("sections", []) if audit_plan else []
        
        # Get the planned cycles from the audit plan
        planned_cycles = []
        for section in plan_sections:
            if section and section.get("cycle"):
                cycle = section.get("cycle")
                planned_cycles.append(cycle)
        
        # If no planned cycles, fall back to all cycles
        if not planned_cycles:
            planned_cycles = [cycle.value for cycle in AuditCycle]
        
        # Find next section to process from planned sections
        for cycle in planned_cycles:
            if cycle not in generated_sections:
                state["current_section"] = cycle
                state = self.update_status(state, WorkflowStatus.ANALYZING)
                state = self.log_message(state, f"Starting analysis of section: {cycle}")
                return state
        
        # All planned sections completed, move to consolidation
        state["current_section"] = None
        state = self.update_status(state, WorkflowStatus.CONSOLIDATING)
        state = self.log_message(state, "All planned sections completed, starting consolidation")
        return state
    
    def _handle_error_recovery(self, state: AgentState, parameters: Dict[str, Any]) -> AgentState:
        """Handle error recovery actions"""
        error_count = state.get("error_recovery_count", 0)
        max_errors = 3  # Maximum number of error recovery attempts
        
        if error_count >= max_errors:
            # Too many errors, force completion
            logger.warning(f"Maximum error recovery attempts ({max_errors}) reached, forcing completion")
            state = self.log_message(state, f"Maximum error recovery attempts reached, forcing workflow completion")
            
            # Create minimal final report and complete
            if not state.get("final_report"):
                state["final_report"] = "# Error Recovery Report\n\nWorkflow completed after error recovery."
            
            state = self.update_status(state, WorkflowStatus.COMPLETED)
            return state
        
        # Increment error recovery count
        state["error_recovery_count"] = error_count + 1
        
        # Log the recovery attempt
        state = self.log_message(state, f"Attempting error recovery (attempt {error_count + 1}/{max_errors})")
        
        # Reset error state and try to continue based on current status
        current_status = state["status"]
        
        if current_status == WorkflowStatus.PLANNING:
            # If planning failed, try to create a basic plan
            state = self.log_message(state, "Error recovery: Creating basic audit plan")
        elif current_status == WorkflowStatus.ANALYZING:
            # If processing failed, try to consolidate what we have
            state = self.log_message(state, "Error recovery: Attempting to consolidate current findings")
        else:
            # For any other status, try to finalize
            state = self.log_message(state, "Error recovery: Attempting to finalize workflow")
        
        return state
    
    async def _observe(self, state: AgentState) -> str:
        """
        ReAct Observation step - analyze results of the action
        
        Args:
            state: Updated workflow state after action
            
        Returns:
            Observation about the results
        """
        status = state["status"]
        recent_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]
        errors = state["errors"][-1:] if state["errors"] else []
        
        context = f"""
        Current status: {status.value}
        Recent messages: {json.dumps(recent_messages, indent=2)}
        Recent errors: {json.dumps(errors, indent=2)}
        Awaiting human: {state.get('awaiting_human', False)}
        """
        
        prompt = f"""
        Observe the results of the recent action and current state:
        
        {context}
        
        Provide a brief observation (1-2 sentences) about:
        1. What happened as a result of the action?
        2. Is the workflow progressing correctly?
        3. Are there any issues or next steps needed?
        """
        
        observation = await self.call_llm(prompt, "You are an observant orchestrator that analyzes workflow progress.")
        return observation.strip()
    
    def _create_simple_final_report(self, state: AgentState, generated_sections: Dict[str, Any]) -> str:
        """
        Create a simple final report from generated sections as a fallback
        
        Args:
            state: Current workflow state
            generated_sections: Dictionary of generated sections
            
        Returns:
            Simple final report string
        """
        enterprise_info = state.get("enterprise_info", {})
        selected_norms = state.get("selected_norms", [])
        
        # Build basic report structure
        report_lines = [
            "# Audit Conformity Report",
            "",
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Enterprise:** {enterprise_info.get('name', 'Unknown')}",
            f"**Selected Norms:** {', '.join(selected_norms) if selected_norms else 'None specified'}",
            "",
            "## Executive Summary",
            "",
            f"This audit report covers {len(generated_sections)} sections of analysis for the enterprise's conformity assessment.",
            "",
            "## Audit Sections",
            ""
        ]
        
        # Add each generated section
        for section_name, section_data in generated_sections.items():
            report_lines.append(f"### {section_name}")
            report_lines.append("")
            
            # Add analysis if available
            if isinstance(section_data, dict) and "analysis" in section_data:
                report_lines.append("**Analysis:**")
                report_lines.append(str(section_data["analysis"]))
                report_lines.append("")
            
            # Add findings if available
            if isinstance(section_data, dict) and "findings" in section_data:
                report_lines.append("**Findings:**")
                findings = section_data["findings"]
                if isinstance(findings, list):
                    for finding in findings:
                        report_lines.append(f"- {finding}")
                else:
                    report_lines.append(str(findings))
                report_lines.append("")
            
            # Add recommendations if available
            if isinstance(section_data, dict) and "recommendations" in section_data:
                report_lines.append("**Recommendations:**")
                recommendations = section_data["recommendations"]
                if isinstance(recommendations, list):
                    for rec in recommendations:
                        report_lines.append(f"- {rec}")
                else:
                    report_lines.append(str(recommendations))
                report_lines.append("")
            
            # If section_data is just a string, add it directly
            if isinstance(section_data, str):
                report_lines.append(section_data)
                report_lines.append("")
        
        # Add footer
        report_lines.extend([
            "## Report Generation Notes",
            "",
            "This report was generated using a simplified consolidation process.",
            "For a more comprehensive analysis, please ensure all workflow components are functioning correctly.",
            "",
            f"**Total Sections Analyzed:** {len(generated_sections)}",
            f"**Report Length:** {sum(len(line) for line in report_lines)} characters"
        ])
        
        return "\n".join(report_lines)

    async def _assess_workflow_risks(self, findings: List[Dict]) -> Dict[str, Any]:
        """
        Assess risks based on current findings
        
        Args:
            findings: List of audit findings
            
        Returns:
            Risk assessment dictionary
        """
        if not findings:
            return {"overall_risk": "low", "risk_factors": [], "recommendations": []}
        
        # Count high, medium, low severity findings
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        risk_factors = []
        
        for finding in findings:
            severity = finding.get("severity", "low").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            if finding.get("type") == "security":
                risk_factors.append("Security vulnerabilities identified")
            elif finding.get("type") == "compliance":
                risk_factors.append("Compliance gaps found")
            elif finding.get("type") == "operational":
                risk_factors.append("Operational risks detected")
        
        # Determine overall risk level
        if severity_counts["high"] > 2:
            overall_risk = "high"
        elif severity_counts["high"] > 0 or severity_counts["medium"] > 3:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        recommendations = []
        if severity_counts["high"] > 0:
            recommendations.append("Immediate action required for high-severity findings")
        if severity_counts["medium"] > 0:
            recommendations.append("Address medium-severity findings within reasonable timeframe")
        
        return {
            "overall_risk": overall_risk,
            "severity_counts": severity_counts,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "total_findings": len(findings)
        }
    
    async def _query_knowledge_base(self, query: str) -> str:
        """
        Query the knowledge base for information
        
        Args:
            query: Query string
            
        Returns:
            Knowledge base response
        """
        try:
            # For now, return a simple response
            # In future, this could connect to a real knowledge base
            return f"Knowledge base query '{query}' - No specific information available at this time."
        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}")
            return f"Knowledge base query failed: {str(e)}"
