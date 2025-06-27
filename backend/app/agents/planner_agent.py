"""
Planner Agent
=============

Creates detailed audit plans based on the uploaded enterprise report and selected norms.
Follows the audit cycle template and provides structured plans for human approval.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from .base_agent import BaseAgent
from .audit_types import AgentState, AgentType, WorkflowStatus, AuditCycle, AUDIT_CYCLE_TEMPLATES

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """Agent responsible for creating audit plans"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(AgentType.PLANNER, llm_config)

    def get_available_actions(self, state: AgentState) -> List[str]:
        """List all available actions for the planner agent"""
        return [
            "analyze_enterprise_report",
            "create_audit_plan",
            "validate_plan",
            "refine_plan"
        ]

    async def execute(self, state: AgentState) -> AgentState:
        """
        Create an audit plan based on enterprise report and selected norms
        """
        # Only log errors and high-level events
        if not self.validate_state(state):
            logger.error("[PlannerAgent] State validation failed")
            return self.log_error(state, "Invalid state received by planner")

        # High-level log for plan creation start
        logger.info(f"[PlannerAgent] Creating audit plan for workflow {state.get('workflow_id')}")
        state = self.log_message(state, "Planner agent starting audit plan creation")

        try:
            # Analyze the enterprise report
            report_analysis = await self._analyze_enterprise_report(state)

            # Create audit plan based on analysis and selected norms
            audit_plan = await self._create_audit_plan(state, report_analysis)

            # Store the plan in state
            state["audit_plan"] = audit_plan
            state["plan_approved"] = False

            # Update status to awaiting approval
            state = self.update_status(state, WorkflowStatus.AWAITING_APPROVAL)
            state = self.log_message(state, f"Audit plan created with {len(audit_plan['sections'])} sections")
        except Exception as e:
            logger.error(f"[PlannerAgent] Failed to create audit plan: {str(e)}")
            state = self.log_error(state, f"Failed to create audit plan: {str(e)}")
            state = self.update_status(state, WorkflowStatus.FAILED)

        return state
    
    async def _analyze_enterprise_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze the uploaded enterprise report to understand its structure and content
        """
        enterprise_report_id = state.get("enterprise_report_id")
        if not enterprise_report_id:
            logger.error("[PlannerAgent] No enterprise report ID provided")
            raise ValueError("No enterprise report ID provided")

        # Retrieve document information and content
        documents = await self.retrieve_documents(
            query="structure overview table of contents sections",
            document_id=enterprise_report_id
        )

        # Analyze document structure
        analysis_prompt = f"""
        Analyze this enterprise audit report to understand its structure and content.
        Document information:
        {json.dumps(documents[:5], indent=2)}  # Limit to first 5 chunks for analysis
        Provide analysis in JSON format:
        {{
            "document_type": "type of report (internal audit, external audit, etc.)",
            "enterprise_sector": "business sector/industry",
            "main_sections": ["list", "of", "main", "sections"],
            "financial_cycles_present": ["cycles", "identified", "in", "report"],
            "complexity_level": "simple/medium/complex",
            "special_considerations": ["any", "special", "considerations"],
            "recommended_focus_areas": ["areas", "that", "need", "attention"]
        }}
        """

        analysis_response = await self.call_llm(
            analysis_prompt,
            "You are an expert auditor analyzing enterprise reports. Always respond with valid JSON."
        )

        try:
            analysis = json.loads(analysis_response.strip())
            return analysis
        except json.JSONDecodeError:
            # Fallback analysis if JSON parsing fails
            return {
                "document_type": "enterprise_audit_report",
                "enterprise_sector": "unknown",
                "main_sections": ["financial_statements", "operations", "controls"],
                "financial_cycles_present": ["ventes_clients", "achats_fournisseurs", "tresorerie"],
                "complexity_level": "medium",
                "special_considerations": [],
                "recommended_focus_areas": ["all_cycles"]
            }
    
    async def _create_audit_plan(self, state: AgentState, report_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a detailed audit plan based on the report analysis and selected norms
        
        Args:
            state: Current workflow state
            report_analysis: Analysis of the enterprise report
            
        Returns:
            Detailed audit plan
        """
        selected_norms = state.get("selected_norms", [])
        
        # Determine which audit cycles are relevant
        relevant_cycles = self._determine_relevant_cycles(report_analysis, selected_norms)
        
        # Create detailed plan
        plan_prompt = f"""
        Create a detailed audit conformity plan based on:
        
        Enterprise Report Analysis:
        {json.dumps(report_analysis, indent=2)}
        
        Selected Norms for Conformity Check:
        {json.dumps(selected_norms, indent=2)}
        
        Relevant Audit Cycles:
        {json.dumps(relevant_cycles, indent=2)}
        
        Create a comprehensive plan that includes:
        1. Overall strategy and approach
        2. Prioritization of audit cycles based on risk and importance
        3. Specific focus areas for each cycle
        4. Estimated timeline and effort
        5. Key risks and mitigation strategies
        
        Respond with JSON:
        {{
            "plan_id": "unique_plan_identifier",
            "created_at": "current_timestamp",
            "overall_strategy": "description of overall approach",
            "priority_order": ["ordered", "list", "of", "cycles"],
            "estimated_duration": "estimated time",
            "sections": [
                {{
                    "cycle": "cycle_name",
                    "priority": "high/medium/low",
                    "focus_areas": ["specific", "areas", "to", "focus"],
                    "expected_findings": ["potential", "issues", "to", "look", "for"],
                    "conformity_checks": ["specific", "norm", "requirements"],
                    "estimated_effort": "time estimate"
                }}
            ],
            "risk_assessment": {{
                "high_risk_areas": ["areas", "with", "high", "risk"],
                "mitigation_strategies": ["strategies", "to", "mitigate", "risks"]
            }},
            "success_criteria": ["criteria", "for", "successful", "audit"]
        }}        """
        
        plan_response = await self.call_llm(
            plan_prompt,
            "You are an expert audit planner. Create comprehensive, actionable audit plans. Always respond with valid JSON."
        )
        
        try:
            # Clean and extract JSON from the response
            plan_response = plan_response.strip()
            
            # Look for JSON in the response (handle cases where LLM adds extra text)
            if "```json" in plan_response:
                start = plan_response.find("```json") + 7
                end = plan_response.find("```", start)
                if end > start:
                    plan_response = plan_response[start:end].strip()
            elif "{" in plan_response and "}" in plan_response:
                start = plan_response.find("{")
                end = plan_response.rfind("}") + 1
                plan_response = plan_response[start:end]
            
            audit_plan = json.loads(plan_response)
            
            # Add timestamp and ensure required fields
            audit_plan["created_at"] = datetime.now().isoformat()
            audit_plan["plan_id"] = f"plan_{state['workflow_id']}_{int(datetime.now().timestamp())}"
            
            # Validate and ensure all relevant cycles are included
            self._validate_and_complete_plan(audit_plan, relevant_cycles)
            
            return audit_plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse audit plan JSON: {str(e)}")
            logger.error(f"Raw LLM response: {plan_response[:500]}...")  # Log first 500 chars for debugging
            # Create a fallback plan
            return self._create_fallback_plan(relevant_cycles)
    
    def _determine_relevant_cycles(self, report_analysis: Dict[str, Any], selected_norms: List[str]) -> List[Dict[str, Any]]:
        """
        Determine which audit cycles are relevant based on the report and norms
        
        Args:
            report_analysis: Analysis of the enterprise report
            selected_norms: List of selected conformity norms
            
        Returns:
            List of relevant audit cycles with their templates
        """
        relevant_cycles = []
        
        # Get cycles mentioned in the report
        financial_cycles = report_analysis.get("financial_cycles_present", [])
        enterprise_sector = report_analysis.get("enterprise_sector", "unknown")
        
        # Map all cycles as potentially relevant (can be filtered later)
        for cycle in AuditCycle:
            cycle_template = AUDIT_CYCLE_TEMPLATES[cycle].copy()
            cycle_template["cycle_enum"] = cycle.value
            
            # Determine relevance score
            relevance_score = self._calculate_cycle_relevance(
                cycle, financial_cycles, enterprise_sector, selected_norms
            )
            cycle_template["relevance_score"] = relevance_score
            
            relevant_cycles.append(cycle_template)
        
        # Sort by relevance score (highest first)
        relevant_cycles.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return relevant_cycles
    
    def _calculate_cycle_relevance(self, cycle: AuditCycle, financial_cycles: List[str], 
                                 enterprise_sector: str, selected_norms: List[str]) -> float:
        """Calculate relevance score for an audit cycle"""
        score = 0.5  # Base score
        
        # Higher score if explicitly mentioned in report
        if cycle.value in financial_cycles:
            score += 0.3
        
        # Sector-specific adjustments
        if enterprise_sector in ["manufacturing", "retail"] and cycle in [AuditCycle.STOCKS, AuditCycle.ACHATS_FOURNISSEURS]:
            score += 0.2
        elif enterprise_sector in ["services", "consulting"] and cycle in [AuditCycle.PAIE_PERSONNEL, AuditCycle.VENTES_CLIENTS]:
            score += 0.2
        
        # Norm-specific adjustments
        if "ISO" in selected_norms and cycle in [AuditCycle.STOCKS, AuditCycle.IMMOBILISATIONS]:
            score += 0.15
        if "IFRS" in selected_norms and cycle == AuditCycle.VENTES_CLIENTS:
            score += 0.15
        
        # Always include treasury and taxes as they're universally important
        if cycle in [AuditCycle.TRESORERIE, AuditCycle.IMPOTS_TAXES]:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _validate_and_complete_plan(self, audit_plan: Dict[str, Any], relevant_cycles: List[Dict[str, Any]]) -> None:
        """Validate and complete the audit plan"""
        if "sections" not in audit_plan:
            audit_plan["sections"] = []
        
        # Ensure all high-relevance cycles are included
        existing_cycles = {section.get("cycle") for section in audit_plan["sections"]}
        
        for cycle_info in relevant_cycles:
            if cycle_info["relevance_score"] >= 0.7 and cycle_info["cycle_enum"] not in existing_cycles:
                # Add missing high-relevance cycle
                audit_plan["sections"].append({
                    "cycle": cycle_info["cycle_enum"],
                    "priority": "high" if cycle_info["relevance_score"] >= 0.8 else "medium",
                    "focus_areas": cycle_info["objectives"],
                    "expected_findings": ["Conformity with established procedures", "Adequacy of controls"],
                    "conformity_checks": cycle_info["key_controls"][:3],  # First 3 controls
                    "estimated_effort": "2-4 hours"
                })
    
    def _create_fallback_plan(self, relevant_cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a fallback audit plan if JSON parsing fails"""
        return {
            "plan_id": f"fallback_plan_{int(datetime.now().timestamp())}",
            "created_at": datetime.now().isoformat(),
            "overall_strategy": "Comprehensive audit of all major financial cycles with focus on conformity",
            "priority_order": [cycle["cycle_enum"] for cycle in relevant_cycles[:5]],
            "estimated_duration": "2-3 weeks",
            "sections": [
                {
                    "cycle": cycle["cycle_enum"],
                    "priority": "high" if cycle["relevance_score"] >= 0.8 else "medium",
                    "focus_areas": cycle["objectives"][:2],
                    "expected_findings": ["Control effectiveness", "Conformity assessment"],
                    "conformity_checks": cycle["key_controls"][:2],
                    "estimated_effort": "2-4 hours"
                }
                for cycle in relevant_cycles[:7]  # Include top 7 cycles
            ],
            "risk_assessment": {
                "high_risk_areas": ["Revenue recognition", "Asset valuation", "Internal controls"],
                "mitigation_strategies": ["Detailed testing", "Management interviews", "Documentation review"]
            },
            "success_criteria": ["All cycles reviewed", "Conformity gaps identified", "Recommendations provided"]
        }
