"""
Writer Agent
============

Generates markdown report sections based on analysis results from the Analyzer agent.
Creates professional, structured audit conformity reports that will be consolidated
into the final report.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from .base_agent import BaseAgent
from .audit_types import AgentState, AgentType, WorkflowStatus, AuditCycle, AUDIT_CYCLE_TEMPLATES

logger = logging.getLogger(__name__)

class WriterAgent(BaseAgent):
    """Agent responsible for writing audit report sections"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(AgentType.WRITER, llm_config)
    
    def get_available_actions(self, state: AgentState) -> List[str]:
        """List all available actions for the writer agent"""
        return [
            "generate_section_content",
            "format_findings",
            "create_conformity_assessment",
            "write_recommendations", 
            "structure_report_section",
            "validate_content"
        ]
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Generate a markdown report section based on analysis results
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with written report section        """
        if not self.validate_state(state):
            return self.log_error(state, "Invalid state received by writer")
        
        current_section = state.get("current_section")
        if not current_section:
            return self.log_error(state, "No current section specified for writing")
        
        state = self.log_message(state, f"Writer starting report generation for section: {current_section}")
        
        try:
            # Get analysis results for current section
            analysis_context = state.get("analysis_context", {})
            section_analysis = analysis_context.get(current_section)
            
            if not section_analysis:
                return self.log_error(state, f"No analysis results found for section: {current_section}")
            
            # Get cycle template for context - use mapping for French section names
            cycle_enum = self._map_section_to_audit_cycle(current_section)
            if not cycle_enum:
                return self.log_error(state, f"Could not map section '{current_section}' to valid AuditCycle")
            
            cycle_template = AUDIT_CYCLE_TEMPLATES[cycle_enum]
            
            # Generate markdown report section
            markdown_content = await self._generate_markdown_section(
                state, cycle_template, section_analysis
            )
            
            # Store generated content
            if "generated_sections" not in state:
                state["generated_sections"] = {}
            
            state["generated_sections"][current_section] = markdown_content
            
            # Move to next section or consolidation
            next_action = self._determine_next_action(state)
            if next_action == "next_section":
                # Find next section to process
                next_section = self._find_next_section(state)
                if next_section:
                    state["current_section"] = next_section
                    state = self.update_status(state, WorkflowStatus.ANALYZING)
                    state = self.log_message(state, f"Moving to next section: {next_section}")
                else:
                    # All sections completed
                    state["current_section"] = None
                    state = self.update_status(state, WorkflowStatus.CONSOLIDATING)
                    state = self.log_message(state, "All sections completed, ready for consolidation")
            
            state = self.log_message(state, f"Report section generated for: {current_section}")
            
        except Exception as e:
            state = self.log_error(state, f"Failed to generate report section for {current_section}: {str(e)}")
        
        return state
    
    async def _generate_markdown_section(self, state: AgentState, cycle_template: Dict[str, Any], 
                                       analysis_result: Dict[str, Any]) -> str:
        """
        Generate a markdown report section based on analysis results
        
        Args:
            state: Current workflow state
            cycle_template: Template for the audit cycle
            analysis_result: Analysis results from analyzer agent
            
        Returns:
            Markdown formatted report section
        """
        selected_norms = state.get("selected_norms", [])
        enterprise_report_id = state.get("enterprise_report_id", "Unknown")
        
        # Prepare context for report generation
        context = self._prepare_writing_context(state, cycle_template, analysis_result)
        
        writing_prompt = f"""
        Generate a professional audit conformity report section in markdown format.
        
        Context Information:
        {context}
        
        Create a comprehensive, well-structured markdown report section that includes:
        
        1. **Executive Summary** - Brief overview of findings
        2. **Scope and Objectives** - What was audited and why
        3. **Methodology** - How the audit was conducted
        4. **Detailed Findings** - Specific findings with evidence
        5. **Conformity Assessment** - Compliance with norms and standards
        6. **Gap Analysis** - Identified gaps and their severity
        7. **Risk Assessment** - Identified risks and their impact
        8. **Recommendations** - Specific, actionable recommendations
        9. **Conclusion** - Summary and next steps
        
        Guidelines:
        - Use proper markdown formatting (headers, lists, tables, emphasis)
        - Be professional and objective in tone
        - Include specific evidence and references where available
        - Make recommendations actionable and specific
        - Use clear, structured presentation
        - Include severity indicators for issues (🔴 High, 🟡 Medium, 🟢 Low)
        - Reference specific norms where applicable
        
        The report should be comprehensive but concise, suitable for executive review.
        """
        
        markdown_content = await self.call_llm(
            writing_prompt,
            "You are a professional audit report writer. Create clear, comprehensive, and well-structured audit reports in markdown format."
        )
        
        # Add section metadata header
        metadata_header = self._generate_metadata_header(cycle_template, analysis_result)
        
        return metadata_header + "\n\n" + markdown_content.strip()
    
    def _prepare_writing_context(self, state: AgentState, cycle_template: Dict[str, Any], 
                               analysis_result: Dict[str, Any]) -> str:
        """Prepare context information for report writing"""
        selected_norms = state.get("selected_norms", [])
        
        context = f"""
Audit Cycle: {cycle_template['title']}

Cycle Objectives:
{json.dumps(cycle_template['objectives'], indent=2)}

Key Controls Framework:
{json.dumps(cycle_template['key_controls'], indent=2)}

Standard Tests:
{json.dumps(cycle_template['tests'], indent=2)}

Selected Conformity Norms:
{json.dumps(selected_norms, indent=2)}

Analysis Results:
Overall Rating: {analysis_result.get('conformity_assessment', {}).get('overall_rating', 'Unknown')}
Compliance Score: {analysis_result.get('conformity_assessment', {}).get('compliance_score', 'N/A')}

Detailed Findings:
{json.dumps(analysis_result.get('detailed_findings', []), indent=2)}

Gap Analysis:
{json.dumps(analysis_result.get('gap_analysis', []), indent=2)}

Identified Risks:
{json.dumps(analysis_result.get('risks', []), indent=2)}

Strengths:
{json.dumps(analysis_result.get('strengths', []), indent=2)}

Evidence Quality Assessment:
{json.dumps(analysis_result.get('evidence_quality', {}), indent=2)}

Documents Analyzed: {analysis_result.get('documents_analyzed', 0)}
"""
        return context
    
    def _generate_metadata_header(self, cycle_template: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """Generate metadata header for the report section"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        header = f"""<!--
Report Section Metadata
Generated: {datetime.now().isoformat()}
Cycle: {cycle_template['title']}
Agent: {self.agent_id}
Analysis Date: {analysis_result.get('analysis_date', 'Unknown')}
Documents Analyzed: {analysis_result.get('documents_analyzed', 0)}
Overall Rating: {analysis_result.get('conformity_assessment', {}).get('overall_rating', 'Unknown')}
-->"""
        
        return header
    
    def _determine_next_action(self, state: AgentState) -> str:
        """Determine what should happen after writing this section"""
        audit_plan = state.get("audit_plan", {})
        plan_sections = audit_plan.get("sections", [])
        generated_sections = state.get("generated_sections", {})
        
        # Check if all planned sections are completed
        planned_cycles = {section.get("cycle") for section in plan_sections}
        completed_cycles = set(generated_sections.keys())
        
        if planned_cycles.issubset(completed_cycles):
            return "consolidate"
        else:
            return "next_section"
    
    def _find_next_section(self, state: AgentState) -> Optional[str]:
        """Find the next section to process based on the audit plan"""
        audit_plan = state.get("audit_plan", {})
        plan_sections = audit_plan.get("sections", [])
        generated_sections = state.get("generated_sections", {})
        
        # Find next unprocessed section based on priority order
        priority_order = audit_plan.get("priority_order", [])
        
        # First, try to follow priority order
        for cycle in priority_order:
            if cycle not in generated_sections:
                return cycle
        
        # If priority order doesn't help, find any unprocessed section
        for section in plan_sections:
            cycle = section.get("cycle")
            if cycle and cycle not in generated_sections:
                return cycle
        
        return None
    
    def _map_section_to_audit_cycle(self, section_name: str) -> Optional[AuditCycle]:
        """
        Map French section names to AuditCycle enum values
        
        Args:
            section_name: Section name from the audit plan (may be in French)
            
        Returns:
            Corresponding AuditCycle enum or None if not found
        """
        # Normalize the section name (lowercase, remove accents, etc.)
        normalized = section_name.lower().strip()
          # French to enum mapping
        section_mappings = {
            "cycle trésorerie": AuditCycle.TRESORERIE,
            "trésorerie": AuditCycle.TRESORERIE,
            "tresorerie": AuditCycle.TRESORERIE,
            "cycle ventes": AuditCycle.VENTES_CLIENTS,
            "ventes clients": AuditCycle.VENTES_CLIENTS,
            "ventes": AuditCycle.VENTES_CLIENTS,
            "cycle achats": AuditCycle.ACHATS_FOURNISSEURS,
            "achats fournisseurs": AuditCycle.ACHATS_FOURNISSEURS,
            "achats": AuditCycle.ACHATS_FOURNISSEURS,
            "immobilisations": AuditCycle.IMMOBILISATIONS,
            "stocks": AuditCycle.STOCKS,
            "paie personnel": AuditCycle.PAIE_PERSONNEL,
            "paie": AuditCycle.PAIE_PERSONNEL,
            "impots taxes": AuditCycle.IMPOTS_TAXES,
            "impôts taxes": AuditCycle.IMPOTS_TAXES,
            "cycle impôts et taxes": AuditCycle.IMPOTS_TAXES,
            "cycle impots et taxes": AuditCycle.IMPOTS_TAXES,
            "impôts et taxes": AuditCycle.IMPOTS_TAXES,
            "impots et taxes": AuditCycle.IMPOTS_TAXES,
            "impots": AuditCycle.IMPOTS_TAXES,
            "fiscalité": AuditCycle.IMPOTS_TAXES,
        }
        
        # Try direct mapping first
        if normalized in section_mappings:
            return section_mappings[normalized]
        
        # Try partial matching for more flexibility
        for key, cycle in section_mappings.items():
            if key in normalized or normalized in key:
                return cycle
        
        # Try enum value direct matching
        try:
            return AuditCycle(normalized)
        except ValueError:
            pass
        
        # Try enum name matching
        for cycle in AuditCycle:
            if cycle.value == normalized or cycle.name.lower() == normalized:
                return cycle
        
        logger.warning(f"Could not map section '{section_name}' to any AuditCycle")
        return None
    
    def validate_state(self, state: AgentState) -> bool:
        """Validate state has required fields for writer"""
        if not super().validate_state(state):
            return False
        
        required_fields = ["current_section", "analysis_context"]
        for field in required_fields:
            if field not in state:
                logger.error(f"Missing required field for writer: {field}")
                return False
        
        return True
