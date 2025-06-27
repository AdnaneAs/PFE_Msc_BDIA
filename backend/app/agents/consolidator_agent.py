"""
Consolidator Agent
==================

Consolidates all individual audit section reports into a comprehensive final 
audit conformity report. Creates executive summary, overall recommendations,
and final conclusions.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from .base_agent import BaseAgent
from .audit_types import AgentState, AgentType, WorkflowStatus

logger = logging.getLogger(__name__)

class ConsolidatorAgent(BaseAgent):
    """Agent responsible for consolidating final audit report"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(AgentType.CONSOLIDATOR, llm_config)
    
    def get_available_actions(self, state: AgentState) -> List[str]:
        """List all available actions for the consolidator agent"""
        return [
            "consolidate_sections",
            "generate_executive_summary",
            "create_overall_assessment",
            "compile_recommendations",
            "generate_final_report",
            "validate_completeness"
        ]
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Consolidate all section reports into a final comprehensive report
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final consolidated report
        """
        if not self.validate_state(state):
            return self.log_error(state, "Invalid state received by consolidator")
        
        state = self.log_message(state, "Consolidator starting final report generation")
        
        try:
            generated_sections = state.get("generated_sections", {})
            
            if not generated_sections:
                return self.log_error(state, "No generated sections found for consolidation")
            
            # Analyze all sections for overall insights
            overall_analysis = await self._analyze_overall_conformity(state, generated_sections)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(state, overall_analysis)
            
            # Create final consolidated report
            final_report = await self._generate_final_report(
                state, generated_sections, overall_analysis, executive_summary
            )
              # Store final report
            state["final_report"] = final_report
            
            # Mark workflow as completed
            state = self.update_status(state, WorkflowStatus.COMPLETED)
            state = self.log_message(state, f"Final audit report generated with {len(generated_sections)} sections")
            state = self.log_message(state, f"Final report length: {len(final_report) if final_report else 0} characters")
            
        except Exception as e:
            state = self.log_error(state, f"Failed to consolidate final report: {str(e)}")
            
            # Create a fallback final report if consolidation fails
            generated_sections = state.get("generated_sections", {})
            if generated_sections:
                fallback_report = self._create_fallback_report(state, generated_sections)
                state["final_report"] = fallback_report
                state = self.log_message(state, "Created fallback final report due to consolidation error")
                state = self.update_status(state, WorkflowStatus.COMPLETED)
            else:
                state = self.update_status(state, WorkflowStatus.FAILED)
        
        return state
    
    async def _analyze_overall_conformity(self, state: AgentState, 
                                        generated_sections: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze overall conformity across all audit sections
        
        Args:
            state: Current workflow state
            generated_sections: All generated section reports
            
        Returns:
            Overall analysis results
        """
        selected_norms = state.get("selected_norms", [])
        audit_plan = state.get("audit_plan", {})
        analysis_context = state.get("analysis_context", {})
        
        # Extract key metrics from individual analyses
        section_summaries = []
        overall_ratings = []
        all_risks = []
        all_recommendations = []
        
        for section_name, content in generated_sections.items():
            section_analysis = analysis_context.get(section_name, {})
            
            if section_analysis:
                conformity_assessment = section_analysis.get("conformity_assessment", {})
                overall_rating = conformity_assessment.get("overall_rating", "unknown")
                compliance_score = conformity_assessment.get("compliance_score", "0")
                
                overall_ratings.append(overall_rating)
                
                section_summaries.append({
                    "section": section_name,
                    "rating": overall_rating,
                    "score": compliance_score,
                    "summary": conformity_assessment.get("summary", "No summary available")
                })
                
                # Collect risks and recommendations
                risks = section_analysis.get("risks", [])
                all_risks.extend(risks)
                
                findings = section_analysis.get("detailed_findings", [])
                for finding in findings:
                    if finding.get("recommendation"):
                        all_recommendations.append({
                            "section": section_name,
                            "recommendation": finding["recommendation"],
                            "priority": self._assess_priority(finding)
                        })
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
        Analyze the overall conformity status across all audit sections.
        
        Selected Norms for Conformity:
        {json.dumps(selected_norms, indent=2)}
        
        Section Summaries:
        {json.dumps(section_summaries, indent=2)}
        
        All Identified Risks:
        {json.dumps(all_risks[:10], indent=2)}  # Limit to top 10 risks
        
        All Recommendations:
        {json.dumps(all_recommendations[:15], indent=2)}  # Limit to top 15 recommendations
        
        Provide overall analysis in JSON format:
        {{
            "overall_conformity_status": "compliant/partially_compliant/non_compliant",
            "overall_score": "0-100 calculated from individual scores",
            "key_insights": [
                "insight 1",
                "insight 2",
                "insight 3"
            ],
            "critical_issues": [
                {{
                    "issue": "description",
                    "affected_sections": ["sections"],
                    "severity": "high/medium/low",
                    "immediate_action_required": true/false
                }}
            ],
            "compliance_by_norm": [
                {{
                    "norm": "norm name",
                    "compliance_status": "compliant/partial/non_compliant",
                    "gaps": ["gap1", "gap2"]
                }}
            ],
            "strengths": ["organizational strengths identified"],
            "improvement_areas": ["areas needing improvement"],
            "risk_profile": {{
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "top_risks": ["risk1", "risk2", "risk3"]
            }},
            "priority_actions": [
                {{
                    "action": "action description",
                    "timeline": "immediate/short-term/long-term",
                    "impact": "high/medium/low",
                    "effort": "high/medium/low"
                }}
            ]
        }}
        """
        
        analysis_response = await self.call_llm(
            analysis_prompt,
            "You are an expert auditor creating overall assessments. Be comprehensive and strategic. Always respond with valid JSON."
        )
        
        try:
            overall_analysis = json.loads(analysis_response.strip())
            overall_analysis["analysis_date"] = datetime.now().isoformat()
            overall_analysis["sections_analyzed"] = list(generated_sections.keys())
            return overall_analysis
        except json.JSONDecodeError:
            # Fallback analysis
            return self._create_fallback_overall_analysis(section_summaries, all_risks, all_recommendations)
    
    def _assess_priority(self, finding: Dict[str, Any]) -> str:
        """Assess priority of a finding based on its characteristics"""
        conformity_status = finding.get("conformity_status", "").lower()
        impact = finding.get("impact", "").lower()
        
        if "non_compliant" in conformity_status or "high" in impact:
            return "high"
        elif "needs_improvement" in conformity_status or "medium" in impact:
            return "medium"
        else:
            return "low"
    
    async def _generate_executive_summary(self, state: AgentState, overall_analysis: Dict[str, Any]) -> str:
        """Generate executive summary for the final report"""
        enterprise_report_id = state.get("enterprise_report_id", "Unknown")
        selected_norms = state.get("selected_norms", [])
        
        summary_prompt = f"""
        Create a professional executive summary for an audit conformity report.
        
        Context:
        - Enterprise Report ID: {enterprise_report_id}
        - Norms Assessed: {selected_norms}
        - Overall Status: {overall_analysis.get('overall_conformity_status', 'Unknown')}
        - Overall Score: {overall_analysis.get('overall_score', 'N/A')}
        
        Key Insights:
        {json.dumps(overall_analysis.get('key_insights', []), indent=2)}
        
        Critical Issues:
        {json.dumps(overall_analysis.get('critical_issues', []), indent=2)}
        
        Top Risks:
        {json.dumps(overall_analysis.get('risk_profile', {}).get('top_risks', []), indent=2)}
        
        Create a concise but comprehensive executive summary (300-500 words) that:
        1. States the purpose and scope of the audit
        2. Summarizes overall conformity status
        3. Highlights key findings and critical issues
        4. Mentions top risks and their implications
        5. Provides high-level recommendations
        6. Concludes with next steps
        
        Write in professional, clear language suitable for senior management.
        """
        
        executive_summary = await self.call_llm(
            summary_prompt,
            "You are writing an executive summary for senior management. Be clear, concise, and professional."
        )
        
        return executive_summary.strip()
    
    async def _generate_final_report(self, state: AgentState, generated_sections: Dict[str, str],
                                   overall_analysis: Dict[str, Any], executive_summary: str) -> str:
        """Generate the final consolidated audit report"""
        enterprise_report_id = state.get("enterprise_report_id", "Unknown")
        selected_norms = state.get("selected_norms", [])
        audit_plan = state.get("audit_plan", {})
        
        # Create report header with metadata
        report_header = self._generate_report_header(state, overall_analysis)
        
        # Generate table of contents
        toc = self._generate_table_of_contents(generated_sections)
        
        # Create final report structure
        final_report_prompt = f"""
        Create a comprehensive final audit conformity report by consolidating all sections.
        
        Report Header:
        {report_header}
        
        Executive Summary:
        {executive_summary}
        
        Overall Analysis:
        {json.dumps(overall_analysis, indent=2)}
        
        Individual Section Reports:
        {self._prepare_sections_summary(generated_sections)}
        
        Create a professional, well-structured final report in markdown format that includes:
        
        1. **Cover Page** - Title, date, scope
        2. **Table of Contents** - {toc}
        3. **Executive Summary** - Use provided summary
        4. **Audit Scope and Methodology** - What was audited and how
        5. **Overall Conformity Assessment** - Summary of overall status
        6. **Detailed Section Reports** - Include all individual sections
        7. **Consolidated Findings and Recommendations** - Prioritized actions
        8. **Risk Assessment Summary** - Overall risk profile
        9. **Implementation Roadmap** - Timeline for recommendations
        10. **Conclusion and Next Steps** - Final conclusions
        11. **Appendices** - Supporting information
        
        Ensure:
        - Professional formatting with clear hierarchy
        - Consistent use of markdown
        - Executive-level presentation
        - Actionable recommendations
        - Clear priority indicators
        """
        
        final_report = await self.call_llm(
            final_report_prompt,
            "You are creating a comprehensive final audit report. Ensure professional quality suitable for board-level review."
        )
        
        # Combine all elements
        complete_report = f"""{report_header}

{final_report.strip()}

---

## Individual Section Reports

{self._combine_section_reports(generated_sections)}

---

*Report generated by AI Audit System on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return complete_report
    
    def _generate_report_header(self, state: AgentState, overall_analysis: Dict[str, Any]) -> str:
        """Generate report header with metadata"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        header = f"""# Audit Conformity Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Workflow ID:** {state.get('workflow_id', 'Unknown')}  
**Enterprise Report:** {state.get('enterprise_report_id', 'Unknown')}  
**Norms Assessed:** {', '.join(state.get('selected_norms', []))}  
**Overall Status:** {overall_analysis.get('overall_conformity_status', 'Unknown')}  
**Compliance Score:** {overall_analysis.get('overall_score', 'N/A')}/100  

---"""
        
        return header
    
    def _generate_table_of_contents(self, generated_sections: Dict[str, str]) -> str:
        """Generate table of contents"""
        sections = list(generated_sections.keys())
        toc_items = []
        
        for i, section in enumerate(sections, 1):
            # Convert section key to readable title
            title = section.replace("_", " ").title()
            toc_items.append(f"{i}. {title}")
        
        return "\n".join(toc_items)
    
    def _prepare_sections_summary(self, generated_sections: Dict[str, str]) -> str:
        """Prepare a summary of sections for the prompt"""
        summaries = []
        
        for section_name, content in generated_sections.items():
            # Extract first 200 characters as summary
            summary = content.replace("<!--", "").replace("-->", "")[:200] + "..."
            summaries.append(f"Section: {section_name}\nSummary: {summary}\n---\n")
        
        return "\n".join(summaries)
    
    def _combine_section_reports(self, generated_sections: Dict[str, str]) -> str:
        """Combine all individual section reports"""
        combined = []
        
        for section_name, content in generated_sections.items():
            title = section_name.replace("_", " ").title()
            combined.append(f"### {title}\n\n{content}\n\n---\n")
        
        return "\n".join(combined)
    
    def _create_fallback_overall_analysis(self, section_summaries: List[Dict[str, Any]], 
                                        all_risks: List[Dict[str, Any]], 
                                        all_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback overall analysis if JSON parsing fails"""
        # Simple analysis based on available data
        compliant_count = sum(1 for s in section_summaries if s.get("rating") == "compliant")
        total_sections = len(section_summaries)
        
        if compliant_count == total_sections:
            overall_status = "compliant"
        elif compliant_count >= total_sections / 2:
            overall_status = "partially_compliant"
        else:
            overall_status = "non_compliant"
        
        return {
            "overall_conformity_status": overall_status,
            "overall_score": str(int((compliant_count / total_sections) * 100)) if total_sections > 0 else "0",
            "key_insights": [
                f"Analyzed {total_sections} audit sections",
                f"{compliant_count} sections fully compliant",
                "Detailed analysis in individual sections"
            ],
            "critical_issues": [
                {
                    "issue": "Analysis completed with limited consolidation",
                    "affected_sections": [s["section"] for s in section_summaries],
                    "severity": "medium",
                    "immediate_action_required": False
                }
            ],
            "compliance_by_norm": [],
            "strengths": ["Comprehensive section analysis"],
            "improvement_areas": ["Detailed consolidation"],
            "risk_profile": {
                "high_risk_count": len([r for r in all_risks if r.get("impact") == "high"]),
                "medium_risk_count": len([r for r in all_risks if r.get("impact") == "medium"]),
                "low_risk_count": len([r for r in all_risks if r.get("impact") == "low"]),
                "top_risks": [r.get("risk_description", "Unknown risk") for r in all_risks[:3]]
            },
            "priority_actions": [
                {
                    "action": "Review individual section findings",
                    "timeline": "immediate",
                    "impact": "high",
                    "effort": "medium"
                }
            ],
            "analysis_date": datetime.now().isoformat(),
            "sections_analyzed": [s["section"] for s in section_summaries]
        }
    
    def _create_fallback_report(self, state: AgentState, generated_sections: Dict[str, str]) -> str:
        """Create a simple fallback final report when consolidation fails"""
        try:
            enterprise_report_id = state.get("enterprise_report_id", "Unknown")
            selected_norms = state.get("selected_norms", [])
            audit_plan = state.get("audit_plan", {})
            
            report = f"""# Audit Conformity Report - Enterprise Report {enterprise_report_id}

## Executive Summary
This audit conformity report was generated for enterprise report {enterprise_report_id} 
against the following conformity norms: {', '.join(selected_norms)}.

## Audit Sections Completed
Total sections analyzed: {len(generated_sections)}

"""
            
            for section_name, section_content in generated_sections.items():
                report += f"### {section_name.replace('_', ' ').title()}\n\n"
                # Take first 500 characters of each section as summary
                summary = section_content[:500] + "..." if len(section_content) > 500 else section_content
                report += f"{summary}\n\n"
            
            report += """## Overall Assessment
This report contains the detailed analysis of all audit sections. 
Please review each section for specific findings and recommendations.

## Disclaimer
This is a fallback report generated due to processing limitations. 
For a complete consolidated analysis, please regenerate the report.
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create fallback report: {str(e)}")
            return f"# Audit Report - {enterprise_report_id}\n\nReport generation encountered technical difficulties. Please contact support."
    
    def validate_state(self, state: AgentState) -> bool:
        """Validate state has required fields for consolidator"""
        if not super().validate_state(state):
            return False
        
        if "generated_sections" not in state or not state["generated_sections"]:
            logger.error("No generated sections found for consolidation")
            return False
        
        return True
