"""
Analyzer Agent
==============

Analyzes specific audit sections using the multimodal RAG system to retrieve
relevant information from the enterprise report and assess conformity with
selected norms.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime

from .base_agent import BaseAgent
from .audit_types import AgentState, AgentType, WorkflowStatus, AuditCycle, AUDIT_CYCLE_TEMPLATES

logger = logging.getLogger(__name__)

class AnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing specific audit sections"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(AgentType.ANALYZER, llm_config)
    
    def get_available_actions(self, state: AgentState) -> List[str]:
        """List all available actions for the analyzer agent"""
        return [
            "query_multimodal_rag",
            "analyze_section_requirements", 
            "assess_conformity",
            "identify_gaps",
            "generate_findings",
            "extract_evidence"
        ]
    
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

    async def execute(self, state: AgentState) -> AgentState:
        """
        Analyze the current audit section using multimodal RAG
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with analysis results
        """
        if not self.validate_state(state):
            return self.log_error(state, "Invalid state received by analyzer")
        
        current_section = state.get("current_section")
        if not current_section:
            return self.log_error(state, "No current section specified for analysis")        
        state = self.log_message(state, f"Analyzer starting analysis of section: {current_section}")
        
        try:
            # Map section name to audit cycle enum
            cycle_enum = self._map_section_to_audit_cycle(current_section)
            if not cycle_enum:
                return self.log_error(state, f"Could not map section '{current_section}' to a valid AuditCycle")
            
            cycle_template = AUDIT_CYCLE_TEMPLATES[cycle_enum]
            
            # Retrieve relevant documents from enterprise report
            retrieved_docs = await self._retrieve_section_documents(state, cycle_template)
            
            # Analyze conformity for this section
            analysis_result = await self._analyze_conformity(state, cycle_template, retrieved_docs)
            
            # Store analysis results
            state["analysis_context"] = state.get("analysis_context", {})
            state["analysis_context"][current_section] = analysis_result
            
            # Update status to writing
            state = self.update_status(state, WorkflowStatus.WRITING)
            state = self.log_message(state, f"Analysis completed for section: {current_section}")
            
        except Exception as e:
            state = self.log_error(state, f"Failed to analyze section {current_section}: {str(e)}")
        
        return state
    
    async def _retrieve_section_documents(self, state: AgentState, cycle_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the specific audit cycle
        
        Args:
            state: Current workflow state
            cycle_template: Template for the current audit cycle
            
        Returns:
            List of relevant document chunks
        """
        enterprise_report_id = state.get("enterprise_report_id")
        
        # Create search queries based on cycle objectives and controls
        search_queries = self._generate_search_queries(cycle_template)
        
        all_documents = []
        
        # Search for each query
        for query in search_queries:
            try:
                docs = await self.retrieve_documents(
                    query=query,
                    document_id=enterprise_report_id
                )
                
                # Add query context to each document
                for doc in docs:
                    doc["search_query"] = query
                    doc["cycle"] = cycle_template["title"]
                
                all_documents.extend(docs)
                
            except Exception as e:
                logger.warning(f"[AnalyzerAgent] Failed to retrieve documents for query '{query}': {str(e)}")
        
        # Remove duplicates and limit results
        unique_docs = self._deduplicate_documents(all_documents)
        
        # Sort by relevance score if available
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique_docs[:15]  # Limit to top 15 most relevant documents
    
    def _generate_search_queries(self, cycle_template: Dict[str, Any]) -> List[str]:
        """
        Generate search queries based on audit cycle template
        
        Args:
            cycle_template: Template for the audit cycle
            
        Returns:
            List of search queries
        """
        title = cycle_template["title"]
        objectives = cycle_template["objectives"]
        key_controls = cycle_template["key_controls"]
        
        queries = []
        
        # Query based on cycle title
        queries.append(f"{title} procedures controls")
        
        # Queries based on objectives
        for objective in objectives:
            # Extract key terms from objective
            key_terms = self._extract_key_terms(objective)
            if key_terms:
                queries.append(" ".join(key_terms))
        
        # Queries based on key controls
        for control in key_controls[:3]:  # Limit to first 3 controls
            key_terms = self._extract_key_terms(control)
            if key_terms:
                queries.append(" ".join(key_terms))
        
        # Specific queries based on cycle type
        cycle_specific_queries = self._get_cycle_specific_queries(title)
        queries.extend(cycle_specific_queries)
        
        return queries[:8]  # Limit to 8 queries to avoid too many API calls
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from a text for search queries"""
        # Simple extraction - remove common words and take important terms
        common_words = {
            "vérifier", "assurer", "contrôler", "garantir", "s'assurer",
            "de", "la", "le", "les", "des", "du", "avec", "pour", "dans",
            "et", "ou", "à", "sur", "entre", "par", "que", "qui", "dont"
        }
        
        # Split and clean
        words = text.lower().replace(":", "").replace("·", "").split()
        key_terms = [word for word in words if len(word) > 3 and word not in common_words]
        
        return key_terms[:4]  # Take first 4 key terms
    
    def _get_cycle_specific_queries(self, cycle_title: str) -> List[str]:
        """Get specific search queries based on cycle type"""
        queries = []
        
        if "Ventes" in cycle_title or "Clients" in cycle_title:
            queries.extend([
                "ventes revenus chiffre affaires",
                "clients créances provisions",
                "facturation livraison commandes"
            ])
        
        elif "Achats" in cycle_title or "Fournisseurs" in cycle_title:
            queries.extend([
                "achats charges fournisseurs",
                "commandes réception factures",
                "dettes fournisseurs règlements"
            ])
        
        elif "Immobilisations" in cycle_title:
            queries.extend([
                "immobilisations actifs amortissements",
                "acquisitions cessions valorisation",
                "inventaire physique assurance"
            ])
        
        elif "Stocks" in cycle_title:
            queries.extend([
                "stocks inventaire valorisation",
                "rotation stocks provisions",
                "entrepôts surveillance accès"
            ])
        
        elif "Paie" in cycle_title or "Personnel" in cycle_title:
            queries.extend([
                "paie salaires charges sociales",
                "personnel mouvements entrées sorties",
                "heures travaillées cotisations"
            ])
        
        elif "Impôts" in cycle_title or "taxes" in cycle_title:
            queries.extend([
                "impôts taxes TVA déclarations",
                "bases imposables taux exemptions",
                "charges fiscales provisions"
            ])
        
        elif "Trésorerie" in cycle_title:
            queries.extend([
                "trésorerie banques rapprochements",
                "encaissements décaissements espèces",
                "soldes bancaires relevés"
            ])
        
        return queries
    
    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on content or document ID"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # Create a unique identifier for the document
            doc_id = doc.get("document_id", "")
            content_hash = hash(doc.get("content", "")[:100])  # Hash first 100 chars
            identifier = f"{doc_id}_{content_hash}"
            
            if identifier not in seen:
                seen.add(identifier)
                unique_docs.append(doc)
        
        return unique_docs
    
    async def _analyze_conformity(self, state: AgentState, cycle_template: Dict[str, Any], 
                                retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conformity for the audit cycle using retrieved documents
        
        Args:
            state: Current workflow state
            cycle_template: Template for the audit cycle
            retrieved_docs: Retrieved document chunks
            
        Returns:
            Analysis results
        """
        selected_norms = state.get("selected_norms", [])
        current_section = state.get("current_section")
        
        # Prepare document context
        doc_context = self._prepare_document_context(retrieved_docs)
        
        # Create analysis prompt
        analysis_prompt = f"""
        Analyze the conformity of the {cycle_template['title']} based on the enterprise audit report content.
        
        Audit Cycle Information:
        Title: {cycle_template['title']}
        Objectives: {json.dumps(cycle_template['objectives'], indent=2)}
        Key Controls: {json.dumps(cycle_template['key_controls'], indent=2)}
        Tests: {json.dumps(cycle_template['tests'], indent=2)}
        
        Selected Conformity Norms:
        {json.dumps(selected_norms, indent=2)}
        
        Retrieved Document Content:
        {doc_context}
        
        Provide a comprehensive analysis in JSON format:
        {{
            "cycle": "{current_section}",
            "analysis_date": "current_timestamp",
            "conformity_assessment": {{
                "overall_rating": "compliant/partially_compliant/non_compliant/insufficient_info",
                "compliance_score": "0-100",
                "summary": "brief summary of conformity status"
            }},
            "detailed_findings": [
                {{
                    "control_area": "specific control area",
                    "finding": "what was found",
                    "conformity_status": "compliant/non_compliant/needs_improvement",
                    "evidence": "supporting evidence from documents",
                    "impact": "potential impact of finding",
                    "recommendation": "specific recommendation"
                }}
            ],
            "gap_analysis": [
                {{
                    "gap_area": "area where gaps were identified",
                    "description": "description of the gap",
                    "severity": "high/medium/low",
                    "norm_reference": "relevant norm or standard"
                }}
            ],
            "strengths": ["identified", "strengths", "in", "controls"],
            "risks": [
                {{
                    "risk_description": "description of identified risk",
                    "likelihood": "high/medium/low",
                    "impact": "high/medium/low",
                    "mitigation": "suggested mitigation"
                }}
            ],
            "next_steps": ["recommended", "next", "steps"],
            "evidence_quality": {{
                "availability": "sufficient/limited/insufficient",
                "relevance": "high/medium/low",
                "reliability": "high/medium/low"
            }}        }}
        """
        
        analysis_response = await self.call_llm(
            analysis_prompt,
            "You are an expert auditor performing conformity analysis. Be thorough and objective. Always respond with valid JSON."
        )
        
        try:
            # Clean the response - remove markdown code blocks and extra text
            cleaned_response = analysis_response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in cleaned_response:
                # Extract JSON from markdown code block
                start_idx = cleaned_response.find("```json") + 7
                end_idx = cleaned_response.find("```", start_idx)
                if end_idx != -1:
                    cleaned_response = cleaned_response[start_idx:end_idx].strip()
            elif "```" in cleaned_response:
                # Handle generic code blocks
                start_idx = cleaned_response.find("```") + 3
                end_idx = cleaned_response.find("```", start_idx)
                if end_idx != -1:
                    cleaned_response = cleaned_response[start_idx:end_idx].strip()
            
            # Try to find JSON object if it's embedded in text
            if not cleaned_response.startswith('{'):
                json_start = cleaned_response.find('{')
                json_end = cleaned_response.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    cleaned_response = cleaned_response[json_start:json_end+1]
            
            logger.info(f"Attempting to parse cleaned JSON: {cleaned_response[:200]}...")
            analysis_result = json.loads(cleaned_response)
            
            # Add metadata
            analysis_result["analysis_date"] = datetime.now().isoformat()
            analysis_result["analyzer_agent_id"] = self.agent_id
            analysis_result["documents_analyzed"] = len(retrieved_docs)
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {str(e)}")
            logger.error(f"Original response: {analysis_response[:500]}...")
            logger.error(f"Cleaned response: {cleaned_response[:500] if 'cleaned_response' in locals() else 'N/A'}...")
            # Return a fallback analysis
            return self._create_fallback_analysis(current_section, cycle_template, retrieved_docs)
    
    def _prepare_document_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare document context for analysis"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:10]):  # Limit to 10 docs for context
            content = doc.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"""
Document {i+1}:
Source: {doc.get('source', 'Unknown')}
Search Query: {doc.get('search_query', 'Unknown')}
Content: {content}
---
""")
        
        return "\n".join(context_parts)
    
    def _create_fallback_analysis(self, current_section: str, cycle_template: Dict[str, Any], 
                                retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a fallback analysis if JSON parsing fails"""
        return {
            "cycle": current_section,
            "analysis_date": datetime.now().isoformat(),
            "conformity_assessment": {
                "overall_rating": "insufficient_info",
                "compliance_score": "0",
                "summary": "Unable to complete detailed analysis due to processing error"
            },
            "detailed_findings": [
                {
                    "control_area": "General Assessment",
                    "finding": "Analysis could not be completed due to technical issues",
                    "conformity_status": "needs_improvement",
                    "evidence": f"Reviewed {len(retrieved_docs)} documents",
                    "impact": "Unknown - requires manual review",
                    "recommendation": "Manual review of cycle controls recommended"
                }
            ],
            "gap_analysis": [
                {
                    "gap_area": "Analysis Completeness",
                    "description": "Automated analysis could not be completed",
                    "severity": "medium",
                    "norm_reference": "General audit standards"
                }
            ],
            "strengths": ["Document availability"],
            "risks": [
                {
                    "risk_description": "Incomplete analysis may miss important findings",
                    "likelihood": "medium",
                    "impact": "medium",
                    "mitigation": "Perform manual review of cycle"
                }
            ],
            "next_steps": ["Manual review required", "Technical issue resolution"],
            "evidence_quality": {
                "availability": "limited" if retrieved_docs else "insufficient",
                "relevance": "unknown",
                "reliability": "unknown"
            },
            "analyzer_agent_id": self.agent_id,
            "documents_analyzed": len(retrieved_docs),
            "error_note": "Fallback analysis due to JSON parsing error"
        }
