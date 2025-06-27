# Agent Actions Summary

This document lists all available actions for each agent in the audit workflow system.

## OrchestratorAgent

The orchestrator agent manages the overall workflow using ReAct architecture and coordinates all other agents.

**Available Actions:**
- `create_audit_plan` - Creates an initial audit plan by calling the planner agent
- `start_processing` - Begins the audit processing phase after plan approval
- `start_next_section` - Initiates processing of the next planned audit section
- `query_knowledge` - Queries the knowledge base for relevant information
- `call_analyzer` - Invokes the analyzer agent for section analysis
- `call_writer` - Invokes the writer agent for report generation
- `call_consolidator` - Invokes the consolidator agent for final report generation
- `wait_human_approval` - Waits for human approval of the audit plan
- `finish_workflow` - Completes the workflow process

**Workflow Logic:**
- Prevents infinite loops by setting maximum iterations
- Pauses workflow when `awaiting_human = True` until plan approval
- Forces deterministic action selection during consolidation
- Excludes knowledge queries during critical states (CONSOLIDATING, AWAITING_APPROVAL, etc.)

## PlannerAgent

Creates detailed audit plans based on enterprise reports and selected norms.

**Available Actions:**
- `analyze_enterprise_report` - Analyzes the uploaded enterprise report
- `create_audit_plan` - Creates a structured audit plan with cycles and sections
- `validate_plan` - Validates the created audit plan for completeness
- `refine_plan` - Refines the audit plan based on feedback or requirements

## AnalyzerAgent

Analyzes specific audit sections using multimodal RAG system.

**Available Actions:**
- `query_multimodal_rag` - Queries the multimodal RAG system for relevant information
- `analyze_section_requirements` - Analyzes requirements for a specific audit section
- `assess_conformity` - Assesses conformity against selected norms
- `identify_gaps` - Identifies gaps in conformity or documentation
- `generate_findings` - Generates analysis findings for the section
- `extract_evidence` - Extracts relevant evidence from enterprise documents

## WriterAgent

Generates structured markdown report sections based on analysis results.

**Available Actions:**
- `generate_section_content` - Generates the main content for a report section
- `format_findings` - Formats analysis findings into readable report format
- `create_conformity_assessment` - Creates conformity assessment documentation
- `write_recommendations` - Writes recommendations based on analysis
- `structure_report_section` - Structures the report section with proper formatting
- `validate_content` - Validates the generated content for completeness

## ConsolidatorAgent

Consolidates all section reports into a comprehensive final audit report.

**Available Actions:**
- `consolidate_sections` - Consolidates all individual section reports
- `generate_executive_summary` - Generates an executive summary of the audit
- `create_overall_assessment` - Creates an overall conformity assessment
- `compile_recommendations` - Compiles all recommendations into a unified list
- `generate_final_report` - Generates the final comprehensive audit report
- `validate_completeness` - Validates the final report for completeness

## Workflow Status States

The workflow progresses through the following states:

1. **PENDING** - Initial state, waiting to start
2. **PLANNING** - Creating the audit plan
3. **AWAITING_APPROVAL** - Waiting for human plan approval
4. **ANALYZING** - Processing audit sections
5. **CONSOLIDATING** - Generating final report
6. **COMPLETED** - Workflow finished successfully
7. **FAILED** - Workflow failed with errors
8. **CANCELLED** - Workflow was cancelled

## Key Orchestration Features

- **Human Approval Pauses**: Workflow stops at `AWAITING_APPROVAL` until `plan_approved = True`
- **Infinite Loop Prevention**: Maximum iteration limits and early exit conditions
- **Deterministic Consolidation**: Forces consolidator action when in `CONSOLIDATING` state
- **ReAct Architecture**: Thought-Action-Observation loops for decision making
- **State Persistence**: All actions and observations are logged for audit trail
