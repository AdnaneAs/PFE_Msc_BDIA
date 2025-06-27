"""
Agentic System Checker
=====================

- Generates and saves an image of the implemented agent workflow graph using LangGraph and IPython.
- Displays the tools (functions) available to each agent.
"""

import os
from IPython.display import display

try:
    from langgraph.graph import StateGraph
except ImportError:
    StateGraph = None

import networkx as nx
import matplotlib.pyplot as plt

from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.workflow_manager import AuditWorkflowManager
from app.agents.planner_agent import PlannerAgent
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.writer_agent import WriterAgent
from app.agents.consolidator_agent import ConsolidatorAgent

AGENT_TOOLS = {
    "OrchestratorAgent": [
        "LLM (reasoning, action selection)",
        "Workflow state updater",
        "Task delegation (calls other agents)"
    ],
    "PlannerAgent": [
        "LLM (plan description)",
        "RAG retrieval (optional)",
        "Plan template mapping"
    ],
    "AnalyzerAgent": [
        "RAG retrieval (vector DB)",
        "LLM (conformity analysis)",
        "Norms checker",
        "Evidence collector"
    ],
    "WriterAgent": [
        "LLM (markdown generation)",
        "Section context tool"
    ],
    "ConsolidatorAgent": [
        "LLM (summary, recommendations)",
        "Section aggregator",
        "Consistency checker"
    ]
}
import os
import matplotlib.pyplot as plt
import networkx as nx

from IPython.display import display, Image as IPyImage


from app.agents.workflow_manager import AuditWorkflowManager
dummy_llm_config = {
    "provider": "ollama",
    "model": "mistral:7b",
    "temperature": 0.0,
    "top_p": 0.8,
    "api_key": None
}
ipy_available = True
def save_and_display_agentic_graph_image(output_path="./agentic_workflow_graph.png"):
    """Generate, save, and (if possible) display the real agent workflow graph as an image using LangGraph and IPython."""
    manager = AuditWorkflowManager(llm_config=dummy_llm_config)
    if hasattr(manager, "graph") and manager.graph is not None:
        print("Generating agentic workflow graph image...")
        G = None
        # Try get_graph(), as_networkx(), to_networkx()
        if hasattr(manager.graph, "get_graph"):
            G = manager.graph.get_graph()
        for method in ["as_networkx", "to_networkx"]:
            if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)) and hasattr(G, method):
                G = getattr(G, method)()
        print("Type of G after conversion attempts:", type(G))
        # Manual fallback: extract nodes and edges
        if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
            if hasattr(G, "nodes") and hasattr(G, "edges"):
                nxG = nx.DiGraph()
                try:
                    # Add nodes
                    nxG.add_nodes_from(list(G.nodes))
                    # Add edges (extract source and target)
                    for edge in G.edges:
                        # Try to extract source and target attributes
                        if hasattr(edge, "source") and hasattr(edge, "target"):
                            nxG.add_edge(edge.source, edge.target)
                        elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                            nxG.add_edge(edge[0], edge[1])
                    G = nxG
                    print("Manually constructed NetworkX graph from nodes/edges.")
                except Exception as e:
                    print("Manual conversion to NetworkX failed:", e)
        if isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
            plt.figure(figsize=(10, 7))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2200, font_size=10, font_weight='bold', arrowsize=20)
            plt.title("Actual Agentic Audit Workflow Graph")
            plt.tight_layout()
            plt.savefig(output_path)
            print(f"Actual agentic workflow graph saved as {output_path}")
            if ipy_available:
                display(IPyImage(filename=output_path))
            plt.close()
        else:
            print("Could not convert workflow graph to a NetworkX graph object.")
    else:
        print("No compiled LangGraph workflow found in AuditWorkflowManager.")

def display_agent_tools():
    """Display the tools available to each agent."""
    print("\nAgent Tools:")
    for agent, tools in AGENT_TOOLS.items():
        print(f"\n- {agent}:")
        for tool in tools:
            print(f"    - {tool}")

if __name__ == "__main__":
    save_and_display_agentic_graph_image()
    display_agent_tools()
