"""
Agentic Audit Report Generation System
=====================================

This module implements a multi-agent system using LangGraph for generating
audit conformity reports. The system uses a ReAct architecture with multiple
specialized agents working together with shared memory/state.

Agents:
- Orchestrator: Manages the overall workflow using ReAct architecture
- Planner: Creates audit plans based on uploaded reports and selected norms
- Analyzer: Analyzes specific audit sections using multimodal RAG
- Writer: Generates markdown report sections
- Consolidator: Combines sections into final report

The system integrates with the existing multimodal RAG system for document
retrieval and analysis.
"""

from typing import Dict, Any, List, Optional, TypedDict
from enum import Enum
import json
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents in the system"""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    ANALYZER = "analyzer"
    WRITER = "writer"
    CONSOLIDATOR = "consolidator"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    ANALYZING = "analyzing"
    WRITING = "writing"
    CONSOLIDATING = "consolidating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AuditCycle(Enum):
    """Audit cycles from the template"""
    VENTES_CLIENTS = "ventes_clients"
    ACHATS_FOURNISSEURS = "achats_fournisseurs"
    IMMOBILISATIONS = "immobilisations"
    STOCKS = "stocks"
    PAIE_PERSONNEL = "paie_personnel"
    IMPOTS_TAXES = "impots_taxes"
    TRESORERIE = "tresorerie"

@dataclass
class AuditSection:
    """Represents a section of the audit report"""
    cycle: AuditCycle
    title: str
    objectives: List[str]
    key_controls: List[str]
    tests: List[str]
    status: str = "pending"
    analysis_result: Optional[str] = None
    written_content: Optional[str] = None
    
class AgentState(TypedDict):
    """Shared state between all agents using LangGraph"""
    # Workflow metadata
    workflow_id: str
    status: WorkflowStatus
    created_at: str
    updated_at: str
    
    # Input data
    enterprise_report_id: str
    selected_norms: List[str]
    user_id: str
    
    # Planning phase
    audit_plan: Optional[Dict[str, Any]]
    plan_approved: bool
    approval_feedback: Optional[str]
    
    # Processing data
    current_section: Optional[AuditCycle]
    audit_sections: List[Dict[str, Any]]  # Serialized AuditSection objects
    
    # RAG context
    retrieved_documents: List[Dict[str, Any]]
    analysis_context: Dict[str, Any]
    
    # Output
    generated_sections: Dict[str, str]  # cycle -> markdown content
    final_report: Optional[str]
    
    # Agent communication
    messages: List[Dict[str, Any]]
    errors: List[str]
    
    # Human in the loop
    human_feedback: Optional[str]
    awaiting_human: bool

# Audit cycle templates with detailed information
AUDIT_CYCLE_TEMPLATES = {
    AuditCycle.VENTES_CLIENTS: {
        "title": "Cycle Ventes/Clients",
        "objectives": [
            "Vérifier la comptabilisation exacte des produits",
            "Assurer l'adéquation des principes comptables avec les normes en vigueur",
            "Contrôler les créances clients et les provisions pour dépréciation"
        ],
        "key_controls": [
            "Commande : Étude de solvabilité des clients, numérotation séquentielle des bons de commande",
            "Livraison : Vérification des stocks, gestion des livraisons partielles, bordereaux signés par les clients",
            "Facturation : Rapprochement entre livraisons et factures, contrôle des tarifs et de la TVA",
            "Enregistrement : Séparation entre systèmes manuel et automatisé, vérification des chevauchements de périodes",
            "Encaissements : Rapprochements bancaires réguliers, analyse des comptes clients",
            "Séparation des tâches : Distinction entre commande, livraison, facturation et encaissement"
        ],
        "tests": [
            "Analytiques : Comparaison des ratios clients/dettes, analyse des marges par produit",
            "Détail : Circularisation des clients, vérification des factures et des retours, tests de cut-off"
        ]
    },
    AuditCycle.ACHATS_FOURNISSEURS: {
        "title": "Cycle Achats/Fournisseurs",
        "objectives": [
            "Garantir l'exactitude des charges comptabilisées",
            "Contrôler les dettes fournisseurs et les avoirs"
        ],
        "key_controls": [
            "Commandes : Autorisation préalable, appels d'offres, suivi chronologique",
            "Réception : Vérification des bons de commande/réception, rapprochement avec les budgets",
            "Factures : Rapprochement avec les bons de commande, contrôle arithmétique",
            "Règlements : Validation des paiements, double signature pour les montants élevés",
            "Séparation des tâches : Distinction entre commande, réception, comptabilisation et paiement"
        ],
        "tests": [
            "Analytiques : Comparaison des délais de crédit fournisseur avec le secteur",
            "Détail : Circularisation des fournisseurs, vérification des factures non parvenues (FNP)"
        ]
    },
    AuditCycle.IMMOBILISATIONS: {
        "title": "Cycle Immobilisations",
        "objectives": [
            "Vérifier l'existence, la propriété et la valorisation des actifs",
            "Contrôler les amortissements et les cessions"
        ],
        "key_controls": [
            "Acquisition : Budget annuel approuvé, bons de commande/réception numérotés",
            "Amortissement : Cohérence des méthodes, dates de mise en service",
            "Protection : Assurance, titres de propriété, inventaire physique périodique",
            "Engagements hors bilan : Suivi des crédits-bail et hypothèques"
        ],
        "tests": [
            "Analytiques : Analyse des mouvements d'immobilisations et des dotations aux amortissements",
            "Détail : Vérification des titres de propriété, circularisation des organismes (ex : cadastre)"
        ]
    },
    AuditCycle.STOCKS: {
        "title": "Cycle Stocks",
        "objectives": [
            "S'assurer de l'existence, de la valorisation et de la rotation des stocks",
            "Contrôler les provisions pour dépréciation"
        ],
        "key_controls": [
            "Mouvements : Réconciliations entre inventaire physique et permanent",
            "Valorisation : Méthode FIFO/CMUP, analyse des écarts de coûts",
            "Protection : Surveillance des entrepôts, accès restreint"
        ],
        "tests": [
            "Analytiques : Comparaison des stocks avec les budgets et les ratios de rotation",
            "Détail : Assistance à l'inventaire physique, vérification des coûts de revient"
        ]
    },
    AuditCycle.PAIE_PERSONNEL: {
        "title": "Cycle Paie-Personnel",
        "objectives": [
            "Vérifier l'exactitude des salaires, charges sociales et provisions (congés payés, 13e mois)"
        ],
        "key_controls": [
            "Mouvements de personnel : Dossiers individuels, registre des entrées/sorties",
            "Paie : Vérification des heures travaillées, mise à jour des taux de cotisations",
            "Séparation des tâches : Distinction entre préparation, paiement et comptabilisation"
        ],
        "tests": [
            "Analytiques : Comparaison des charges sociales avec les budgets",
            "Détail : Vérification des bulletins de paie, circularisation des organismes sociaux"
        ]
    },
    AuditCycle.IMPOTS_TAXES: {
        "title": "Cycle Impôts et taxes",
        "objectives": [
            "S'assurer du calcul correct des impôts (TVA, taxes sur salaires) et des provisions"
        ],
        "key_controls": [
            "Déclarations : Rapprochement CA déclaré/comptabilisé, revue annuelle des comptes",
            "TVA : Vérification des taux, des exemptions et des règles de déduction"
        ],
        "tests": [
            "Détail : Contrôle des déclarations fiscales, analyse des bases imposables"
        ]
    },
    AuditCycle.TRESORERIE: {
        "title": "Cycle Trésorerie",
        "objectives": [
            "Garantir la fiabilité des soldes bancaires et la gestion des flux"
        ],
        "key_controls": [
            "Rapprochements bancaires : Mensuels, supervision par la direction",
            "Séparation des tâches : Distinction entre gestion des encaissements/décaissements",
            "Contrôle d'accès : Limitation des accès aux comptes et chéquiers"
        ],
        "tests": [
            "Détail : Circularisation des banques, inventaire des espèces en caisse, vérification des relevés"
        ]
    }
}
