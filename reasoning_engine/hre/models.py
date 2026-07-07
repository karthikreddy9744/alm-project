"""
Strictly typed deterministic models for the Hypothesis Reasoning Engine (HRE).
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import time

from reasoning_engine.hre.exceptions import InvalidHypothesisUpdateError
from reasoning_engine import config
from reasoning_engine.awm.models import HierarchicalConfidence

from reasoning_engine.semantic.models import SemanticSceneObject

class HypothesisStatus(Enum):
    CANDIDATE = auto()
    ACTIVE = auto()
    REJECTED = auto()

@dataclass
class ManagedHypothesisState:
    id: str
    situation: str # e.g. "A presentation is concluding"
    interaction_type: str 
    likely_environment: str 
    
    active_concepts: List[str] = field(default_factory=list)
    
    # Evidence reasoning (separated)
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    
    # Scores & Competition
    semantic_confidence: float = 0.0 # Provided by Semantic Engine
    temporal_consistency_score: float = 0.0 # From WSE history
    acoustic_evidence_score: float = 0.0 # From raw perception events
    
    composite_score: float = 0.0
    uncertainty: float = 0.5
    
    # Traceability
    reasoning_chain: List[str] = field(default_factory=list)
    why_it_won: str = ""
    why_alternatives_lost: str = ""
    
    # Source Reference
    source_semantic_object: Optional[SemanticSceneObject] = None
    
    creation_time: float = field(default_factory=time.time)
    
    status: HypothesisStatus = HypothesisStatus.CANDIDATE

    def __post_init__(self):
        if not self.id or not self.situation:
            raise InvalidHypothesisUpdateError("ManagedHypothesisState must have a valid ID and Situation.")

