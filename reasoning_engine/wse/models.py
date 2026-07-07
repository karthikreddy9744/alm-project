"""
Strictly typed deterministic models for the World State Estimation Engine (WSE).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

from reasoning_engine.wse.exceptions import InvalidWorldStateUpdateError
from reasoning_engine.awm.models import HierarchicalConfidence

@dataclass
class DecomposedConfidence:
    perception: float
    context: float
    situation: float
    projection: float
    overall: float

@dataclass
class ReasoningEvidence:
    name: str
    weight: float
    reason: str
    classification: str # "Observed Evidence", "Reasonable Inference", "Unknown"

@dataclass
class CognitiveState:
    """
    The ultimate language-independent structured cognitive state.
    It manages semantic meaning but never creates it.
    """
    current_situation: str = "Unknown"
    previous_situation: Optional[str] = None
    
    actors: List[str] = field(default_factory=list)
    environment: str = "Unknown"
    intentions: List[str] = field(default_factory=list)
    interaction: str = "Unknown"
    
    confidence: DecomposedConfidence = field(default_factory=lambda: DecomposedConfidence(0,0,0,0,0))
    overall_stability: float = 0.5
    uncertainty: float = 0.5
    
    projection_hint: str = "None"
    projection: str = "PENDING"
    risk: str = "Unknown"
    urgency: str = "Unknown"
    
    historical_memory: List[str] = field(default_factory=list)
    state_timeline: List[str] = field(default_factory=list)
    
    evidence_references: List[ReasoningEvidence] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)


@dataclass
class TemporalEvent:
    concept: str
    timestamp: float

@dataclass
class WorldState:
    id: str
    dominant_state: str
    past_state_id: Optional[str] = None
    cognitive_state: Optional[CognitiveState] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.id:
            raise InvalidWorldStateUpdateError("WorldState must have a valid ID.")

