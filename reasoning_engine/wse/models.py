"""
Strictly typed deterministic models for the World State Estimation Engine (WSE).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

from reasoning_engine.wse.exceptions import InvalidWorldStateUpdateError
from reasoning_engine.awm.models import HierarchicalConfidence

@dataclass
class WorldState:
    """
    Represents the estimated current reality based on competing hypotheses.
    Maintains causal memory.
    """
    id: str
    dominant_state: str
    secondary_states: List[str] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)
    active_events: List[str] = field(default_factory=list)
    environmental_context: str = "Unknown"
    speech_context: str = "Unknown"
    
    confidence: HierarchicalConfidence = field(default_factory=HierarchicalConfidence)
    uncertainty: float = 0.5
    ambiguity_score: float = 0.0
    consistency_score: float = 1.0
    ecological_plausibility: float = 0.5 # Task 7
    
    supporting_hypotheses: List[str] = field(default_factory=list)
    rejected_hypotheses: List[str] = field(default_factory=list)
    
    past_state_id: Optional[str] = None # Task 6: Causal Memory link
    
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.id:
            raise InvalidWorldStateUpdateError("WorldState must have a valid ID.")
        if not self.dominant_state:
            raise InvalidWorldStateUpdateError("WorldState must have a dominant state.")
        if not isinstance(self.confidence, HierarchicalConfidence):
            raise InvalidWorldStateUpdateError("Confidence must be a HierarchicalConfidence object.")
            
        self.uncertainty = min(1.0, max(0.0, self.uncertainty))
        self.ambiguity_score = min(1.0, max(0.0, self.ambiguity_score))
        self.consistency_score = min(1.0, max(0.0, self.consistency_score))
        self.ecological_plausibility = min(1.0, max(0.0, self.ecological_plausibility))
