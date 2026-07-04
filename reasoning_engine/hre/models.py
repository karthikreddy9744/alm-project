"""
Strictly typed deterministic models for the Hypothesis Reasoning Engine (HRE).
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
import time

from reasoning_engine.hre.exceptions import InvalidHypothesisUpdateError
from reasoning_engine import config
from reasoning_engine.awm.models import HierarchicalConfidence

class HypothesisStatus(Enum):
    CANDIDATE = auto()
    ACTIVE = auto()
    REJECTED = auto()

@dataclass
class Hypothesis:
    """
    Represents a deterministic hypothesis tracking a possible scenario.
    Maintains competing interpretations.
    """
    id: str
    statement: str
    supporting_beliefs: List[str] = field(default_factory=list)
    supporting_entities: List[str] = field(default_factory=list)
    supporting_events: List[str] = field(default_factory=list)
    supporting_relationships: List[str] = field(default_factory=list)
    
    contradicting_beliefs: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    
    confidence: HierarchicalConfidence = field(default_factory=HierarchicalConfidence)
    uncertainty: float = 0.5
    plausibility: float = 0.5
    
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    last_reinforced_time: float = field(default_factory=time.time)
    
    status: HypothesisStatus = HypothesisStatus.CANDIDATE

    def __post_init__(self):
        if not self.id:
            raise InvalidHypothesisUpdateError("Hypothesis must have a valid ID.")
        if not self.statement:
            raise InvalidHypothesisUpdateError("Hypothesis must have a statement.")
        if not isinstance(self.confidence, HierarchicalConfidence):
            raise InvalidHypothesisUpdateError("Confidence must be a HierarchicalConfidence object.")
            
        self.uncertainty = min(1.0, max(0.0, self.uncertainty))
        self.plausibility = min(1.0, max(0.0, self.plausibility))
        
        # Determine initial status
        self._update_status()

    def _update_status(self):
        # Using the base beliefs/hypotheses layers to judge status
        val = self.confidence.hypotheses if self.confidence.hypotheses > 0 else self.confidence.beliefs
        if val < config.HRE_REJECTION_THRESHOLD or self.plausibility < config.HRE_REJECTION_THRESHOLD:
            # We don't instantly reject, competition continues
            pass 
        elif self.status == HypothesisStatus.CANDIDATE and val >= 0.5:
            self.status = HypothesisStatus.ACTIVE

    def update_metrics(self, confidence_val: float, uncertainty: float, plausibility: float, timestamp: float = None):
        """Deterministically updates core metrics."""
        self.confidence.hypotheses = min(1.0, max(0.0, confidence_val))
        self.uncertainty = min(1.0, max(0.0, uncertainty))
        self.plausibility = min(1.0, max(0.0, plausibility))
        self.last_update = timestamp or time.time()
        self._update_status()

    def add_supporting_belief(self, belief_id: str):
        if belief_id not in self.supporting_beliefs:
            self.supporting_beliefs.append(belief_id)
            self.last_update = time.time()
            self.last_reinforced_time = time.time()

    def add_contradicting_belief(self, belief_id: str):
        if belief_id not in self.contradicting_beliefs:
            self.contradicting_beliefs.append(belief_id)
            self.last_update = time.time()
