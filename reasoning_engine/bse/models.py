"""
Strictly typed deterministic models for the Belief State Engine.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
import time

from reasoning_engine.bse.exceptions import InvalidBeliefUpdateError
from reasoning_engine import config

class BeliefType(Enum):
    ENTITY = auto()
    EVENT = auto()
    CONTEXT = auto()
    INTERACTION = auto()
    TEMPORAL = auto()
    ENVIRONMENTAL = auto()
    SPEECH = auto()
    COMPOSITE = auto()
    UNKNOWN = auto()

class BeliefStatus(Enum):
    ACTIVE = auto()
    WEAKENING = auto()
    INACTIVE = auto()

from reasoning_engine.awm.models import HierarchicalConfidence

@dataclass
class BeliefObject:
    """
    Represents an intermediate cognitive understanding of the environment.
    """
    id: str
    statement: str
    belief_type: BeliefType
    supporting_entities: List[str] = field(default_factory=list)
    supporting_events: List[str] = field(default_factory=list)
    supporting_relationships: List[str] = field(default_factory=list)
    confidence: HierarchicalConfidence = field(default_factory=HierarchicalConfidence)
    uncertainty: float = 0.5
    creation_timestamp: float = field(default_factory=time.time)
    update_timestamp: float = field(default_factory=time.time)
    expiration_timestamp: float = field(default_factory=lambda: time.time() + config.BSE_EXPIRATION_TIME_SECONDS)
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    status: BeliefStatus = BeliefStatus.ACTIVE
    
    # Task 5: Modality Conflict Resolution
    conflict_score: float = 0.0
    conflict_explanation: str = ""
    dominant_evidence: str = ""
    
    def __post_init__(self):
        if not self.id:
            raise InvalidBeliefUpdateError("BeliefObject must have a valid ID.")
        if not self.statement:
            raise InvalidBeliefUpdateError("BeliefObject must have a statement.")
        if not isinstance(self.confidence, HierarchicalConfidence):
            raise InvalidBeliefUpdateError("Confidence must be a HierarchicalConfidence object.")
        if not (0.0 <= self.uncertainty <= 1.0):
            raise InvalidBeliefUpdateError(f"Uncertainty must be between 0.0 and 1.0, got {self.uncertainty}")

    def add_supporting_evidence(self, evidence_id: str):
        if evidence_id not in self.supporting_evidence_ids:
            self.supporting_evidence_ids.append(evidence_id)
            self.update_timestamp = time.time()

    def add_contradicting_evidence(self, evidence_id: str):
        if evidence_id not in self.contradicting_evidence_ids:
            self.contradicting_evidence_ids.append(evidence_id)
            self.update_timestamp = time.time()
