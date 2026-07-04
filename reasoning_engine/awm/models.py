"""
Strictly typed data structures and models for the Auditory World Model (AWM).
These models enforce internal consistency through __post_init__ validation.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional
import time

from reasoning_engine.awm.exceptions import StateConsistencyError

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class NodeState(Enum):
    MOVING = auto()
    STATIONARY = auto()
    DISTRESSED = auto()
    UNKNOWN = auto()

class Trajectory(Enum):
    APPROACHING = auto()
    RECEDING = auto()
    STATIC = auto()
    IMPULSIVE = auto()
    UNKNOWN = auto()

class RelationType(Enum):
    GENERATING = auto()
    REACTING = auto()
    CO_OCCURRING = auto()

# -----------------------------------------------------------------------------
# Base Objects
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintyObject:
    """Represents reasons for uncertainty (missing or conflicting evidence)."""
    reason: str
    is_conflict: bool = False

@dataclass
class HierarchicalConfidence:
    """Tracks confidence independently across all processing stages."""
    speech_recognition: float = 0.0
    translation: float = 0.0
    sound_detection: float = 0.0
    semantic_embeddings: float = 0.0
    fusion: float = 0.0
    beliefs: float = 0.0
    hypotheses: float = 0.0
    world_state: float = 0.0
    projection: float = 0.0
    
    is_uncertain: bool = False
    uncertainty_metadata: Optional[UncertaintyObject] = None

    def __post_init__(self):
        # Validate all are between 0.0 and 1.0
        for attr in ['speech_recognition', 'translation', 'sound_detection', 
                     'semantic_embeddings', 'fusion', 'beliefs', 
                     'hypotheses', 'world_state', 'projection']:
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise StateConsistencyError(f"Confidence {attr} must be between 0.0 and 1.0. Got: {val}")

    @property
    def p_val(self) -> float:
        """Helper to maintain backwards compatibility for basic ranking tests.
        Returns the highest non-zero confidence stage."""
        stages = [self.projection, self.world_state, self.hypotheses, self.beliefs, self.fusion,
                  self.semantic_embeddings, self.sound_detection, self.speech_recognition, self.translation]
        for s in stages:
            if s > 0.0:
                return s
        return 0.0

@dataclass(frozen=True)
class ReasoningPointer:
    """A direct pointer to a specific node/edge ID to fulfill interpretability requirements."""
    target_id: str

# -----------------------------------------------------------------------------
# Nodes & Edges (Internal World Representation)
# -----------------------------------------------------------------------------

@dataclass
class EntityNode:
    """Represents active actors (e.g., Human, Animal)."""
    id: str
    entity_type: str
    state: NodeState
    confidence: HierarchicalConfidence
    last_updated_timestamp: float = field(default_factory=time.time)
    persistence_count: int = 1
    
    def __post_init__(self):
        if not self.id:
            raise StateConsistencyError("EntityNode must have a valid ID.")
        if not self.entity_type:
            raise StateConsistencyError("EntityNode must have a valid entity_type.")

@dataclass
class EventNode:
    """Represents environmental sounds."""
    id: str
    class_map: str
    trajectory: Trajectory
    acoustic_salience: float
    confidence: HierarchicalConfidence
    last_updated_timestamp: float = field(default_factory=time.time)
    persistence_count: int = 1

    def __post_init__(self):
        if not self.id:
            raise StateConsistencyError("EventNode must have a valid ID.")
        if not (0.0 <= self.acoustic_salience <= 1.0):
            raise StateConsistencyError("Acoustic salience must be between 0.0 and 1.0.")

@dataclass
class RelationshipEdge:
    """Represents interactions between nodes."""
    source_id: str
    target_id: str
    relation_type: RelationType
    last_updated_timestamp: float = field(default_factory=time.time)
    persistence_count: int = 1

# -----------------------------------------------------------------------------
# Output / Inference State Objects
# -----------------------------------------------------------------------------

@dataclass
class BeliefObject:
    """Wraps an AWM Node reference with an evaluated system confidence."""
    node_id: str
    system_confidence: float
    uncertainty: Optional[UncertaintyObject] = None

@dataclass
class HypothesisObject:
    """A proposed combination of beliefs representing a potential situation."""
    hypothesis_id: str
    primitives_list: List[str]
    supporting_node_ids: List[str]
    support_score: float
    contradiction_penalty: float

    def __post_init__(self):
        if not self.hypothesis_id:
            raise StateConsistencyError("HypothesisObject must have an ID.")

    @property
    def final_score(self) -> float:
        return self.support_score - self.contradiction_penalty

@dataclass
class WorldStateObject:
    """The accepted reality at time T."""
    timestamp: float
    winning_hypothesis_id: str
    active_nodes: List[str]

@dataclass
class ProjectionObject:
    """Prediction for T+1 state."""
    timestamp: float
    expected_node_states: Dict[str, str]

@dataclass
class ReasoningTrace:
    sentence_fragment: str
    evidence_pointers: List[str]

@dataclass
class SituationReport:
    report_string: str
    traces: List[ReasoningTrace]

from enum import Enum
class SemanticIntent(Enum):
    DISTRESS = 1
    COMMAND = 2
    INQUIRY = 3
    UNKNOWN = 4
    CASUAL = 5
