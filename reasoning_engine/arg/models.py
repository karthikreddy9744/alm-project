"""
Strictly typed data structures and models for the Acoustic Relationship Graph (ARG).
"""
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any

from reasoning_engine.arg.exceptions import ARGError
from reasoning_engine.awm.models import HierarchicalConfidence

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class NodeType(Enum):
    PERSON = auto()
    SPEAKER = auto()
    VEHICLE = auto()
    ANIMAL = auto()
    MACHINE = auto()
    OBJECT = auto()
    ENVIRONMENTAL_SOURCE = auto()
    ENVIRONMENTAL_EVENT = auto()
    SPEECH_EVENT = auto()
    ACOUSTIC_EVENT = auto()
    UNKNOWN_ENTITY = auto()

class EdgeType(Enum):
    PRODUCES = auto()
    CAUSES = auto()
    APPROACHES = auto()
    MOVES_AWAY = auto()
    CO_OCCURS = auto()
    SUPPORTS = auto()
    CONTRADICTS = auto()
    CONTEXTUALISES = auto()
    BACKGROUND_TO = auto()
    INTERRUPTS = auto()
    OVERLAPS = auto()
    FOLLOWS = auto()
    PRECEDES = auto()
    INDEPENDENT_OF = auto()
    CONTAINS = auto()

# -----------------------------------------------------------------------------
# Base Models
# -----------------------------------------------------------------------------

@dataclass
class ARGNode:
    """
    Graph node representing an auditory object or event.
    """
    id: str
    node_type: NodeType
    confidence: HierarchicalConfidence = field(default_factory=HierarchicalConfidence)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_state: str = "active"

    def __post_init__(self):
        if not self.id:
            raise ARGError("ARGNode must have a valid non-empty ID.")
        if not isinstance(self.confidence, HierarchicalConfidence):
            raise ARGError("Confidence must be a HierarchicalConfidence object.")

@dataclass
class ARGEdge:
    """
    Directed graph edge representing a relationship between two nodes.
    Task 3: Relationship Enhancement included.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: HierarchicalConfidence = field(default_factory=HierarchicalConfidence)
    strength: float = 0.5
    duration: float = 0.0
    source: str = "Unknown"
    evidence_count: int = 1
    last_update: float = field(default_factory=time.time)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_id or not self.target_id:
            raise ARGError("ARGEdge must have valid source_id and target_id.")
        if not isinstance(self.confidence, HierarchicalConfidence):
            raise ARGError("Confidence must be a HierarchicalConfidence object.")
