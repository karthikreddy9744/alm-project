"""
Strictly typed deterministic models for the Situation Projection Engine (SPE).
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import time

from reasoning_engine.spe.exceptions import InvalidProjectionError
from reasoning_engine.wse.models import DecomposedConfidence

class RiskLevel(Enum):
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()

class UrgencyLevel(Enum):
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    IMMEDIATE = auto()

class StabilityLevel(Enum):
    STABLE = auto()
    DEGRADING = auto()
    VOLATILE = auto()
    IMPROVING = auto()

@dataclass
class ProjectedState:
    """Represents a specific projection outcome."""
    state_description: str
    confidence: DecomposedConfidence
    uncertainty: float = 0.5
    expected_duration: str = "Unknown"
    assumptions: List[str] = field(default_factory=list)

@dataclass
class SituationProjection:
    """
    Represents an anticipated future state deterministically projected 
    from the current world state using semantic hints without fabricating unobserved evidence.
    """
    id: str
    
    primary_projection: ProjectedState
    alternative_projection: Optional[ProjectedState] = None
    
    transition_sequence: List[str] = field(default_factory=list)
    
    urgency: str = "LOW"
    risk_level: str = "LOW"
    stability: str = "STABLE"
    
    supporting_world_state: str = "" # ID of the world state this was projected from
    
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.id:
            raise InvalidProjectionError("SituationProjection must have a valid ID.")
        if not self.primary_projection:
            raise InvalidProjectionError("SituationProjection must have a primary projection.")
