from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum, auto
from reasoning_engine.awm.models import EventNode, EntityNode

class ContextImportance(Enum):
    PRIMARY = auto()
    SUPPORTING = auto()
    BACKGROUND = auto()
    IGNORED = auto()

@dataclass
class SegregatedStreams:
    """
    Holds the segregated streams determined by the PSE (Selective Attention Filter).
    """
    primary_events: List[EventNode] = field(default_factory=list)
    supporting_events: List[EventNode] = field(default_factory=list)
    background_events: List[EventNode] = field(default_factory=list)
    ignored_events: List[EventNode] = field(default_factory=list)
    
    primary_entities: List[EntityNode] = field(default_factory=list)
    supporting_entities: List[EntityNode] = field(default_factory=list)
    
    ignored_reasons: Dict[str, str] = field(default_factory=dict) # Event ID -> reason ignored
    metadata: Dict[str, Any] = field(default_factory=dict)
