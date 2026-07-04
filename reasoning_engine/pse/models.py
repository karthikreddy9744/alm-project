from dataclasses import dataclass, field
from typing import List, Dict, Any
from reasoning_engine.arg.models import ARGNode

@dataclass
class SegregatedStreams:
    """
    Holds the segregated foreground and background streams determined by the PSE.
    """
    foreground_nodes: List[ARGNode] = field(default_factory=list)
    background_nodes: List[ARGNode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
