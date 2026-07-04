"""
Strictly typed deterministic models for the Transparent Reasoning Engine (TRE).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

from reasoning_engine.tre.exceptions import InvalidTraceError
from reasoning_engine.bse.models import BeliefObject
from reasoning_engine.hre.models import Hypothesis
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe.models import SituationProjection

@dataclass
class ExplanationLink:
    """Explains a single step in the reasoning chain."""
    source_ids: List[str]
    target_id: str
    explanation: str
    confidence_explanation: str
    uncertainty_explanation: str

@dataclass
class HypothesisCompetitionExplanation:
    """Explains why a hypothesis won and why others lost."""
    winning_hypothesis_id: str
    winning_reason: str
    losing_hypotheses: Dict[str, str] = field(default_factory=dict) # id -> reason lost

@dataclass
class TransparentReasoningTrace:
    """
    Complete audit trail of how CASRE arrived at a projection.
    Maps evidence backwards step-by-step.
    """
    reasoning_id: str
    
    # Pointers to the raw objects used in this trace
    observations: List[str] = field(default_factory=list) # AWM objects
    entities: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list) # ARG objects
    beliefs: List[str] = field(default_factory=list)       # BSE objects
    hypotheses: List[str] = field(default_factory=list)    # HRE objects
    world_state: str = ""                                  # WSE object
    projections: List[str] = field(default_factory=list)   # SPE objects
    
    # Chains tracking exactly WHY something happened (Task 9 enhancements)
    confidence_chain: Dict[str, str] = field(default_factory=dict) # Explains confidence propagation
    uncertainty_chain: Dict[str, str] = field(default_factory=dict) # Explains uncertainty evolution
    evidence_chain: List[ExplanationLink] = field(default_factory=list)
    contradiction_chain: Dict[str, List[str]] = field(default_factory=dict)
    conflict_resolutions: Dict[str, str] = field(default_factory=dict) # How Modality Conflicts were resolved
    hypothesis_competition: Optional[HypothesisCompetitionExplanation] = None # Explains hypothesis wins/losses
    
    # High level takeaways
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.reasoning_id:
            raise InvalidTraceError("Trace must have a reasoning_id.")
