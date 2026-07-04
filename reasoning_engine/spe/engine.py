"""
Situation Projection Engine (SPE) Core.
Anticipates future events deterministically based on the current World State.
Enhanced with Alternative Projections, Low Probability Projections, and History tracking.
"""
import time
import uuid
import re
import copy
from typing import Dict, List, Optional, Tuple

from reasoning_engine.spe.models import SituationProjection, ProjectedState, RiskLevel, UrgencyLevel, StabilityLevel
from reasoning_engine.spe.exceptions import InvalidProjectionError
from reasoning_engine import config
from reasoning_engine.wse.models import WorldState
from reasoning_engine.awm.models import HierarchicalConfidence

class SituationProjectionEngine:
    def __init__(self):
        self.current_projection: Optional[SituationProjection] = None
        self.history: List[SituationProjection] = []

    def initialize(self):
        self.reset()

    def reset(self):
        self.current_projection = None
        self.history.clear()

    def project(self, world_state: WorldState, active_hypotheses: list, active_beliefs: list) -> SituationProjection:
        """
        Main projection generation method. Computes primary, alternative, and low probability projections.
        """
        # Base confidence propagates from WSE to SPE layer
        proj_conf = copy.deepcopy(world_state.confidence)
        proj_conf.projection = max(0.0, world_state.confidence.world_state - 0.1) 
        
        # Uncertainty inherently increases when projecting
        proj_uncertainty = min(1.0, world_state.uncertainty + 0.1)

        # 1. Primary Projection
        primary_state_text, primary_seq = self.estimate_transition(world_state.dominant_state)
        primary_proj = ProjectedState(
            state_description=primary_state_text,
            confidence=proj_conf,
            uncertainty=proj_uncertainty,
            assumptions=["Based on dominant world state"]
        )
        
        # 2. Alternative Projection
        alt_proj = None
        alt_seq = []
        if world_state.secondary_states and world_state.ambiguity_score > 0.3:
            alt_state_text, alt_seq = self.estimate_transition(world_state.secondary_states[0])
            
            alt_conf = copy.deepcopy(proj_conf)
            alt_conf.projection = max(0.0, alt_conf.projection - 0.2)
            
            alt_proj = ProjectedState(
                state_description=alt_state_text,
                confidence=alt_conf,
                uncertainty=min(1.0, proj_uncertainty + 0.2),
                assumptions=["Based on highest-ranking secondary state"]
            )
            
        # 3. Low Probability Projection
        low_prob_proj = None
        if len(world_state.secondary_states) > 1:
            low_state_text, _ = self.estimate_transition(world_state.secondary_states[-1])
            
            low_conf = copy.deepcopy(proj_conf)
            low_conf.projection = max(0.0, low_conf.projection - 0.5)
            
            low_prob_proj = ProjectedState(
                state_description=low_state_text,
                confidence=low_conf,
                uncertainty=min(1.0, proj_uncertainty + 0.4),
                assumptions=["Based on lowest-ranking viable hypothesis"]
            )

        # Estimate Risk, Urgency, Stability based on Primary
        risk = self.estimate_risk(primary_state_text, world_state.active_entities)
        urgency = self.estimate_urgency(risk, primary_proj.confidence.projection)
        stability = self.estimate_stability(world_state.ambiguity_score, risk)

        # Construct full projection
        new_proj = SituationProjection(
            id=str(uuid.uuid4()),
            primary_projection=primary_proj,
            alternative_projection=alt_proj,
            low_probability_projection=low_prob_proj,
            transition_sequence=primary_seq,
            urgency=urgency,
            risk_level=risk,
            stability=stability,
            supporting_world_state=world_state.id
        )
        
        self.update(new_proj)
        return new_proj

    def update(self, projection: SituationProjection):
        """Updates the current projection and stores history."""
        if self.current_projection:
            self.history.append(self.current_projection)
        self.current_projection = projection
        self.synchronize()

    def synchronize(self, current_time: Optional[float] = None):
        """Garbage collects old projections."""
        t = current_time if current_time is not None else time.time()
        self.history = [p for p in self.history if t - p.timestamp < config.SPE_PROJECTION_EXPIRATION_SECONDS]

    def estimate_transition(self, current_state_text: str) -> Tuple[str, List[str]]:
        """
        Deterministically maps a current state to a projected state based on 
        pre-defined configuration rules, avoiding hallucination.
        """
        for pattern, next_state in config.SPE_TRANSITION_RULES:
            if re.search(pattern, current_state_text):
                return next_state, [current_state_text, next_state]
        
        return f"Continued {current_state_text}", [current_state_text]

    def estimate_risk(self, projected_state: str, active_entities: List[str]) -> RiskLevel:
        """Deterministically evaluates risk level based on state and entities."""
        state_lower = projected_state.lower()
        entities_lower = [e.lower() for e in active_entities]
        
        # Check CRITICAL / HIGH
        for kw in config.SPE_HIGH_RISK_KEYWORDS:
            if kw in state_lower or any(kw in e for e in entities_lower):
                if "emergency" in state_lower:
                    return RiskLevel.CRITICAL
                return RiskLevel.HIGH
                
        # Check MODERATE
        for kw in config.SPE_MODERATE_RISK_KEYWORDS:
            if kw in state_lower or any(kw in e for e in entities_lower):
                return RiskLevel.MODERATE
                
        return RiskLevel.LOW

    def estimate_urgency(self, risk: RiskLevel, confidence: float) -> UrgencyLevel:
        """Maps Risk and Confidence to Urgency."""
        if risk == RiskLevel.CRITICAL:
            return UrgencyLevel.IMMEDIATE
        if risk == RiskLevel.HIGH and confidence > 0.6:
            return UrgencyLevel.HIGH
        if risk == RiskLevel.MODERATE:
            return UrgencyLevel.MODERATE
        return UrgencyLevel.LOW

    def estimate_stability(self, ambiguity_score: float, risk: RiskLevel) -> StabilityLevel:
        """Determines stability based on ambiguity and risk trends."""
        if risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if ambiguity_score > 0.5:
                return StabilityLevel.VOLATILE
            return StabilityLevel.DEGRADING
            
        if ambiguity_score > 0.6:
            return StabilityLevel.VOLATILE
            
        return StabilityLevel.STABLE

    def snapshot(self) -> dict:
        if not self.current_projection:
            return {"status": "NO_PROJECTION"}
            
        return {
            "primary_projected_state": self.current_projection.primary_projection.state_description,
            "risk_level": self.current_projection.risk_level.name,
            "urgency": self.current_projection.urgency.name,
            "stability": self.current_projection.stability.name,
            "confidence": self.current_projection.primary_projection.confidence.p_val
        }
