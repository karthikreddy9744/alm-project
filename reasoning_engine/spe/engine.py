import time
import uuid
import copy
from typing import Dict, List, Optional, Tuple

from reasoning_engine.spe.models import SituationProjection, ProjectedState
from reasoning_engine.spe.exceptions import InvalidProjectionError
from reasoning_engine import config
from reasoning_engine.wse.models import WorldState, DecomposedConfidence
from reasoning_engine.hre.models import ManagedHypothesisState

class SituationProjectionEngine:
    def __init__(self):
        self.current_projection: Optional[SituationProjection] = None
        self.history: List[SituationProjection] = []

    def initialize(self):
        self.reset()

    def reset(self):
        self.current_projection = None
        self.history.clear()

    def project(self, world_state: WorldState) -> SituationProjection:
        """
        Main projection generation method based on CognitiveState and projection_hints.
        Does NOT invent semantics. Relies purely on the managed state.
        """
        if not world_state.cognitive_state:
            return self._create_empty_proj()
            
        cognitive_state = world_state.cognitive_state
        
        # Decomposed confidence
        proj_conf = copy.deepcopy(cognitive_state.confidence)
        proj_conf.projection = max(0.0, cognitive_state.confidence.situation - 0.1)
        
        proj_uncertainty = cognitive_state.uncertainty

        # 1. Primary Projection (from Qwen's projection_hint)
        primary_state_text = cognitive_state.projection_hint
        if not primary_state_text or primary_state_text == "None":
             primary_state_text = f"Continued {cognitive_state.current_situation} until interrupted."
             
        primary_seq = [cognitive_state.current_situation, primary_state_text]
             
        primary_proj = ProjectedState(
            state_description=primary_state_text,
            confidence=proj_conf,
            uncertainty=proj_uncertainty,
            assumptions=[f"Based on semantic hint for situation: {cognitive_state.current_situation}"]
        )
        
        # 2. Risk & Urgency (Deterministic checks)
        risk = self.estimate_risk(primary_state_text, cognitive_state.actors, cognitive_state.intentions)
        urgency = self.estimate_urgency(risk, primary_proj.confidence.projection)
        stability = self.estimate_stability(proj_uncertainty, risk)
        
        # Update CognitiveState directly
        cognitive_state.projection = primary_state_text
        cognitive_state.confidence.projection = primary_proj.confidence.projection
        cognitive_state.risk = risk
        cognitive_state.urgency = urgency

        new_proj = SituationProjection(
            id=str(uuid.uuid4()),
            primary_projection=primary_proj,
            alternative_projection=None,
            transition_sequence=primary_seq,
            urgency=urgency,
            risk_level=risk,
            stability=stability,
            supporting_world_state=world_state.id
        )
        
        self.update(new_proj)
        return new_proj

    def update(self, projection: SituationProjection):
        if self.current_projection:
            self.history.append(self.current_projection)
        self.current_projection = projection
        self.synchronize()

    def synchronize(self, current_time: Optional[float] = None):
        t = current_time if current_time is not None else time.time()
        self.history = [p for p in self.history if t - p.timestamp < config.SPE_PROJECTION_EXPIRATION_SECONDS]

    def estimate_risk(self, projected_state: str, active_entities: List[str], active_concepts: List[str]) -> str:
        state_lower = projected_state.lower()
        entities_lower = [e.lower() for e in active_entities]
        concepts_lower = [c.lower() for c in active_concepts]
        
        for kw in config.SPE_HIGH_RISK_KEYWORDS:
            if kw in state_lower or any(kw in e for e in entities_lower) or any(kw in c for c in concepts_lower):
                if "emergency" in state_lower or "emergency" in concepts_lower:
                    return "CRITICAL"
                return "HIGH"
                
        for kw in config.SPE_MODERATE_RISK_KEYWORDS:
            if kw in state_lower or any(kw in e for e in entities_lower) or any(kw in c for c in concepts_lower):
                return "MODERATE"
                
        return "LOW"

    def estimate_urgency(self, risk: str, confidence: float) -> str:
        if risk == "CRITICAL":
            return "IMMEDIATE"
        if risk == "HIGH" and confidence > 0.6:
            return "HIGH"
        if risk == "MODERATE":
            return "MODERATE"
        return "LOW"

    def estimate_stability(self, ambiguity_score: float, risk: str) -> str:
        if risk in ["HIGH", "CRITICAL"]:
            if ambiguity_score > 0.5:
                return "VOLATILE"
            return "DEGRADING"
            
        if ambiguity_score > 0.6:
            return "VOLATILE"
            
        return "STABLE"

    def snapshot(self) -> dict:
        if not self.current_projection:
            return {"status": "NO_PROJECTION"}
            
        return {
            "primary_projected_state": self.current_projection.primary_projection.state_description,
            "risk_level": self.current_projection.risk_level,
            "urgency": self.current_projection.urgency,
            "stability": self.current_projection.stability,
            "confidence": getattr(self.current_projection.primary_projection.confidence, 'projection', 0.0)
        }

    def _create_empty_proj(self) -> SituationProjection:
        empty_conf = DecomposedConfidence(0,0,0,0,0)
        
        p = ProjectedState(
            state_description="Unknown",
            confidence=empty_conf,
            uncertainty=1.0,
            assumptions=["No world state available"]
        )
        
        return SituationProjection(
            id=str(uuid.uuid4()),
            primary_projection=p,
            urgency="LOW",
            risk_level="LOW",
            stability="STABLE",
            supporting_world_state=""
        )
