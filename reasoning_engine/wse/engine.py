import time
import uuid
import copy
from typing import Dict, List, Optional

from reasoning_engine.wse.models import WorldState, CognitiveState, ReasoningEvidence, DecomposedConfidence, TemporalEvent
from reasoning_engine.wse.exceptions import InvalidWorldStateUpdateError
from reasoning_engine import config
from reasoning_engine.hre.models import ManagedHypothesisState, HypothesisStatus
from reasoning_engine.pse.models import SegregatedStreams

class WorldStateEngine:
    def __init__(self):
        self.current_state: Optional[WorldState] = None
        self.history: List[WorldState] = []
        self.temporal_history: List[TemporalEvent] = []

    def initialize(self):
        self.reset()

    def reset(self):
        self.current_state = None
        self.history.clear()
        self.temporal_history.clear()

    def estimate(self, ranked_hypotheses: List[ManagedHypothesisState], streams: SegregatedStreams) -> WorldState:
        """
        Builds the CognitiveState strictly from structured evidence (NO ENGLISH inference).
        Tracks Temporal Transitions.
        """
        if not ranked_hypotheses:
            return self._create_empty_state()

        top_hyp = ranked_hypotheses[0]
        semantic_json = top_hyp.source_semantic_object
        
        # 1. Evidence Collection
        evidence_list = []
        
        # Observed Evidence (Raw audio events)
        for e in streams.primary_events:
            evidence_list.append(ReasoningEvidence(e.class_map, top_hyp.acoustic_evidence_score, "Raw acoustic perception", "Observed Evidence"))
        for e in streams.supporting_events:
            evidence_list.append(ReasoningEvidence(e.class_map, top_hyp.acoustic_evidence_score * 0.5, "Background acoustic perception", "Observed Evidence"))
            
        # Reasonable Inference (Semantic extraction)
        evidence_list.append(ReasoningEvidence(top_hyp.situation, top_hyp.semantic_confidence, "Semantic extraction", "Reasonable Inference"))
        evidence_list.append(ReasoningEvidence(top_hyp.likely_environment, top_hyp.semantic_confidence, "Semantic environment inference", "Reasonable Inference"))
        
        # 2. Decomposed Confidence
        decomp_conf = DecomposedConfidence(
            perception=top_hyp.acoustic_evidence_score,
            context=top_hyp.temporal_consistency_score,
            situation=top_hyp.semantic_confidence,
            projection=0.0, # Filled by SPE
            overall=top_hyp.composite_score
        )

        # 3. Transition Detection
        prev_situation = self.current_state.dominant_state if self.current_state else None
        historical_memory = [s.dominant_state for s in self.history[-5:]] if self.history else []
        timeline = [f"[{s.timestamp:.1f}] {s.dominant_state}" for s in self.history[-5:]] if self.history else []
        
        # Transition recording
        if prev_situation and prev_situation != top_hyp.situation:
            self.temporal_history.append(TemporalEvent(f"Transition: {prev_situation} -> {top_hyp.situation}", time.time()))
        
        reasoning_trace = top_hyp.reasoning_chain.copy()
        reasoning_trace.append(f"WSE selected '{top_hyp.situation}' as dominant situation.")
        
        missing = top_hyp.missing_evidence if top_hyp.missing_evidence else []
        
        cognitive_state = CognitiveState(
            current_situation=top_hyp.situation,
            previous_situation=prev_situation,
            actors=semantic_json.actors if semantic_json else [],
            environment=top_hyp.likely_environment,
            intentions=getattr(semantic_json, 'human_goals', []) if semantic_json else [],
            interaction=top_hyp.interaction_type,
            confidence=decomp_conf,
            overall_stability=top_hyp.temporal_consistency_score,
            uncertainty=top_hyp.uncertainty,
            projection_hint=getattr(semantic_json, 'projection', "None") if semantic_json else "None",
            projection="PENDING",
            risk="Unknown",
            urgency="Unknown",
            historical_memory=historical_memory,
            state_timeline=timeline,
            evidence_references=evidence_list,
            missing_evidence=missing,
            reasoning_trace=reasoning_trace
        )

        new_state = WorldState(
            id=str(uuid.uuid4()),
            dominant_state=top_hyp.situation,
            past_state_id=self.current_state.id if self.current_state else None,
            cognitive_state=cognitive_state
        )

        self.update(new_state)
        return new_state

    def update(self, world_state: WorldState):
        if self.current_state:
            self.history.append(self.current_state)
        self.current_state = world_state
        self.synchronize()

    def synchronize(self, current_time: Optional[float] = None):
        t = current_time if current_time is not None else time.time()
        self.history = [s for s in self.history if t - s.timestamp < config.WSE_STATE_EXPIRATION_SECONDS]

    def snapshot(self) -> dict:
        if not self.current_state:
            return {"status": "NO_STATE"}
        return {
            "dominant_state": self.current_state.dominant_state,
            "confidence": self.current_state.cognitive_state.confidence.overall if self.current_state.cognitive_state else 0.0,
            "timestamp": self.current_state.timestamp,
            "past_state_id": self.current_state.past_state_id
        }

    def _create_empty_state(self) -> WorldState:
        return WorldState(
            id=str(uuid.uuid4()),
            dominant_state="Unknown_Situation"
        )
