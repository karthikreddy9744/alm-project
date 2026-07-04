"""
World State Estimation Engine (WSE) Core.
Estimates the current real-world situation from competing hypotheses.
Supports Causal Memory (history sequences) and Ecological Plausibility estimation.
"""
import time
import uuid
import copy
from typing import Dict, List, Optional

from reasoning_engine.wse.models import WorldState
from reasoning_engine.wse.exceptions import InvalidWorldStateUpdateError
from reasoning_engine import config
from reasoning_engine.hre.models import Hypothesis, HypothesisStatus
from reasoning_engine.awm.models import HierarchicalConfidence

class WorldStateEngine:
    def __init__(self):
        self.current_state: Optional[WorldState] = None
        # Task 6: Causal Memory
        self.history: List[WorldState] = []

    def initialize(self):
        """Initializes or resets the WSE."""
        self.reset()

    def reset(self):
        """Clears all states."""
        self.current_state = None
        self.history.clear()

    def estimate(self, ranked_hypotheses: List[Hypothesis], active_beliefs: list, awm: dict, arg: dict) -> WorldState:
        """
        Estimates the current world state based on HRE output and existing data.
        ranked_hypotheses is assumed to be sorted by score descending.
        """
        if not ranked_hypotheses:
            return self._create_empty_state()

        top_hyp = ranked_hypotheses[0]
        secondary_hyps = ranked_hypotheses[1:]

        # Calculate ambiguity
        ambiguity = self.compute_ambiguity(top_hyp, secondary_hyps)
        
        # Calculate uncertainty based on top hypothesis uncertainty and ambiguity
        uncertainty = self.compute_uncertainty(top_hyp.uncertainty, ambiguity)
        
        # Calculate consistency
        consistency = self.compute_consistency(top_hyp, active_beliefs)
        
        # Task 7: Ecological Plausibility
        eco_plausibility = self.compute_ecological_plausibility(top_hyp, arg)

        # Transition Smoothing
        dominant_state = top_hyp.statement
        new_conf = copy.deepcopy(top_hyp.confidence)
        
        if self.current_state and self.current_state.dominant_state != dominant_state:
            # Check if top hypothesis beats current dominant state momentum
            current_conf = self.current_state.confidence.world_state
            if (top_hyp.confidence.hypotheses + eco_plausibility)/2.0 < current_conf + config.WSE_STATE_TRANSITION_MOMENTUM:
                # Retain current dominant state, smooth transition
                dominant_state = self.current_state.dominant_state
                # Current state loses some confidence due to conflicting top_hyp
                new_conf.world_state = max(0.0, current_conf - 0.1)
            else:
                # State transition successful
                new_conf.world_state = (top_hyp.confidence.hypotheses + eco_plausibility) / 2.0
        else:
            new_conf.world_state = (top_hyp.confidence.hypotheses + eco_plausibility) / 2.0

        # Construct new state
        new_state = WorldState(
            id=str(uuid.uuid4()),
            dominant_state=dominant_state,
            secondary_states=[h.statement for h in secondary_hyps if h.confidence.hypotheses > 0.2],
            active_entities=top_hyp.supporting_entities.copy(),
            active_events=top_hyp.supporting_events.copy(),
            confidence=new_conf,
            uncertainty=uncertainty,
            ambiguity_score=ambiguity,
            consistency_score=consistency,
            ecological_plausibility=eco_plausibility,
            supporting_hypotheses=[top_hyp.id],
            rejected_hypotheses=[h.id for h in secondary_hyps if h.status == HypothesisStatus.REJECTED],
            past_state_id=self.current_state.id if self.current_state else None
        )

        self.update(new_state)
        return new_state

    def update(self, world_state: WorldState):
        """Updates the current state and maintains history."""
        if self.current_state:
            self.history.append(self.current_state)
        self.current_state = world_state
        self.synchronize()

    def synchronize(self, current_time: Optional[float] = None):
        """Maintains history length and temporal bounds."""
        t = current_time if current_time is not None else time.time()
        # Clean up old history
        self.history = [s for s in self.history if t - s.timestamp < config.WSE_STATE_EXPIRATION_SECONDS]

    def compute_ambiguity(self, top_hyp: Hypothesis, secondary_hyps: List[Hypothesis]) -> float:
        """
        Determines how ambiguous the situation is based on the distance 
        between the top hypothesis score and the runner-up.
        """
        if not secondary_hyps:
            return 0.0
            
        runner_up = secondary_hyps[0]
        score_diff = top_hyp.confidence.hypotheses - runner_up.confidence.hypotheses
        
        # If difference is small, ambiguity is high. Max ambiguity is 1.0.
        if score_diff >= config.WSE_AMBIGUITY_THRESHOLD:
            return 0.0
        
        ambiguity = 1.0 - (score_diff / config.WSE_AMBIGUITY_THRESHOLD)
        return min(1.0, max(0.0, ambiguity))

    def compute_uncertainty(self, base_uncertainty: float, ambiguity: float) -> float:
        """Combines base uncertainty with situational ambiguity."""
        # Simple deterministic combination
        return min(1.0, max(0.0, base_uncertainty * 0.5 + ambiguity * 0.5))

    def compute_consistency(self, hypothesis: Hypothesis, active_beliefs: list) -> float:
        """Evaluates how consistent the hypothesis is with currently active beliefs."""
        if not hypothesis.supporting_beliefs and not active_beliefs:
            return 1.0
        if not hypothesis.supporting_beliefs:
            return 0.5
            
        # Simplified deterministic consistency: 
        # ratio of hypothesis supporting beliefs to total active beliefs relevant
        # or checking against contradictory beliefs.
        conflict_penalty = min(1.0, len(hypothesis.contradicting_beliefs) * 0.1)
        consistency = 1.0 - conflict_penalty
        return min(1.0, max(0.0, consistency))
        
    def compute_ecological_plausibility(self, top_hyp: Hypothesis, arg: dict) -> float:
        """
        Task 7: Ecological Plausibility.
        Evaluates deterministic plausibility of the entities and events occurring together.
        E.g. Dog Bark + Thunder + Rain = High Plausibility.
        Dog Bark + Helicopter + Church Bell = Moderate Plausibility.
        """
        if not top_hyp.supporting_entities and not top_hyp.supporting_events:
            return 0.5 # Neutral
            
        # Mock calculation: count number of co-occurring nodes in ARG for semantic coherence
        # A more advanced implementation would map semantic embeddings, but we are keeping it deterministic.
        coherence_score = 0.5
        items = len(top_hyp.supporting_entities) + len(top_hyp.supporting_events)
        if items >= 3:
            # Assume multiple supporting elements implies rich context (High plausibility)
            coherence_score = 0.9
        elif items == 2:
            coherence_score = 0.7
            
        # Adjust based on prior history context to simulate Causal Plausibility
        if self.current_state and top_hyp.statement == self.current_state.dominant_state:
            coherence_score += 0.1
            
        return min(1.0, max(0.0, coherence_score))

    def snapshot(self) -> dict:
        """Returns a serialized snapshot of the current state."""
        if not self.current_state:
            return {"status": "NO_STATE"}
        return {
            "dominant_state": self.current_state.dominant_state,
            "confidence": self.current_state.confidence.p_val,
            "ambiguity": self.current_state.ambiguity_score,
            "eco_plaus": self.current_state.ecological_plausibility,
            "timestamp": self.current_state.timestamp,
            "past_state_id": self.current_state.past_state_id
        }

    def _create_empty_state(self) -> WorldState:
        empty_conf = HierarchicalConfidence()
        empty_conf.world_state = 0.0
        return WorldState(
            id=str(uuid.uuid4()),
            dominant_state="Unknown situation",
            confidence=empty_conf,
            uncertainty=1.0,
            ambiguity_score=0.0,
            consistency_score=0.0,
            ecological_plausibility=0.0
        )
