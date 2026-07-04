"""
Belief State Engine (BSE) Core.
Deterministically maintains, merges, updates, and expires beliefs based on incoming structured observations.
Supports Temporal Memory and Modality Conflict Resolution.
"""
from typing import Dict, List, Optional
import time

from reasoning_engine.bse.models import BeliefObject, BeliefType, BeliefStatus
from reasoning_engine.bse.exceptions import BeliefLimitExceededError, InvalidBeliefUpdateError
from reasoning_engine import config
from reasoning_engine.awm.models import HierarchicalConfidence

class BeliefStateEngine:
    def __init__(self):
        self.beliefs: Dict[str, BeliefObject] = {}
        # Task 2: Temporal Memory - keep rolling history of beliefs
        self.belief_history: List[Dict[str, dict]] = []

    def initialize(self):
        """Initializes or resets the Belief State Engine."""
        self.reset()

    def reset(self):
        """Clears all beliefs and resets internal state."""
        self.beliefs.clear()
        self.belief_history.clear()

    def create_belief(self, 
                      belief_id: str, 
                      statement: str, 
                      belief_type: BeliefType, 
                      confidence: HierarchicalConfidence = None,
                      uncertainty: float = 0.5,
                      supporting_entities: List[str] = None,
                      supporting_events: List[str] = None,
                      supporting_relationships: List[str] = None,
                      supporting_evidence_ids: List[str] = None) -> BeliefObject:
        """Creates and stores a new belief."""
        if len(self.beliefs) >= config.BSE_MAX_ACTIVE_BELIEFS:
            self._garbage_collect()
            if len(self.beliefs) >= config.BSE_MAX_ACTIVE_BELIEFS:
                raise BeliefLimitExceededError("Maximum belief limit exceeded after garbage collection.")

        if belief_id in self.beliefs:
            return self.update_belief(belief_id, confidence_delta=config.BSE_REINFORCEMENT_BOOST)

        new_belief = BeliefObject(
            id=belief_id,
            statement=statement,
            belief_type=belief_type,
            confidence=confidence or HierarchicalConfidence(),
            uncertainty=min(1.0, max(0.0, uncertainty)),
            supporting_entities=supporting_entities or [],
            supporting_events=supporting_events or [],
            supporting_relationships=supporting_relationships or [],
            supporting_evidence_ids=supporting_evidence_ids or []
        )
        self.beliefs[belief_id] = new_belief
        return new_belief

    def update_belief(self, belief_id: str, confidence_delta: float, current_time: Optional[float] = None) -> BeliefObject:
        """Deterministically updates confidence for an existing belief."""
        if belief_id not in self.beliefs:
            raise InvalidBeliefUpdateError(f"Belief {belief_id} not found.")

        t = current_time if current_time is not None else time.time()
        belief = self.beliefs[belief_id]
        
        # Apply deterministic delta to the beliefs hierarchical layer
        new_confidence = belief.confidence.beliefs + confidence_delta
        belief.confidence.beliefs = min(1.0, max(0.0, new_confidence))
        
        belief.update_timestamp = t
        belief.expiration_timestamp = t + config.BSE_EXPIRATION_TIME_SECONDS

        if belief.confidence.beliefs < config.BSE_MIN_CONFIDENCE_THRESHOLD:
            belief.status = BeliefStatus.INACTIVE
        else:
            belief.status = BeliefStatus.ACTIVE
            
        return belief

    def resolve_modality_conflicts(self):
        """
        Task 5: Modality Conflict Resolution.
        Detects conflicts between speech, environment, and temporal history.
        Estimates reliability, assigns conflict scores and explanations.
        """
        for belief in self.beliefs.values():
            if len(belief.contradicting_evidence_ids) > 0:
                # We have a conflict. Compare speech vs environment vs history
                # Simple deterministic rule: 
                # If speech confidence > environment confidence, speech dominates.
                # Else environment dominates.
                speech_conf = belief.confidence.speech_recognition + belief.confidence.translation
                env_conf = belief.confidence.sound_detection + belief.confidence.fusion
                
                conflict_magnitude = min(1.0, len(belief.contradicting_evidence_ids) * 0.1)
                belief.conflict_score = conflict_magnitude
                
                if speech_conf > env_conf:
                    belief.dominant_evidence = "Speech Modality"
                    belief.conflict_explanation = "Speech evidence outweighs environmental evidence."
                    # Boost speech layer confidence slightly
                    belief.confidence.speech_recognition = min(1.0, belief.confidence.speech_recognition + 0.05)
                elif env_conf > speech_conf:
                    belief.dominant_evidence = "Environmental Modality"
                    belief.conflict_explanation = "Environmental audio evidence outweighs speech evidence."
                    belief.confidence.sound_detection = min(1.0, belief.confidence.sound_detection + 0.05)
                else:
                    belief.dominant_evidence = "Temporal History"
                    belief.conflict_explanation = "Equivocal modalities; falling back to temporal historical weights."
                    belief.confidence.beliefs = min(1.0, belief.confidence.beliefs + 0.05)

    def merge_beliefs(self, source_id: str, target_id: str, merged_statement: str) -> BeliefObject:
        """Merges source belief into target belief and removes source."""
        if source_id not in self.beliefs or target_id not in self.beliefs:
            raise InvalidBeliefUpdateError("Both source and target beliefs must exist to merge.")

        source = self.beliefs[source_id]
        target = self.beliefs[target_id]

        target.statement = merged_statement
        
        # Merge supporting data
        target.supporting_entities = list(set(target.supporting_entities + source.supporting_entities))
        target.supporting_events = list(set(target.supporting_events + source.supporting_events))
        target.supporting_relationships = list(set(target.supporting_relationships + source.supporting_relationships))
        target.supporting_evidence_ids = list(set(target.supporting_evidence_ids + source.supporting_evidence_ids))
        target.contradicting_evidence_ids = list(set(target.contradicting_evidence_ids + source.contradicting_evidence_ids))

        # Deterministic confidence merge: use higher confidence + boost
        target.confidence.beliefs = min(1.0, max(target.confidence.beliefs, source.confidence.beliefs) + config.BSE_REINFORCEMENT_BOOST)
        target.uncertainty = min(target.uncertainty, source.uncertainty)

        target.update_timestamp = time.time()
        target.expiration_timestamp = target.update_timestamp + config.BSE_EXPIRATION_TIME_SECONDS

        del self.beliefs[source_id]
        return target

    def reinforce_belief(self, belief_id: str, evidence_id: str) -> BeliefObject:
        """Strengthens belief based on new evidence."""
        belief = self.update_belief(belief_id, config.BSE_REINFORCEMENT_BOOST)
        belief.add_supporting_evidence(evidence_id)
        return belief

    def weaken_belief(self, belief_id: str, evidence_id: Optional[str] = None) -> BeliefObject:
        """Weakens belief explicitly (e.g., due to contradiction)."""
        belief = self.update_belief(belief_id, -config.BSE_CONTRADICTION_PENALTY)
        if evidence_id:
            belief.add_contradicting_evidence(evidence_id)
        return belief

    def deactivate_belief(self, belief_id: str):
        """Manually sets belief to INACTIVE."""
        if belief_id in self.beliefs:
            self.beliefs[belief_id].status = BeliefStatus.INACTIVE

    def remove_belief(self, belief_id: str):
        """Hard removes belief from memory."""
        if belief_id in self.beliefs:
            del self.beliefs[belief_id]

    def get_active_beliefs(self) -> List[BeliefObject]:
        """Returns all ACTIVE and WEAKENING beliefs."""
        return [b for b in self.beliefs.values() if b.status in (BeliefStatus.ACTIVE, BeliefStatus.WEAKENING)]

    def get_conflicting_beliefs(self) -> List[BeliefObject]:
        """Returns beliefs that have significant contradictions."""
        return [b for b in self.get_active_beliefs() if len(b.contradicting_evidence_ids) > 0]

    def update(self, events: list, relationships: list):
        """Mock update function for the end-to-end integration."""
        self.resolve_modality_conflicts()
        self._record_history()
        self._decay_beliefs()

    def snapshot(self) -> Dict[str, dict]:
        """Produces a serializable snapshot of current beliefs."""
        return {b_id: {
            "statement": b.statement,
            "confidence_val": b.confidence.p_val,
            "status": b.status.name,
            "conflict_score": b.conflict_score,
            "dominant_evidence": b.dominant_evidence
        } for b_id, b in self.beliefs.items()}

    def _record_history(self):
        """Records a frame into temporal memory."""
        self.belief_history.append(self.snapshot())
        if len(self.belief_history) > 100:
            self.belief_history.pop(0)

    def _decay_beliefs(self, current_time: Optional[float] = None):
        """Automatically weakens beliefs that haven't been updated recently and expires old ones."""
        t = current_time if current_time is not None else time.time()
        expired_ids = []
        for b_id, b in self.beliefs.items():
            if t > b.expiration_timestamp:
                expired_ids.append(b_id)
            elif t - b.update_timestamp > config.BSE_WEAKEN_TIME_SECONDS:
                if b.status == BeliefStatus.ACTIVE:
                    b.status = BeliefStatus.WEAKENING
                # Apply gradual decay
                b.confidence.beliefs = max(0.0, b.confidence.beliefs - config.BSE_WEAKENING_RATE)
                if b.confidence.beliefs < config.BSE_MIN_CONFIDENCE_THRESHOLD:
                    b.status = BeliefStatus.INACTIVE
        
        for b_id in expired_ids:
            self.remove_belief(b_id)

    def _garbage_collect(self):
        """Removes INACTIVE beliefs to free up memory up to the limits."""
        inactive = [b_id for b_id, b in self.beliefs.items() if b.status == BeliefStatus.INACTIVE]
        for b_id in inactive:
            self.remove_belief(b_id)
