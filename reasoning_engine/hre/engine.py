"""
Hypothesis Reasoning Engine (HRE) Core.
Deterministically generates, maintains, evaluates and ranks competing hypotheses based on the Belief State Engine.
Supports continuous competition, automatic reinforcement, divergence, and history.
"""
import time
from typing import Dict, List, Optional

from reasoning_engine.hre.models import Hypothesis, HypothesisStatus
from reasoning_engine.hre.exceptions import HypothesisLimitExceededError, InvalidHypothesisUpdateError
from reasoning_engine import config
from reasoning_engine.awm.models import HierarchicalConfidence

class HypothesisReasoningEngine:
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        # Task 4: Hypothesis Evolution History
        self.hypothesis_history: List[Dict[str, dict]] = []

    def initialize(self):
        """Initializes or resets the Hypothesis Reasoning Engine."""
        self.reset()

    def reset(self):
        """Clears all hypotheses and resets internal state."""
        self.hypotheses.clear()
        self.hypothesis_history.clear()

    def generate(self, hypothesis_id: str, statement: str, 
                 supporting_beliefs: List[str] = None,
                 confidence: HierarchicalConfidence = None,
                 uncertainty: float = 0.5,
                 plausibility: float = 0.5) -> Hypothesis:
        """Generates a new candidate hypothesis deterministically."""
        # Note: Do not discard alternatives prematurely; relaxed garbage collection logic
        if hypothesis_id in self.hypotheses:
            # Automatic reinforcement
            return self.update(hypothesis_id, config.HRE_REINFORCEMENT_BOOST, 0.0, 0.0)

        hyp = Hypothesis(
            id=hypothesis_id,
            statement=statement,
            supporting_beliefs=supporting_beliefs or [],
            confidence=confidence or HierarchicalConfidence(),
            uncertainty=uncertainty,
            plausibility=plausibility
        )
        self.hypotheses[hypothesis_id] = hyp
        return hyp

    def evaluate(self, hypothesis_id: str, conf_delta: float, unc_delta: float, plaus_delta: float) -> Hypothesis:
        """Evaluates hypothesis with new metrics."""
        return self.update(hypothesis_id, conf_delta, unc_delta, plaus_delta)

    def update(self, hypothesis_id: str, conf_delta: float, unc_delta: float, plaus_delta: float) -> Hypothesis:
        """Deterministically updates metrics. Uses delta values rather than absolute to support continuous competition."""
        if hypothesis_id not in self.hypotheses:
            raise InvalidHypothesisUpdateError(f"Hypothesis {hypothesis_id} not found.")

        hyp = self.hypotheses[hypothesis_id]
        
        new_conf = hyp.confidence.hypotheses + conf_delta
        new_unc = hyp.uncertainty + unc_delta
        new_plaus = hyp.plausibility + plaus_delta
        
        hyp.update_metrics(new_conf, new_unc, new_plaus)
        return hyp

    def merge(self, source_id: str, target_id: str, merged_statement: str) -> Hypothesis:
        """Merges a source hypothesis into a target hypothesis, effectively combining competing identical ideas (Convergence)."""
        if source_id not in self.hypotheses or target_id not in self.hypotheses:
            raise InvalidHypothesisUpdateError("Both hypotheses must exist to merge.")

        source = self.hypotheses[source_id]
        target = self.hypotheses[target_id]

        target.statement = merged_statement
        
        # Combine evidence
        target.supporting_beliefs = list(set(target.supporting_beliefs + source.supporting_beliefs))
        target.supporting_entities = list(set(target.supporting_entities + source.supporting_entities))
        target.supporting_events = list(set(target.supporting_events + source.supporting_events))
        target.supporting_relationships = list(set(target.supporting_relationships + source.supporting_relationships))
        
        target.contradicting_beliefs = list(set(target.contradicting_beliefs + source.contradicting_beliefs))
        target.contradicting_evidence = list(set(target.contradicting_evidence + source.contradicting_evidence))
        target.missing_evidence = list(set(target.missing_evidence + source.missing_evidence))

        # Merge metrics
        target.confidence.hypotheses = min(1.0, max(target.confidence.hypotheses, source.confidence.hypotheses) + config.HRE_REINFORCEMENT_BOOST)
        target.uncertainty = min(target.uncertainty, source.uncertainty)
        target.plausibility = max(target.plausibility, source.plausibility)
        
        target.update_metrics(target.confidence.hypotheses, target.uncertainty, target.plausibility)
        
        # Remove source
        del self.hypotheses[source_id]
        return target

    def diverge(self, source_id: str, branch_id: str, branch_statement: str) -> Hypothesis:
        """Creates a divergent hypothesis from a source (Hypothesis Divergence)."""
        if source_id not in self.hypotheses:
            raise InvalidHypothesisUpdateError("Source hypothesis not found for divergence.")
            
        source = self.hypotheses[source_id]
        
        import copy
        new_conf = copy.deepcopy(source.confidence)
        new_conf.hypotheses = max(0.0, source.confidence.hypotheses - 0.2) # Penalty for diverging
        
        hyp = self.generate(
            branch_id, 
            branch_statement, 
            supporting_beliefs=list(source.supporting_beliefs),
            confidence=new_conf,
            uncertainty=min(1.0, source.uncertainty + 0.1),
            plausibility=source.plausibility
        )
        return hyp

    def rank(self) -> List[Hypothesis]:
        """
        Sorts and returns hypotheses by score. 
        Maintains multiple active competing hypotheses.
        Ranking formula: (confidence * 0.5) + (plausibility * 0.5) - (uncertainty * 0.2)
        """
        active = [h for h in self.hypotheses.values() if h.status != HypothesisStatus.REJECTED]
        
        def score_fn(h: Hypothesis) -> float:
            base_conf = h.confidence.hypotheses if h.confidence.hypotheses > 0 else h.confidence.beliefs
            return (base_conf * 0.5) + (h.plausibility * 0.5) - (h.uncertainty * 0.2)

        active.sort(key=score_fn, reverse=True)
        return active

    def synchronize(self, current_time: Optional[float] = None):
        """
        Synchronizes engine. Tracks history, applies automatic weakening, and garbage collects cautiously.
        """
        t = current_time if current_time is not None else time.time()
        
        for hyp in list(self.hypotheses.values()):
            # Automatic weakening if not reinforced recently
            if t - hyp.last_reinforced_time > config.HRE_WEAKEN_TIME_SECONDS:
                hyp.update_metrics(hyp.confidence.hypotheses - config.HRE_WEAKENING_RATE, hyp.uncertainty + 0.05, hyp.plausibility)
                
            # Cautious expiration
            if t - hyp.last_update > config.HRE_EXPIRATION_TIME_SECONDS * 2: # Keep them around longer for continuous competition
                hyp.status = HypothesisStatus.REJECTED
                
        self._record_history()
        self._garbage_collect()

    def snapshot(self) -> Dict[str, dict]:
        """Produces a serializable snapshot of the HRE state."""
        return {h.id: {
            "statement": h.statement,
            "confidence": h.confidence.p_val,
            "status": h.status.name,
            "plausibility": h.plausibility
        } for h in self.hypotheses.values()}

    def _record_history(self):
        """Records a frame into temporal memory."""
        self.hypothesis_history.append(self.snapshot())
        if len(self.hypothesis_history) > 100:
            self.hypothesis_history.pop(0)

    def _garbage_collect(self):
        """Removes REJECTED hypotheses cautiously."""
        # Only collect if we exceed limits to avoid premature deletion of alternatives
        if len(self.hypotheses) > config.HRE_MAX_ACTIVE_HYPOTHESES:
            rejected = [h.id for h in self.hypotheses.values() if h.status == HypothesisStatus.REJECTED]
            for h_id in rejected:
                del self.hypotheses[h_id]
