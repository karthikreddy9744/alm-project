"""
Transparent Reasoning Engine (TRE) Core.
Generates an auditable trace mapping the entire reasoning chain for CASRE.
Explains confidence propagation, hypothesis competition, and conflict resolution.
"""
import uuid
import time
from typing import Dict, List, Optional

from reasoning_engine.tre.models import TransparentReasoningTrace, ExplanationLink, HypothesisCompetitionExplanation
from reasoning_engine import config
from reasoning_engine.bse.models import BeliefObject
from reasoning_engine.hre.models import Hypothesis
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe.models import SituationProjection
from reasoning_engine.awm.models import HierarchicalConfidence

class TransparentReasoningEngine:
    def __init__(self):
        self.current_trace: Optional[TransparentReasoningTrace] = None
        self.history: List[TransparentReasoningTrace] = []

    def initialize(self):
        self.reset()

    def reset(self):
        self.current_trace = None
        self.history.clear()

    def trace(self, awm: dict, arg: dict, beliefs: List[BeliefObject], 
              hypotheses: List[Hypothesis], world_state: WorldState, 
              projection: SituationProjection) -> TransparentReasoningTrace:
        """
        Builds a complete, deterministic, backwards-traceable reasoning chain.
        """
        trace_id = str(uuid.uuid4())
        trace = TransparentReasoningTrace(reasoning_id=trace_id)

        # 1. Capture Raw Pointers
        trace.observations = list(awm.keys())
        trace.relationships = list(arg.keys())
        trace.beliefs = [b.id for b in beliefs]
        trace.hypotheses = [h.id for h in hypotheses]
        trace.world_state = world_state.id
        trace.projections = [projection.id]

        # 2. Build Evidence Chain (Backwards Trace)
        
        # Projection -> World State
        trace.evidence_chain.append(
            ExplanationLink(
                source_ids=[world_state.id],
                target_id=projection.id,
                explanation=f"Projected {projection.primary_projection.state_description} from current dominant state {world_state.dominant_state}",
                confidence_explanation=self.explain_confidence(projection.primary_projection.confidence),
                uncertainty_explanation=self.explain_uncertainty(projection.primary_projection.uncertainty)
            )
        )

        # World State -> Hypotheses
        trace.evidence_chain.append(
            ExplanationLink(
                source_ids=world_state.supporting_hypotheses,
                target_id=world_state.id,
                explanation=f"Derived World State from top hypotheses.",
                confidence_explanation=self.explain_confidence(world_state.confidence),
                uncertainty_explanation=self.explain_uncertainty(world_state.uncertainty)
            )
        )
        
        self.explain_contradictions(trace, world_state, hypotheses)
        self.explain_hypothesis_competition(trace, world_state, hypotheses)

        # Hypotheses -> Beliefs
        for hyp in hypotheses:
            if hyp.id in world_state.supporting_hypotheses or hyp.id in world_state.rejected_hypotheses:
                trace.evidence_chain.append(
                    ExplanationLink(
                        source_ids=hyp.supporting_beliefs,
                        target_id=hyp.id,
                        explanation=f"Hypothesis '{hyp.statement}' supported by underlying beliefs.",
                        confidence_explanation=self.explain_confidence(hyp.confidence),
                        uncertainty_explanation=self.explain_uncertainty(hyp.uncertainty)
                    )
                )

        # Extract Modality Conflicts from Beliefs
        for belief in beliefs:
            if getattr(belief, "conflict_score", 0) > 0:
                trace.conflict_resolutions[belief.id] = f"Dominant Modality: {belief.dominant_evidence} - {belief.conflict_explanation}"

        # Extract Assumptions
        self.explain_assumptions(trace, world_state, projection)

        # Update engine state
        if self.current_trace:
            self.history.append(self.current_trace)
        self.current_trace = trace
        
        return trace

    def explain_confidence(self, score: HierarchicalConfidence) -> str:
        """Deterministically explains a hierarchical confidence score."""
        max_layer = score.p_val
        if max_layer >= config.TRE_HIGH_CONFIDENCE_THRESHOLD:
            return config.TRE_EXP_STRONG_SUPPORT
        return config.TRE_EXP_WEAK_SUPPORT

    def explain_uncertainty(self, score: float) -> str:
        """Deterministically explains an uncertainty score."""
        if score >= config.TRE_HIGH_UNCERTAINTY_THRESHOLD:
            return config.TRE_EXP_HIGH_UNCERTAINTY
        return config.TRE_EXP_LOW_UNCERTAINTY

    def explain_contradictions(self, trace: TransparentReasoningTrace, world_state: WorldState, hypotheses: List[Hypothesis]):
        """Documents evidence that was overridden."""
        for hyp in hypotheses:
            if hyp.id in world_state.rejected_hypotheses:
                trace.contradiction_chain[hyp.id] = [world_state.dominant_state]
                
    def explain_hypothesis_competition(self, trace: TransparentReasoningTrace, world_state: WorldState, hypotheses: List[Hypothesis]):
        """Explains why one hypothesis won and why others lost."""
        if not world_state.supporting_hypotheses:
            return
            
        winning_id = world_state.supporting_hypotheses[0]
        comp_exp = HypothesisCompetitionExplanation(
            winning_hypothesis_id=winning_id,
            winning_reason=f"Achieved highest combination of evidence confidence and ecological plausibility (Eco Plausibility: {world_state.ecological_plausibility:.2f})"
        )
        
        for hyp in hypotheses:
            if hyp.id in world_state.rejected_hypotheses:
                comp_exp.losing_hypotheses[hyp.id] = f"Defeated by higher confidence alternative or lower ecological plausibility ({hyp.plausibility:.2f})"
                
        trace.hypothesis_competition = comp_exp

    def explain_assumptions(self, trace: TransparentReasoningTrace, world_state: WorldState, projection: SituationProjection):
        """Documents assumptions based on missing evidence or high ambiguity."""
        if world_state.ambiguity_score > config.TRE_HIGH_CONTRADICTION_THRESHOLD:
            trace.assumptions.append("Assuming primary state despite strong secondary interpretations.")
        
        if projection.primary_projection.uncertainty >= config.TRE_HIGH_UNCERTAINTY_THRESHOLD:
            trace.assumptions.append("Projection heavily relies on assumed continuity of unverified entities.")

    def snapshot(self) -> dict:
        """Returns a serialized snapshot of the current trace."""
        if not self.current_trace:
            return {"status": "NO_TRACE"}
            
        return {
            "reasoning_id": self.current_trace.reasoning_id,
            "evidence_chain_length": len(self.current_trace.evidence_chain),
            "assumptions": self.current_trace.assumptions,
            "contradictions_tracked": len(self.current_trace.contradiction_chain),
            "conflict_resolutions": len(self.current_trace.conflict_resolutions)
        }
