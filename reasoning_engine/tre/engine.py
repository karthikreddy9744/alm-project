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
from reasoning_engine.hre.models import ManagedHypothesisState
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe.models import SituationProjection

class TransparentReasoningEngine:
    def __init__(self):
        self.current_trace: Optional[TransparentReasoningTrace] = None
        self.history: List[TransparentReasoningTrace] = []

    def initialize(self):
        self.reset()

    def reset(self):
        self.current_trace = None
        self.history.clear()

    def trace(self, awm_entities: dict, awm_events: dict, hypotheses: List[ManagedHypothesisState], world_state: WorldState, 
              projection: SituationProjection) -> TransparentReasoningTrace:
        """
        Builds a complete, deterministic, backwards-traceable reasoning chain without ARG/BSE.
        """
        trace_id = str(uuid.uuid4())
        trace = TransparentReasoningTrace(reasoning_id=trace_id)

        # 1. Capture Raw Pointers
        trace.observations = list(awm_entities.keys()) + list(awm_events.keys())
        trace.hypotheses = [h.id for h in hypotheses]
        trace.world_state = world_state.id
        trace.projections = [projection.id]

        # 2. Build Evidence Chain (Backwards Trace)
        
        # Projection -> World State
        cs = world_state.cognitive_state
        
        trace.evidence_chain.append(
            ExplanationLink(
                source_ids=[world_state.id],
                target_id=projection.id,
                explanation=f"Projected {projection.primary_projection.state_description} from current dominant state {world_state.dominant_state}",
                confidence_explanation=self.explain_decomposed_confidence(cs.confidence) if cs else "Unknown",
                uncertainty_explanation=self.explain_uncertainty(1.0 - (cs.confidence.overall if cs else 0.0))
            )
        )

        # World State -> Hypotheses
        if cs:
            trace.evidence_chain.append(
                ExplanationLink(
                    source_ids=[h.id for h in hypotheses], # Map all to WorldState
                    target_id=world_state.id,
                    explanation=f"Derived Cognitive State using Semantic Concepts and Context.",
                    confidence_explanation=self.explain_decomposed_confidence(cs.confidence),
                    uncertainty_explanation=self.explain_uncertainty(cs.uncertainty)
                )
            )
            
        self.explain_contradictions(trace, world_state, hypotheses)
        self.explain_hypothesis_competition(trace, world_state, hypotheses)

        # Hypotheses -> Evidence (AWM Entities/Events & Semantic Engine)
        for hyp in hypotheses:
            trace.evidence_chain.append(
                ExplanationLink(
                    source_ids=[], # Textual evidence is in explanation
                    target_id=hyp.id,
                    explanation=f"Hypothesis '{hyp.situation}' supported by evidence: {hyp.supporting_evidence}",
                    confidence_explanation=f"Composite Score: {hyp.composite_score:.2f}",
                    uncertainty_explanation=self.explain_uncertainty(hyp.uncertainty)
                )
            )

        # Extract Assumptions
        self.explain_assumptions(trace, world_state, projection)

        # Update engine state
        if self.current_trace:
            self.history.append(self.current_trace)
        self.current_trace = trace
        
        return trace

    def explain_decomposed_confidence(self, score) -> str:
        if score.overall >= config.TRE_HIGH_CONFIDENCE_THRESHOLD:
            return config.TRE_EXP_STRONG_SUPPORT
        return config.TRE_EXP_WEAK_SUPPORT

    def explain_uncertainty(self, score: float) -> str:
        if score >= config.TRE_HIGH_UNCERTAINTY_THRESHOLD:
            return config.TRE_EXP_HIGH_UNCERTAINTY
        return config.TRE_EXP_LOW_UNCERTAINTY

    def explain_contradictions(self, trace: TransparentReasoningTrace, world_state: WorldState, hypotheses: List[ManagedHypothesisState]):
        cs = world_state.cognitive_state
        if cs and cs.missing_evidence:
            for contradiction in cs.missing_evidence:
                trace.contradiction_chain[contradiction] = [world_state.dominant_state]
                
    def explain_hypothesis_competition(self, trace: TransparentReasoningTrace, world_state: WorldState, hypotheses: List[ManagedHypothesisState]):
        if not hypotheses:
            return
            
        winner = hypotheses[0]
        losers = hypotheses[1:]
        
        comp_exp = HypothesisCompetitionExplanation(
            winning_hypothesis_id=winner.situation,
            winning_reason=winner.why_it_won
        )
        
        for alt in losers:
            comp_exp.losing_hypotheses[alt.situation] = alt.why_alternatives_lost
                
        trace.hypothesis_competition = comp_exp

    def explain_assumptions(self, trace: TransparentReasoningTrace, world_state: WorldState, projection: SituationProjection):
        cs = world_state.cognitive_state
        if cs and cs.confidence.overall < 0.5:
            trace.assumptions.append("Assuming primary state despite low overall confidence.")
        
        if projection.primary_projection.uncertainty >= config.TRE_HIGH_UNCERTAINTY_THRESHOLD:
            trace.assumptions.append("Projection heavily relies on assumed continuity of unverified entities.")

    def snapshot(self) -> dict:
        if not self.current_trace:
            return {"status": "NO_TRACE"}
            
        return {
            "reasoning_id": self.current_trace.reasoning_id,
            "evidence_chain_length": len(self.current_trace.evidence_chain),
            "assumptions": self.current_trace.assumptions,
            "contradictions_tracked": len(self.current_trace.contradiction_chain),
            "conflict_resolutions": len(self.current_trace.conflict_resolutions)
        }
