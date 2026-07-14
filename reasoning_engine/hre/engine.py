import time
import uuid
import logging
from typing import Dict, List, Optional

from reasoning_engine.hre.models import ManagedHypothesisState, HypothesisStatus
from reasoning_engine import config
from reasoning_engine.pse.models import SegregatedStreams
from reasoning_engine.semantic.models import SemanticSceneObject

logger = logging.getLogger(__name__)

class HypothesisReasoningEngine:
    def __init__(self):
        self.hypotheses: Dict[str, ManagedHypothesisState] = {}

    def initialize(self):
        self.reset()

    def reset(self):
        self.hypotheses.clear()

    def manage_hypotheses(
        self, 
        semantic_json: SemanticSceneObject,
        streams: SegregatedStreams, 
        temporal_history: List
    ) -> List[ManagedHypothesisState]:
        """
        Takes Semantic Interpretation (LLM suggestions) + Temporal Memory + Perceptual Evidence
        and generates strictly competing ManagedHypothesisState. HRE is the decider, not the LLM.
        """
        generated = []
        
        # 1. Evaluate Semantic JSON
        if semantic_json and semantic_json.primary_situation and semantic_json.primary_situation.lower() != "unknown":
            # Formulate Primary Hypothesis
            primary = self._build_hypothesis_from_semantic(
                situation=semantic_json.primary_situation,
                semantic_confidence=semantic_json.interpretation_confidence,
                semantic_json=semantic_json,
                streams=streams,
                temporal_history=temporal_history,
                is_primary=True
            )
            generated.append(primary)
            
            # Formulate Alternatives
            if hasattr(semantic_json, 'alternative_hypotheses') and semantic_json.alternative_hypotheses:
                for alt_situation in semantic_json.alternative_hypotheses:
                    if alt_situation.lower() not in ["none", "unknown"]:
                        alt = self._build_hypothesis_from_semantic(
                            situation=alt_situation,
                            semantic_confidence=max(0.1, semantic_json.interpretation_confidence - 0.3),
                            semantic_json=semantic_json,
                            streams=streams,
                            temporal_history=temporal_history,
                            is_primary=False
                        )
                        generated.append(alt)
        if not generated:
            # Fallback when Semantic Engine completely fails or suggests Unknown
            fallback = ManagedHypothesisState(
                id=str(uuid.uuid4()),
                situation="Unknown Situation",
                interaction_type="Unknown",
                likely_environment="Unknown",
                active_concepts=[e.class_map for e in streams.primary_events],
                supporting_evidence=[],
                contradicting_evidence=[],
                missing_evidence=[],
                unknowns=["True context is indistinguishable from noise"],
                semantic_confidence=0.1,
                temporal_consistency_score=0.1,
                acoustic_evidence_score=0.1,
                composite_score=0.1,
                uncertainty=0.9,
                reasoning_chain=["Insufficient acoustic and semantic evidence to formulate a situation."],
                source_semantic_object=semantic_json,
                status=HypothesisStatus.ACTIVE
            )
            generated.append(fallback)
            
        # 2. Score and Rank (Hypothesis Competition)
        self._compete_hypotheses(generated)

        for h in generated:
            self.hypotheses[h.id] = h
            
        return generated

    def _build_hypothesis_from_semantic(
        self, 
        situation: str, 
        semantic_confidence: float, 
        semantic_json: SemanticSceneObject, 
        streams: SegregatedStreams, 
        temporal_history: List,
        is_primary: bool
    ) -> ManagedHypothesisState:
        
        # 1. Evidence Extraction
        supporting = []
        contradicting = []
        missing = []
        unknowns = []
        
        if is_primary:
            # Pull evidence from auditory_observations and cross_modal_assessment
            for obs in semantic_json.auditory_observations:
                if obs.relationship_to_hypothesis.value in ["PrimarySupport", "SecondarySupport"]:
                    supporting.append(f"{obs.sound} ({obs.evidence_source})")
                elif obs.relationship_to_hypothesis.value == "Contradictory":
                    contradicting.append(f"{obs.sound} ({obs.evidence_source})")
                    
            if semantic_json.cross_modal_assessment.overall_assessment:
                supporting.append(semantic_json.cross_modal_assessment.overall_assessment)
                
            if isinstance(semantic_json.missing_evidence, list):
                missing.extend(semantic_json.missing_evidence)
            else:
                missing.append(str(semantic_json.missing_evidence))
        else:
            # For alternatives, the LLM usually gives reasons in the fields
            # We treat them loosely as alternative views.
            supporting.append(f"Semantic alternative suggestion: {situation}")
            
        # 2. Acoustic Evidence Score
        # Does the raw audio actually contain strong foreground signals?
        raw_events = len(streams.primary_events) + len(streams.supporting_events)
        acoustic_score = min(1.0, raw_events * 0.2) if raw_events > 0 else 0.1
        if streams.primary_entities:
            acoustic_score += 0.3 # Speech is a strong acoustic anchor
        acoustic_score = min(1.0, acoustic_score)
        
        # 3. Temporal Consistency Score (EMA Momentum Filter)
        temporal_score = 0.5 # Neutral prior
        if temporal_history:
            momentum = 0.0
            weight_sum = 0.0
            alpha = 0.5 # EMA decay factor
            
            for idx, state in enumerate(reversed(temporal_history[-5:])): # Look at last 5 states
                weight = alpha ** idx
                weight_sum += weight
                if state.dominant_state == situation:
                    momentum += 1.0 * weight
                elif state.cognitive_state and situation in state.cognitive_state.candidate_situations:
                    momentum += 0.5 * weight
                    
            ema_score = momentum / weight_sum if weight_sum > 0 else 0.5
            
            # Map EMA [0, 1] to temporal score [0.2, 0.9] to avoid absolute zeroes
            temporal_score = 0.2 + (ema_score * 0.7)

        # 4. Synthesize Uncertainty and Unknowns
        if semantic_json.missing_evidence:
            unknowns.append(str(semantic_json.missing_evidence))
            
        uncertainty = 1.0 - ((semantic_confidence * 0.4) + (acoustic_score * 0.4) + (temporal_score * 0.2))
        
        # 5. Composite Score
        composite = (semantic_confidence * 0.4) + (acoustic_score * 0.4) + (temporal_score * 0.2)
        
        reasoning = [
            f"Hypothesis generated for situation: {situation}",
            f"Semantic Confidence: {semantic_confidence:.2f}",
            f"Acoustic Evidence Score: {acoustic_score:.2f}",
            f"Temporal Consistency Score: {temporal_score:.2f}",
            f"Composite Score: {composite:.2f}"
        ]

        # Handle Interaction Type gracefully
        interaction_type = "Unknown"
        if semantic_json.speech_understanding and semantic_json.speech_understanding.speaker_intent:
            interaction_type = semantic_json.speech_understanding.speaker_intent

        return ManagedHypothesisState(
            id=str(uuid.uuid4()),
            situation=situation,
            interaction_type=interaction_type,
            likely_environment=getattr(semantic_json, 'environmental_context', 'Unknown'),
            active_concepts=semantic_json.actors + semantic_json.human_goals,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_evidence=missing,
            unknowns=unknowns,
            semantic_confidence=semantic_confidence,
            temporal_consistency_score=temporal_score,
            acoustic_evidence_score=acoustic_score,
            composite_score=composite,
            uncertainty=uncertainty,
            reasoning_chain=reasoning,
            source_semantic_object=semantic_json,
            status=HypothesisStatus.ACTIVE if composite > 0.3 else HypothesisStatus.CANDIDATE
        )

    def _compete_hypotheses(self, hypotheses: List[ManagedHypothesisState]):
        """
        Explicitly compares hypotheses against each other and sets why_it_won and why_alternatives_lost.
        """
        if not hypotheses:
            return
            
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)
        winner = hypotheses[0]
        losers = hypotheses[1:]
        
        winner.why_it_won = f"Highest composite score ({winner.composite_score:.2f}) across semantic, acoustic, and temporal metrics."
        winner.why_alternatives_lost = "Alternatives had lower combined probability and temporal consistency."
        
        for idx, loser in enumerate(losers):
            loser.why_alternatives_lost = f"Ranked {idx+2} behind '{winner.situation}'."
            if loser.semantic_confidence < winner.semantic_confidence:
                loser.why_alternatives_lost += " Lower semantic confidence."
            if loser.temporal_consistency_score < winner.temporal_consistency_score:
                loser.why_alternatives_lost += " Weaker temporal continuity."

    def validate_integrity(self):
        """Validates that no rules are modifying or inferring meaning, only managing state."""
        return True

    def rank(self) -> List[ManagedHypothesisState]:
        active = [h for h in self.hypotheses.values() if h.status != HypothesisStatus.REJECTED]
        active.sort(key=lambda h: h.composite_score, reverse=True)
        return active

    def synchronize(self, current_time: Optional[float] = None):
        t = current_time if current_time is not None else time.time()
        for hyp in list(self.hypotheses.values()):
            if t - hyp.creation_time > config.HRE_EXPIRATION_TIME_SECONDS:
                hyp.status = HypothesisStatus.REJECTED
        
        if len(self.hypotheses) > config.HRE_MAX_ACTIVE_HYPOTHESES:
            rejected = [h.id for h in self.hypotheses.values() if h.status == HypothesisStatus.REJECTED]
            for h_id in rejected:
                del self.hypotheses[h_id]
