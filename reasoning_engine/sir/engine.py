"""
Situation Intelligence Renderer (SIR) Core.
Converts structured CASRE data into human- and machine-readable reports.
No reasoning, no inference, strictly rendering.
"""
import json
from typing import List, Dict, Any, Optional

from reasoning_engine.sir.models import RenderMode
from reasoning_engine.sir import templates
from reasoning_engine import config
from reasoning_engine.bse.models import BeliefObject
from reasoning_engine.hre.models import Hypothesis
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe.models import SituationProjection
from reasoning_engine.tre.models import TransparentReasoningTrace

class SituationIntelligenceRenderer:
    def __init__(self):
        self._last_snapshot: Optional[Dict[str, Any]] = None

    def initialize(self):
        self.reset()

    def reset(self):
        self._last_snapshot = None

    def export(self, mode: RenderMode, world_state: WorldState, projection: SituationProjection, 
               trace: TransparentReasoningTrace, hypotheses: List[Hypothesis], beliefs: List[BeliefObject]) -> str:
        """Main delegator to specific rendering functions."""
        if mode == RenderMode.HUMAN:
            return self.render_human(world_state, projection, trace, hypotheses)
        elif mode == RenderMode.COMPACT:
            return self.render_compact(world_state, projection)
        elif mode == RenderMode.EMERGENCY:
            return self.render_emergency(world_state, projection)
        elif mode == RenderMode.DEVELOPER:
            return self.render_developer(world_state, projection, trace, hypotheses, beliefs)
        elif mode == RenderMode.JSON:
            return self.render_json(world_state, projection, trace)
        elif mode == RenderMode.API:
            return self.render_api(world_state, projection, trace)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def render_human(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace, hypotheses: List[Hypothesis]) -> str:
        supporting_evidence = []
        for link in trace.evidence_chain:
            if link.target_id == world_state.id:
                supporting_evidence.append(f"- {link.explanation}")
                
        alt_interps = []
        for state in world_state.secondary_states:
            alt_interps.append(f"- {state}")
            
        monitoring = "Continue monitoring active entities."
        if projection.urgency.name in ["HIGH", "IMMEDIATE"]:
            monitoring = "Immediate escalation recommended."

        return templates.HUMAN_TEMPLATE.format(
            current_situation=world_state.dominant_state,
            confidence=world_state.confidence.p_val,
            uncertainty=world_state.uncertainty,
            supporting_evidence="\n".join(supporting_evidence) if supporting_evidence else "None identified.",
            alternative_interpretations="\n".join(alt_interps) if alt_interps else "None identified.",
            projected_state=projection.primary_projection.state_description,
            alt_projected_state=projection.alternative_projection.state_description if projection.alternative_projection else "None",
            risk_level=projection.risk_level.name,
            urgency=projection.urgency.name,
            stability=projection.stability.name,
            monitoring_recommendation=monitoring
        )

    def render_compact(self, world_state: WorldState, projection: SituationProjection) -> str:
        output = templates.COMPACT_TEMPLATE.format(
            urgency=projection.urgency.name,
            current_situation=world_state.dominant_state,
            projected_state=projection.primary_projection.state_description,
            confidence=world_state.confidence.p_val
        )
        if len(output) > config.SIR_MAX_COMPACT_LENGTH:
            return output[:config.SIR_MAX_COMPACT_LENGTH-3] + "..."
        return output

    def render_emergency(self, world_state: WorldState, projection: SituationProjection) -> str:
        rec = "Evacuate / escalate immediately" if projection.urgency.name == "IMMEDIATE" else "Prepare for escalation"
        output = templates.EMERGENCY_TEMPLATE.format(
            current_situation=world_state.dominant_state,
            urgency=projection.urgency.name,
            risk_level=projection.risk_level.name,
            projected_state=projection.primary_projection.state_description,
            confidence=world_state.confidence.p_val,
            emergency_recommendation=rec
        )
        if len(output) > config.SIR_MAX_EMERGENCY_SUMMARY_LENGTH:
            return output[:config.SIR_MAX_EMERGENCY_SUMMARY_LENGTH-3] + "..."
        return output

    def render_developer(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace, hypotheses: List[Hypothesis], beliefs: List[BeliefObject]) -> str:
        contradictions = []
        for target, sources in trace.contradiction_chain.items():
            contradictions.append(f"- Hypothesis {target} overridden by {sources}")
            
        for b_id, res in trace.conflict_resolutions.items():
            contradictions.append(f"- Conflict in Belief {b_id}: {res}")
            
        if trace.hypothesis_competition:
            contradictions.append(f"- Winning Hypothesis {trace.hypothesis_competition.winning_hypothesis_id}: {trace.hypothesis_competition.winning_reason}")

        return templates.DEVELOPER_TEMPLATE.format(
            current_situation=world_state.dominant_state,
            confidence=world_state.confidence.p_val,
            uncertainty=world_state.ambiguity_score,
            active_entities=", ".join(world_state.active_entities),
            trace_id=trace.reasoning_id,
            hypotheses_count=len(hypotheses),
            beliefs_count=len(beliefs),
            projected_state=projection.primary_projection.state_description,
            alt_projected_state=projection.alternative_projection.state_description if projection.alternative_projection else "None",
            risk_level=projection.risk_level.name,
            urgency=projection.urgency.name,
            stability=projection.stability.name,
            contradictions="\n".join(contradictions) if contradictions else "No major contradictions logged."
        )

    def render_json(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace) -> str:
        """Returns a raw serializable dictionary payload as a JSON string."""
        payload = {
            "world_state": {
                "id": world_state.id,
                "dominant_state": world_state.dominant_state,
                "confidence": world_state.confidence.p_val,
                "uncertainty": world_state.uncertainty
            },
            "projection": {
                "id": projection.id,
                "projected_state": projection.primary_projection.state_description,
                "risk_level": projection.risk_level.name,
                "urgency": projection.urgency.name
            },
            "reasoning": {
                "trace_id": trace.reasoning_id,
                "assumptions": trace.assumptions
            }
        }
        return json.dumps(payload, indent=2)

    def render_api(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace) -> str:
        """Returns a REST-friendly wrapped JSON response."""
        # Uses render_json's payload under the hood
        raw_json = self.render_json(world_state, projection, trace)
        data_payload = json.loads(raw_json)
        
        api_response = {
            "status": "success",
            "code": 200,
            "data": data_payload,
            "meta": {
                "system": "CASRE v10.1",
                "timestamp": trace.timestamp
            }
        }
        return json.dumps(api_response, indent=2)

    def snapshot(self) -> dict:
        return {"status": "ACTIVE"}
