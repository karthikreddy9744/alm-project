import json
from typing import List, Dict, Any, Optional

from reasoning_engine.sir.models import RenderMode
from reasoning_engine.sir import templates
from reasoning_engine import config
from reasoning_engine.hre.models import ManagedHypothesisState
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
               trace: TransparentReasoningTrace, hypotheses: List[ManagedHypothesisState], streams: Any = None, awm: Any = None) -> str:
        """Main delegator to specific rendering functions."""
        if mode == RenderMode.HUMAN_COGNITIVE:
            return self.render_human_cognitive(world_state)
        elif mode == RenderMode.HUMAN:
            return self.render_human(world_state, projection, trace, hypotheses)
        elif mode == RenderMode.COMPACT:
            return self.render_compact(world_state, projection)
        elif mode == RenderMode.EMERGENCY:
            return self.render_emergency(world_state, projection)
        elif mode == RenderMode.DEVELOPER:
            return self.render_developer(world_state, projection, trace, hypotheses)
        elif mode == RenderMode.JSON:
            return self.render_json(world_state, projection, trace)
        elif mode == RenderMode.API:
            return self.render_api(world_state, projection, trace)
        elif mode == RenderMode.THREE_TIER_REPORT:
            return self.render_three_tier_report(world_state, projection, trace, hypotheses, streams, awm)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def render_human_cognitive(self, world_state: WorldState) -> str:
        """Strictly serializes the structured CognitiveState object into text using a natural language template."""
        cs = world_state.cognitive_state
        if not cs:
            return "Cognitive Understanding Pending..."
            
        evidence_lines = []
        for ev in cs.evidence_references:
            evidence_lines.append(f"- [{ev.classification}] **{ev.name}** (Weight: {ev.weight:.2f}): {ev.reason}")
        evidence_list = "\n".join(evidence_lines) if evidence_lines else "No evidence recorded."

        chain_lines = []
        for idx, step in enumerate(cs.reasoning_trace):
            chain_lines.append(f"{idx+1}. {step}")
        reasoning_chain = "\n".join(chain_lines) if chain_lines else "No reasoning chain recorded."

        contradictions = "\n".join([f"- {c}" for c in cs.missing_evidence]) if cs.missing_evidence else "- Consistent: No cognitive contradictions detected."

        return templates.HUMAN_COGNITIVE_TEMPLATE.format(
            evidence_list=evidence_list,
            environmental_context=cs.environment,
            active_concepts=", ".join(cs.intentions) if cs.intentions else "None",
            missing_concepts=", ".join(cs.missing_evidence) if cs.missing_evidence else "None",
            current_situation=cs.current_situation,
            who_is_involved=", ".join(cs.actors) if cs.actors else "None identified",
            rejected_alternatives=cs.previous_situation if cs.previous_situation else "None",
            internal_contradictions=contradictions,
            conf_perception=cs.confidence.perception,
            conf_context=cs.confidence.context,
            conf_situation=cs.confidence.situation,
            conf_projection=cs.confidence.projection,
            conf_overall=cs.confidence.overall,
            reasoning_chain=reasoning_chain,
            future_projection=cs.projection
        )

    def render_human(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace, hypotheses: List[ManagedHypothesisState]) -> str:
        cs = world_state.cognitive_state
        supporting_evidence = []
        for link in trace.evidence_chain:
            if link.target_id == world_state.id:
                supporting_evidence.append(f"- {link.explanation}")
                
        alt_interps = []
        if hypotheses and len(hypotheses) > 1:
            for state in hypotheses[1:]:
                alt_interps.append(f"- {state.situation}")
            
        monitoring = "Continue monitoring active entities."
        if projection.urgency in ["HIGH", "IMMEDIATE"]:
            monitoring = "Immediate escalation recommended."

        return templates.HUMAN_TEMPLATE.format(
            current_situation=world_state.dominant_state,
            confidence=cs.confidence.overall if cs else 0.0,
            uncertainty=cs.uncertainty if cs else 1.0,
            supporting_evidence="\n".join(supporting_evidence) if supporting_evidence else "None identified.",
            alternative_interpretations="\n".join(alt_interps) if alt_interps else "None identified.",
            projected_state=projection.primary_projection.state_description,
            alt_projected_state=projection.alternative_projection.state_description if projection.alternative_projection else "None",
            risk_level=projection.risk_level,
            urgency=projection.urgency,
            stability=projection.stability,
            monitoring_recommendation=monitoring
        )

    def render_compact(self, world_state: WorldState, projection: SituationProjection) -> str:
        cs = world_state.cognitive_state
        output = templates.COMPACT_TEMPLATE.format(
            urgency=projection.urgency,
            current_situation=world_state.dominant_state,
            projected_state=projection.primary_projection.state_description,
            confidence=cs.confidence.overall if cs else 0.0
        )
        if len(output) > config.SIR_MAX_COMPACT_LENGTH:
            return output[:config.SIR_MAX_COMPACT_LENGTH-3] + "..."
        return output

    def render_emergency(self, world_state: WorldState, projection: SituationProjection) -> str:
        cs = world_state.cognitive_state
        rec = "Evacuate / escalate immediately" if projection.urgency == "IMMEDIATE" else "Prepare for escalation"
        output = templates.EMERGENCY_TEMPLATE.format(
            current_situation=world_state.dominant_state,
            urgency=projection.urgency,
            risk_level=projection.risk_level,
            projected_state=projection.primary_projection.state_description,
            confidence=cs.confidence.overall if cs else 0.0,
            emergency_recommendation=rec
        )
        if len(output) > config.SIR_MAX_EMERGENCY_SUMMARY_LENGTH:
            return output[:config.SIR_MAX_EMERGENCY_SUMMARY_LENGTH-3] + "..."
        return output

    def render_developer(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace, hypotheses: List[ManagedHypothesisState]) -> str:
        cs = world_state.cognitive_state
        if not cs:
            return "CognitiveState is completely empty."
            
        output = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                             COGNITIVE STATE                                  ║",
            "╚══════════════════════════════════════════════════════════════════════════════╝",
            "",
            "### SCENE UNDERSTANDING",
            f"- **What is happening:** {cs.current_situation}",
            f"- **Where is it happening:** {cs.environment}",
            f"- **Who is involved:** {', '.join(cs.actors) if cs.actors else 'Unknown'}",
            f"- **Missing Information:** {', '.join(cs.missing_evidence) if cs.missing_evidence else 'None'}",
            "",
            "### EVIDENCE (OBSERVED VS INFERRED)",
        ]
        
        for ev in cs.evidence_references:
            output.append(f"- [{ev.classification.upper()}] {ev.name} ({ev.reason}) [Weight: {ev.weight:.2f}]")
            
        output.extend([
            "",
            "### COMPETING HYPOTHESES"
        ])
        
        for idx, h in enumerate(hypotheses):
            output.append(f"{idx+1}. **{h.situation}**")
            output.append(f"   - Semantic Confidence: {h.semantic_confidence:.2f}")
            output.append(f"   - Acoustic Score: {h.acoustic_evidence_score:.2f}")
            output.append(f"   - Temporal Score: {h.temporal_consistency_score:.2f}")
            output.append(f"   - Composite Rank: {h.composite_score:.2f}")
            if h.id == cs.current_situation or h.situation == cs.current_situation:
                output.append(f"   - Why it won: {h.why_it_won}")
            else:
                output.append(f"   - Why it lost: {h.why_alternatives_lost}")
                
        output.extend([
            "",
            "### REASONING CHAIN"
        ])
        
        for step in cs.reasoning_trace:
            output.append(f"> {step}")
            
        output.extend([
            "",
            "### CONFIDENCE & UNCERTAINTY",
            f"- Overall Confidence: {cs.confidence.overall:.2f}",
            f"- Uncertainty/Unknowns: {cs.uncertainty:.2f}",
            "",
            "### PROJECTION",
            f"- {projection.primary_projection.state_description}"
        ])
        
        return "\n".join(output)

    def render_json(self, world_state: WorldState, projection: SituationProjection, trace: TransparentReasoningTrace) -> str:
        """Returns a raw serializable dictionary payload as a JSON string."""
        cs = world_state.cognitive_state
        payload = {
            "world_state": {
                "id": world_state.id,
                "dominant_state": world_state.dominant_state,
                "confidence": cs.confidence.overall if cs else 0.0,
                "perception_confidence": cs.confidence.perception if cs else 0.0,
                "context_confidence": cs.confidence.context if cs else 0.0,
                "context": cs.environment if cs else "Unknown"
            },
            "projection": {
                "id": projection.id,
                "projected_state": projection.primary_projection.state_description,
                "risk_level": projection.risk_level,
                "urgency": projection.urgency
            },
            "reasoning": {
                "trace_id": trace.reasoning_id,
                "assumptions": trace.assumptions,
                "reasoning_chain": cs.reasoning_trace if cs else []
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
                "system": "ALM v12.0",
                "timestamp": trace.timestamp
            }
        }
        return json.dumps(api_response, indent=2)

    def snapshot(self) -> dict:
        return {"status": "ACTIVE"}

    def render_three_tier_report(self, world_state: WorldState, projection: SituationProjection, 
                                 trace: TransparentReasoningTrace, hypotheses: List[ManagedHypothesisState], streams: Any, awm: Any = None) -> str:
        
        # ---------------------------------------------------------
        # OUTPUT 1: SPEECH UNDERSTANDING
        # ---------------------------------------------------------
        speech_lines = []
        has_speech = False
        all_entities = []
        if streams:
            all_entities = streams.primary_entities + streams.supporting_entities
            
        for e in all_entities:
            if getattr(e, 'entity_type', '') == 'Speaker':
                has_speech = True
                transcript = getattr(e, 'transcript', 'Unknown')
                language = getattr(e, 'language', 'en')
                speech_conf = e.confidence.speech_recognition if hasattr(e, 'confidence') else 0.0
                speaker_id = e.id
                
                speech_lines.append(f"- **Speaker ID:** {speaker_id}")
                speech_lines.append(f"  - **Transcript:** \"{transcript}\"")
                speech_lines.append(f"  - **Language:** {language}")
                speech_lines.append(f"  - **Confidence:** {speech_conf:.2f}")
                
        if not has_speech:
            speech_lines.append("- No speech detected.")
            
        speech_understanding = "\n".join(speech_lines)
        
        # ---------------------------------------------------------
        # OUTPUT 2: ENVIRONMENTAL UNDERSTANDING
        # ---------------------------------------------------------
        env_lines = []
        source_semantic = hypotheses[0].source_semantic_object if (hypotheses and hasattr(hypotheses[0], 'source_semantic_object')) else None
        
        if source_semantic and hasattr(source_semantic, 'auditory_observations') and source_semantic.auditory_observations:
            env_lines.append("### Auditory Observations")
            for obs in source_semantic.auditory_observations:
                used = " (Used in Final Reasoning)" if obs.used_in_final_reasoning else ""
                env_lines.append(f"- **{obs.sound}** [{obs.id}]: Detected by {obs.evidence_source} ({obs.start_time}-{obs.end_time}s) | Influence: {obs.influence.value}{used}")
                env_lines.append(f"  - *Relationship:* {obs.relationship_to_hypothesis.value}")
                env_lines.append(f"  - *Justification:* {obs.justification}")
        else:
            if not streams:
                env_lines.append("- No environmental events recorded.")
            else:
                if streams.primary_events:
                    env_lines.append("- **Primary Sounds:** " + ", ".join([f"{e.class_map} ({e.acoustic_salience:.2f})" for e in streams.primary_events]))
                else:
                    env_lines.append("- **Primary Sounds:** None")
                    
                if streams.supporting_events:
                    env_lines.append("- **Supporting Sounds:** " + ", ".join([f"{e.class_map} ({e.acoustic_salience:.2f})" for e in streams.supporting_events]))
                else:
                    env_lines.append("- **Supporting Sounds:** None")
                    
                if streams.background_events:
                    env_lines.append("- **Background Sounds:** " + ", ".join([f"{e.class_map} ({e.acoustic_salience:.2f})" for e in streams.background_events]))
                else:
                    env_lines.append("- **Background Sounds:** None")
                    
                if streams.ignored_events:
                    env_lines.append("- **Ignored Sounds:** " + ", ".join([f"{e.class_map}" for e in streams.ignored_events]))
                else:
                    env_lines.append("- **Ignored Sounds:** None")

        if source_semantic and hasattr(source_semantic, 'cross_modal_assessment') and source_semantic.cross_modal_assessment:
            cma = source_semantic.cross_modal_assessment
            env_lines.append("\n### Cross-Modal Evidence Assessment")
            env_lines.append(f"- **Agreement Level:** {cma.agreement_level.value}")
            env_lines.append(f"- **Verification Status:** {cma.verification_status.value}")
            env_lines.append(f"- **Dominant Modality:** {cma.dominant_modality.value}")
            env_lines.append(f"- **Major Supports:** {', '.join(cma.major_supports) if cma.major_supports else 'None'}")
            env_lines.append(f"- **Major Conflicts:** {', '.join(cma.major_conflicts) if cma.major_conflicts else 'None'}")
            env_lines.append(f"- **Overall Assessment:** {cma.overall_assessment}")
            env_lines.append(f"- **Remaining Uncertainty:** {cma.remaining_uncertainty}")

        if source_semantic and hasattr(source_semantic, 'audio_provenance_reasoning') and source_semantic.audio_provenance_reasoning:
            prov = source_semantic.audio_provenance_reasoning
            env_lines.append("\n### Audio Provenance Reasoning")
            env_lines.append(f"- **Source Type:** {prov.source_type.value}")
            env_lines.append(f"- **Representation Type:** {prov.representation_type.value}")
            env_lines.append(f"- **Reliability:** {prov.provenance_reliability.value}")
            env_lines.append(f"- **Confidence:** {prov.confidence:.2f}")
            env_lines.append(f"- **Supporting Evidence:** {', '.join(prov.supporting_evidence) if prov.supporting_evidence else 'None'}")
            env_lines.append(f"- **Remaining Uncertainty:** {prov.remaining_uncertainty}")

        environmental_understanding = "\n".join(env_lines)

        # ---------------------------------------------------------
        # OUTPUT 3: HUMAN-ORIENTED AUDITORY SITUATION UNDERSTANDING
        # ---------------------------------------------------------
        cs = world_state.cognitive_state
        if not cs:
            return "Cognitive State Missing"
            
        situation = cs.current_situation
        context = cs.environment
        
        # Evidence
        obs_ev = [e.name for e in cs.evidence_references if e.classification == "Observed Evidence"]
        if obs_ev:
            primary_evidence = ", ".join(obs_ev)
        else:
            primary_evidence = "unclear acoustic cues"
            
        # Alternatives
        if hypotheses and len(hypotheses) > 1:
            alts = ", ".join([h.situation for h in hypotheses[1:]])
            why_won = hypotheses[0].why_it_won if hypotheses else "it fit the evidence better"
            alt_sentence = f"Alternative interpretations such as {alts} were considered but rejected because {why_won}."
        else:
            alt_sentence = "No strong alternative interpretations were considered."
            
        # Missing
        if cs.missing_evidence:
            missing = ", ".join(cs.missing_evidence)
            missing_sentence = f"Missing information such as {missing} adds to the uncertainty (Confidence: {cs.confidence.overall:.2f})."
        else:
            missing_sentence = f"The available evidence provides a comprehensive picture (Confidence: {cs.confidence.overall:.2f})."
            
        # Projection
        if projection and projection.primary_projection:
            future_proj = projection.primary_projection.state_description
        else:
            future_proj = "Unknown"
            
        # Prioritize the LLM's empathetic human_oriented_summary if it exists in the source semantic object.
        source_semantic = hypotheses[0].source_semantic_object if (hypotheses and hasattr(hypotheses[0], 'source_semantic_object')) else None
        if source_semantic and hasattr(source_semantic, 'human_oriented_summary') and source_semantic.human_oriented_summary and source_semantic.human_oriented_summary != "Unknown Situation":
            human_explanation = source_semantic.human_oriented_summary
        else:
            conf = cs.confidence.overall
            if conf >= 0.85:
                certainty_phrase = f"It is highly evident that a **{situation}** is taking place"
            elif conf >= 0.60:
                certainty_phrase = f"The evidence strongly suggests a **{situation}** is taking place"
            elif conf >= 0.40:
                certainty_phrase = f"There are indications of a **{situation}** taking place, though it is not definitive"
            else:
                certainty_phrase = f"There is a remote possibility of a **{situation}** taking place, but the audio is highly ambiguous"
                
            proj_sentence = f"Looking ahead, we can expect that {future_proj}." if future_proj != "Unknown" else ""
            human_explanation = f"{certainty_phrase} in {context}. The primary acoustic evidence driving this conclusion includes {primary_evidence}. {alt_sentence} {missing_sentence} {proj_sentence}"

        intent = ", ".join(cs.actors) if cs.actors else "Unknown"
        alts_str = ", ".join([h.situation for h in hypotheses[1:]]) if hypotheses and len(hypotheses) > 1 else "None"
        missing_str = ", ".join(cs.missing_evidence) if cs.missing_evidence else "None"
        
        speaker_role = "Unknown"
        if source_semantic and hasattr(source_semantic, 'speech_understanding') and source_semantic.speech_understanding:
            speaker_role = getattr(source_semantic.speech_understanding, 'speaker_role', 'Unknown')
            
        recording_char_str = "Unknown"
        if awm and hasattr(awm, 'recording_characterization') and awm.recording_characterization:
            rc = awm.recording_characterization
            recording_char_str = f"Quality: {rc.recording_quality} | Dynamic Range: {rc.dynamic_range} | Reverb: {rc.reverberation_profile} | Noise: {rc.noise_profile}"

        situation_details = (
            f"**Detailed Breakdown:**\n"
            f"- **Situation:** {situation}\n"
            f"- **Context:** {context}\n"
            f"- **Intent (Who/What):** {intent}\n"
            f"- **Speaker Role:** {speaker_role}\n"
            f"- **Recording Characterization:** {recording_char_str}\n"
            f"- **Supporting Evidence:** {primary_evidence}\n"
            f"- **Alternative Interpretations:** {alts_str}\n"
            f"- **Missing Evidence:** {missing_str}\n"
            f"- **Uncertainty:** {cs.uncertainty:.2f}\n"
            f"- **Future Projection:** {future_proj}"
        )
        
        situation_understanding = f"{human_explanation}\n\n{situation_details}"

        full_report = templates.THREE_TIER_REPORT_TEMPLATE.format(
            speech_understanding=speech_understanding,
            environmental_understanding=environmental_understanding,
            human_explanation=human_explanation,
            situation=situation,
            context=context,
            intent=intent,
            supporting_evidence=primary_evidence,
            alternatives=alts_str,
            missing_evidence=missing_str,
            uncertainty=cs.uncertainty,
            future_projection=future_proj
        )

        return {
            "speech": speech_understanding,
            "environment": environmental_understanding,
            "situation": situation_understanding,
            "full_report": full_report
        }
