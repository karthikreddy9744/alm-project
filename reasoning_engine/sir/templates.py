"""
Deterministic templates for the Situation Intelligence Renderer.
"""

HUMAN_TEMPLATE = """# SITUATION REPORT
**Current Situation**: {current_situation}
**Confidence**: {confidence:.2f}
**Uncertainty**: {uncertainty:.2f}

## Supporting Evidence
{supporting_evidence}

## Alternative Interpretations
{alternative_interpretations}

## Projected Evolution
**Primary Projected State**: {projected_state}
**Alternative Projected State**: {alt_projected_state}
**Risk Level**: {risk_level}
**Urgency**: {urgency}
**Stability**: {stability}

## Recommended Monitoring
{monitoring_recommendation}
"""

COMPACT_TEMPLATE = "[{urgency}] {current_situation} -> {projected_state} (Conf: {confidence:.2f})"

EMERGENCY_TEMPLATE = """!!! EMERGENCY ALERT !!!
SITUATION: {current_situation}
URGENCY: {urgency} (Risk: {risk_level})
PROJECTION: {projected_state}
CONFIDENCE: {confidence:.2f}

IMMEDIATE RECOMMENDATION: {emergency_recommendation}
"""

DEVELOPER_TEMPLATE = """--- DEVELOPER AUDIT REPORT ---
# 0. System Status
STATUS: Controlled Deployment Ready
ALM v12.0 Cognitive Pipeline

# 1. State Information
Dominant State: {current_situation}
Confidence: {confidence:.2f}
Ambiguity: {uncertainty:.2f}
Active Entities: {active_entities}

# 2. Reasoning Trace
Trace ID: {trace_id}
Hypotheses Considered: {hypotheses_count}
Beliefs Processed: {beliefs_count}

# 3. Projection Details
Projected State: {projected_state}
Alternative Projection: {alt_projected_state}
Risk Level: {risk_level}
Urgency Level: {urgency}
Stability Level: {stability}

# 4. Conflict Resolutions & Contradictions
{contradictions}
------------------------------
"""

HUMAN_COGNITIVE_TEMPLATE = """# HUMAN SITUATION UNDERSTANDING

## Evidence (Perception)
{evidence_list}

## Context Inference
- **Inferred Environment**: {environmental_context}

## Semantic Concepts
- **Active Concepts**: {active_concepts}
- **Missing Concepts**: {missing_concepts}

## Situation Comprehension
- **Primary Hypothesis**: {current_situation}
- **Actors**: {who_is_involved}
- **Rejected Alternatives**: {rejected_alternatives}

## Cognitive Consistency
{internal_contradictions}

## Confidence Breakdown
- **Perception Confidence**: {conf_perception:.2f}
- **Context Confidence**: {conf_context:.2f}
- **Situation Score**: {conf_situation:.2f}
- **Projection Confidence**: {conf_projection:.2f}
- **Overall Certainty**: {conf_overall:.2f}

## Reasoning Chain
{reasoning_chain}

## Expected Future (Projection)
{future_projection}
"""

THREE_TIER_REPORT_TEMPLATE = """╔══════════════════════════════════════════════════════════════════════════════╗
║                  ALM v12.0: THREE INDEPENDENT OUTPUTS                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

### OUTPUT 1 — SPEECH UNDERSTANDING
{speech_understanding}

### OUTPUT 2 — ENVIRONMENTAL UNDERSTANDING
{environmental_understanding}

### OUTPUT 3 — HUMAN-ORIENTED AUDITORY SITUATION UNDERSTANDING

{human_explanation}

**Detailed Breakdown:**
- **Situation:** {situation}
- **Context:** {context}
- **Intent (Who/What):** {intent}
- **Supporting Evidence:** {supporting_evidence}
- **Alternative Interpretations:** {alternatives}
- **Missing Evidence:** {missing_evidence}
- **Uncertainty:** {uncertainty:.2f}
- **Future Projection:** {future_projection}
"""
