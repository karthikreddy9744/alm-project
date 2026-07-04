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
ALM v10.1 CASRE Cognitive Pipeline

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
