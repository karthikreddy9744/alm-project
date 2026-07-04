import unittest
import json

from reasoning_engine.sir.engine import SituationIntelligenceRenderer
from reasoning_engine.sir.models import RenderMode
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe.models import SituationProjection, RiskLevel, UrgencyLevel, StabilityLevel
from reasoning_engine.tre.models import TransparentReasoningTrace, ExplanationLink
from reasoning_engine.hre.models import Hypothesis
from reasoning_engine.bse.models import BeliefObject, BeliefType

class TestSituationIntelligenceRenderer(unittest.TestCase):
    def setUp(self):
        self.sir = SituationIntelligenceRenderer()
        
        self.ws = WorldState(
            id="ws_01",
            dominant_state="Argument",
            secondary_states=["Discussion"],
            confidence=0.8,
            ambiguity_score=0.2,
            active_entities=["People"]
        )
        
        self.proj = SituationProjection(
            id="proj_01",
            projected_state="Possible Distress",
            projection_confidence=0.7,
            uncertainty=0.3,
            risk_level=RiskLevel.HIGH,
            urgency=UrgencyLevel.HIGH,
            stability=StabilityLevel.DEGRADING,
            supporting_world_state="ws_01"
        )
        
        self.trace = TransparentReasoningTrace(reasoning_id="trace_01")
        self.trace.evidence_chain.append(
            ExplanationLink(["h_1"], "ws_01", "Derived from top hypotheses", "Strong", "Low")
        )
        
        self.hyps = []
        self.beliefs = []

    def test_human_render(self):
        result = self.sir.export(RenderMode.HUMAN, self.ws, self.proj, self.trace, self.hyps, self.beliefs)
        self.assertIn("SITUATION REPORT", result)
        self.assertIn("Argument", result)
        self.assertIn("Possible Distress", result)
        self.assertIn("HIGH", result)
        self.assertIn("Derived from top hypotheses", result)
        
    def test_compact_render(self):
        result = self.sir.export(RenderMode.COMPACT, self.ws, self.proj, self.trace, self.hyps, self.beliefs)
        self.assertIn("[HIGH] Argument -> Possible Distress (Conf: 0.80)", result)
        
    def test_emergency_render(self):
        result = self.sir.export(RenderMode.EMERGENCY, self.ws, self.proj, self.trace, self.hyps, self.beliefs)
        self.assertIn("!!! EMERGENCY ALERT !!!", result)
        self.assertIn("Prepare for escalation", result)

    def test_json_validity(self):
        result = self.sir.export(RenderMode.JSON, self.ws, self.proj, self.trace, self.hyps, self.beliefs)
        # Should not raise exception
        data = json.loads(result)
        
        self.assertEqual(data["world_state"]["dominant_state"], "Argument")
        self.assertEqual(data["projection"]["risk_level"], "HIGH")

    def test_api_validity(self):
        result = self.sir.export(RenderMode.API, self.ws, self.proj, self.trace, self.hyps, self.beliefs)
        data = json.loads(result)
        
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["code"], 200)
        self.assertEqual(data["data"]["world_state"]["dominant_state"], "Argument")

    def test_developer_render(self):
        self.trace.contradiction_chain["h_2"] = ["Argument"]
        result = self.sir.export(RenderMode.DEVELOPER, self.ws, self.proj, self.trace, self.hyps, self.beliefs)
        
        self.assertIn("DEVELOPER AUDIT REPORT", result)
        self.assertIn("h_2 overridden by ['Argument']", result)

if __name__ == '__main__':
    unittest.main()
