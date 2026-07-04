import unittest
import time

from reasoning_engine.tre.engine import TransparentReasoningEngine
from reasoning_engine.tre import config
from reasoning_engine.bse.models import BeliefObject, BeliefType
from reasoning_engine.hre.models import Hypothesis, HypothesisStatus
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe.models import SituationProjection, RiskLevel, UrgencyLevel, StabilityLevel

class TestTransparentReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.tre = TransparentReasoningEngine()
        
    def _create_mock_objects(self):
        b1 = BeliefObject(id="b_1", statement="Siren sound", belief_type=BeliefType.EVENT, confidence=0.9, supporting_evidence_ids=["e_1"])
        
        h1 = Hypothesis(id="h_1", statement="Emergency", confidence=0.85, plausibility=0.9, uncertainty=0.1)
        h1.status = HypothesisStatus.ACTIVE
        h1.supporting_beliefs = ["b_1"]
        
        h2 = Hypothesis(id="h_2", statement="Drill", confidence=0.3, plausibility=0.4, uncertainty=0.7)
        h2.status = HypothesisStatus.REJECTED
        h2.supporting_beliefs = ["b_1"]

        ws = WorldState(
            id="ws_1",
            dominant_state="Emergency",
            secondary_states=["Drill"],
            confidence=0.85,
            ambiguity_score=0.4,
            active_entities=["Siren"]
        )
        ws.supporting_hypotheses = ["h_1"]
        ws.rejected_hypotheses = ["h_2"]
        
        proj = SituationProjection(
            id="proj_1",
            projected_state="Resolution",
            projection_confidence=0.75,
            uncertainty=0.2,
            supporting_world_state="ws_1"
        )
        
        return [b1], [h1, h2], ws, proj

    def test_trace_generation_completeness(self):
        beliefs, hyps, ws, proj = self._create_mock_objects()
        
        trace = self.tre.trace(
            awm={"e_1": "raw_audio_event"},
            arg={"rel_1": "sound_origin"},
            beliefs=beliefs,
            hypotheses=hyps,
            world_state=ws,
            projection=proj
        )
        
        self.assertIsNotNone(trace.reasoning_id)
        self.assertEqual(len(trace.observations), 1)
        self.assertEqual(len(trace.relationships), 1)
        self.assertEqual(len(trace.beliefs), 1)
        self.assertEqual(len(trace.hypotheses), 2)
        self.assertEqual(trace.world_state, "ws_1")
        self.assertEqual(len(trace.projections), 1)

    def test_confidence_and_uncertainty_explanation(self):
        beliefs, hyps, ws, proj = self._create_mock_objects()
        
        trace = self.tre.trace({}, {}, beliefs, hyps, ws, proj)
        
        # Projection confidence = 0.75 (< 0.8), should be WEAK support
        proj_link = next(link for link in trace.evidence_chain if link.target_id == "proj_1")
        self.assertEqual(proj_link.confidence_explanation, config.EXP_WEAK_SUPPORT)
        
        # WS confidence = 0.85 (>= 0.8), should be STRONG support
        ws_link = next(link for link in trace.evidence_chain if link.target_id == "ws_1")
        self.assertEqual(ws_link.confidence_explanation, config.EXP_STRONG_SUPPORT)

    def test_contradiction_explanation(self):
        beliefs, hyps, ws, proj = self._create_mock_objects()
        
        trace = self.tre.trace({}, {}, beliefs, hyps, ws, proj)
        
        # h_2 was rejected in the WS
        self.assertIn("h_2", trace.contradiction_chain)
        self.assertIn("Emergency", trace.contradiction_chain["h_2"])

    def test_assumption_generation(self):
        beliefs, hyps, ws, proj = self._create_mock_objects()
        
        # Force high ambiguity
        ws.ambiguity_score = 0.9
        
        trace = self.tre.trace({}, {}, beliefs, hyps, ws, proj)
        
        self.assertTrue(any("Assuming primary state despite strong secondary" in a for a in trace.assumptions))

if __name__ == '__main__':
    unittest.main()
