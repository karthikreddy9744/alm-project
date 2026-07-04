import unittest
import time
from typing import List

from reasoning_engine.wse.engine import WorldStateEngine
from reasoning_engine.wse import config
from reasoning_engine.hre.models import Hypothesis, HypothesisStatus

class TestWorldStateEngine(unittest.TestCase):
    def setUp(self):
        self.wse = WorldStateEngine()
        
    def _create_mock_hypothesis(self, id: str, statement: str, conf: float, plaus: float, unc: float = 0.1) -> Hypothesis:
        hyp = Hypothesis(
            id=id,
            statement=statement,
            confidence=conf,
            plausibility=plaus,
            uncertainty=unc
        )
        hyp.status = HypothesisStatus.ACTIVE
        return hyp

    def test_state_estimation_single_hypothesis(self):
        h1 = self._create_mock_hypothesis("h_1", "Emergency", 0.9, 0.9)
        state = self.wse.estimate([h1], active_beliefs=[], awm={}, arg={})
        
        self.assertEqual(state.dominant_state, "Emergency")
        self.assertEqual(state.ambiguity_score, 0.0) # No secondary hypotheses
        self.assertAlmostEqual(state.confidence, 0.9)

    def test_hypothesis_conflict_ambiguity(self):
        h1 = self._create_mock_hypothesis("h_1", "Emergency", 0.7, 0.7)
        h2 = self._create_mock_hypothesis("h_2", "Drill", 0.65, 0.65) # Very close! High conflict
        
        state = self.wse.estimate([h1, h2], active_beliefs=[], awm={}, arg={})
        
        self.assertEqual(state.dominant_state, "Emergency")
        self.assertIn("Drill", state.secondary_states)
        self.assertTrue(state.ambiguity_score > 0.0) # Ambiguity should be high

    def test_state_transition_smoothing(self):
        # Initial dominant state
        h_initial = self._create_mock_hypothesis("h_norm", "Normal", 0.8, 0.8)
        self.wse.estimate([h_initial], active_beliefs=[], awm={}, arg={})
        self.assertEqual(self.wse.current_state.dominant_state, "Normal")
        
        # New hypothesis barely beats the old one (difference < config.STATE_TRANSITION_MOMENTUM)
        # Momentum is 0.15 by default.
        # Current confidence = 0.8. New needs to be >= 0.95 to break it instantly.
        h_new = self._create_mock_hypothesis("h_distress", "Distress", 0.85, 0.85)
        
        state2 = self.wse.estimate([h_new], active_beliefs=[], awm={}, arg={})
        # Because of smoothing, dominant state stays "Normal"
        self.assertEqual(state2.dominant_state, "Normal")
        self.assertTrue(state2.confidence < 0.8) # Confidence weakened
        
        # Now a very strong hypothesis appears
        h_strong = self._create_mock_hypothesis("h_emergency", "Emergency", 0.99, 0.99)
        state3 = self.wse.estimate([h_strong], active_beliefs=[], awm={}, arg={})
        self.assertEqual(state3.dominant_state, "Emergency") # Transition succeeds

    def test_consistency_computation(self):
        h1 = self._create_mock_hypothesis("h_1", "Test", 0.8, 0.8)
        h1.supporting_beliefs = ["b_3"]
        h1.contradicting_beliefs = ["b_1", "b_2"]
        consistency = self.wse.compute_consistency(h1, active_beliefs=["b_1", "b_2", "b_3"])
        
        self.assertAlmostEqual(consistency, 0.8) # 1.0 - 0.2 penalty

    def test_empty_estimation(self):
        state = self.wse.estimate([], active_beliefs=[], awm={}, arg={})
        self.assertEqual(state.dominant_state, "Unknown situation")

    def test_temporal_synchronization(self):
        h1 = self._create_mock_hypothesis("h_1", "Test", 0.8, 0.8)
        self.wse.estimate([h1], active_beliefs=[], awm={}, arg={})
        
        self.assertEqual(len(self.wse.history), 0) # only current state exists
        
        self.wse.estimate([h1], active_beliefs=[], awm={}, arg={})
        self.assertEqual(len(self.wse.history), 1)
        
        # Fake expiration
        old_time = time.time() + config.STATE_EXPIRATION_SECONDS + 10.0
        self.wse.synchronize(current_time=old_time)
        self.assertEqual(len(self.wse.history), 0)

if __name__ == '__main__':
    unittest.main()
