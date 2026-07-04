import unittest
import time

from reasoning_engine.hre.engine import HypothesisReasoningEngine
from reasoning_engine.hre.models import Hypothesis, HypothesisStatus
from reasoning_engine.hre import config
from reasoning_engine.hre.exceptions import HypothesisLimitExceededError, InvalidHypothesisUpdateError

class TestHypothesisReasoningEngine(unittest.TestCase):
    def setUp(self):
        # Deterministically set short timeouts for testing
        self.original_expiration = config.EXPIRATION_TIME_SECONDS
        self.original_limit = config.MAX_ACTIVE_HYPOTHESES
        self.original_threshold = config.REJECTION_THRESHOLD
        
        config.EXPIRATION_TIME_SECONDS = 10.0
        config.MAX_ACTIVE_HYPOTHESES = 5
        config.REJECTION_THRESHOLD = 0.2

        self.hre = HypothesisReasoningEngine()

    def tearDown(self):
        config.EXPIRATION_TIME_SECONDS = self.original_expiration
        config.MAX_ACTIVE_HYPOTHESES = self.original_limit
        config.REJECTION_THRESHOLD = self.original_threshold

    def test_hypothesis_generation(self):
        hyp = self.hre.generate(
            hypothesis_id="h_1",
            statement="Test scenario",
            supporting_beliefs=["b_1"],
            confidence=0.8,
            plausibility=0.7
        )
        self.assertEqual(hyp.id, "h_1")
        self.assertEqual(hyp.status, HypothesisStatus.ACTIVE)
        self.assertIn("h_1", self.hre.hypotheses)

    def test_hypothesis_evaluation_and_update(self):
        self.hre.generate("h_1", "Test", confidence=0.5, uncertainty=0.5, plausibility=0.5)
        # Update metrics
        self.hre.evaluate("h_1", new_confidence=0.9, new_uncertainty=0.1, new_plausibility=0.8)
        
        hyp = self.hre.hypotheses["h_1"]
        self.assertAlmostEqual(hyp.confidence, 0.9)
        self.assertAlmostEqual(hyp.uncertainty, 0.1)
        self.assertAlmostEqual(hyp.plausibility, 0.8)

    def test_hypothesis_rejection_threshold(self):
        self.hre.generate("h_1", "Test", confidence=0.5, plausibility=0.5)
        # Drop confidence below threshold
        self.hre.update("h_1", confidence=0.1, uncertainty=0.5, plausibility=0.5)
        
        hyp = self.hre.hypotheses["h_1"]
        self.assertEqual(hyp.status, HypothesisStatus.REJECTED)

    def test_hypothesis_merge(self):
        self.hre.generate("h_1", "Emergency", supporting_beliefs=["b_1"], confidence=0.6, plausibility=0.5)
        self.hre.generate("h_2", "Distress", supporting_beliefs=["b_2"], confidence=0.7, plausibility=0.6)
        
        merged = self.hre.merge("h_1", "h_2", "Emergency Distress")
        
        self.assertEqual(merged.id, "h_2")
        self.assertNotIn("h_1", self.hre.hypotheses)
        self.assertIn("b_1", merged.supporting_beliefs)
        self.assertIn("b_2", merged.supporting_beliefs)
        # Target conf (0.7) merged with source conf (0.6) + boost -> 0.7 + boost
        self.assertAlmostEqual(merged.confidence, min(1.0, 0.7 + config.REINFORCEMENT_BOOST))
        self.assertAlmostEqual(merged.plausibility, 0.6) # max(0.5, 0.6)

    def test_competing_hypotheses_ranking(self):
        self.hre.generate("h_drill", "Training Drill", confidence=0.4, plausibility=0.8, uncertainty=0.3)
        self.hre.generate("h_emergency", "Real Emergency", confidence=0.9, plausibility=0.9, uncertainty=0.1)
        self.hre.generate("h_ignore", "False Alarm", confidence=0.1, plausibility=0.1, uncertainty=0.9) # Will be rejected
        
        ranked = self.hre.rank()
        
        # h_ignore should be rejected and not in ranking
        self.assertEqual(len(ranked), 2)
        # h_emergency should rank higher than h_drill
        self.assertEqual(ranked[0].id, "h_emergency")
        self.assertEqual(ranked[1].id, "h_drill")

    def test_memory_limits_and_cleanup(self):
        for i in range(5):
            self.hre.generate(f"h_{i}", "Test", confidence=0.5, plausibility=0.5)
            
        with self.assertRaises(HypothesisLimitExceededError):
            self.hre.generate("h_overflow", "Test")

        # Reject one to allow GC
        self.hre.reject("h_0")
        
        # Now creation should succeed after GC
        self.hre.generate("h_overflow", "Test", confidence=0.5, plausibility=0.5)
        self.assertIn("h_overflow", self.hre.hypotheses)
        self.assertNotIn("h_0", self.hre.hypotheses)

    def test_temporal_expiration(self):
        self.hre.generate("h_1", "Test", confidence=0.5, plausibility=0.5)
        
        # Simulate time moving past expiration
        current_time = time.time() + config.EXPIRATION_TIME_SECONDS + 0.1
        self.hre.synchronize(current_time=current_time)
        
        # Should be rejected and garbage collected
        self.assertNotIn("h_1", self.hre.hypotheses)

if __name__ == '__main__':
    unittest.main()
