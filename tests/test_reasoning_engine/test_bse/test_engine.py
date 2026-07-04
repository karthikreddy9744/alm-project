import unittest
import time
from typing import Optional

from reasoning_engine.bse.engine import BeliefStateEngine
from reasoning_engine.bse.models import BeliefObject, BeliefType, BeliefStatus
from reasoning_engine.bse import config
from reasoning_engine.bse.exceptions import BeliefLimitExceededError, InvalidBeliefUpdateError

class TestBeliefStateEngine(unittest.TestCase):
    def setUp(self):
        # Deterministically set short timeouts for testing, overriding config safely
        self.original_expiration = config.EXPIRATION_TIME_SECONDS
        self.original_weaken = config.WEAKEN_TIME_SECONDS
        self.original_limit = config.MAX_ACTIVE_BELIEFS
        
        config.EXPIRATION_TIME_SECONDS = 10.0
        config.WEAKEN_TIME_SECONDS = 2.0
        config.MAX_ACTIVE_BELIEFS = 5

        self.bse = BeliefStateEngine()

    def tearDown(self):
        config.EXPIRATION_TIME_SECONDS = self.original_expiration
        config.WEAKEN_TIME_SECONDS = self.original_weaken
        config.MAX_ACTIVE_BELIEFS = self.original_limit

    def test_belief_creation(self):
        belief = self.bse.create_belief(
            belief_id="b_1",
            statement="Speech detected",
            belief_type=BeliefType.SPEECH,
            confidence=0.6
        )
        self.assertEqual(belief.id, "b_1")
        self.assertEqual(belief.confidence, 0.6)
        self.assertEqual(belief.status, BeliefStatus.ACTIVE)
        self.assertIn("b_1", self.bse.beliefs)

    def test_belief_update_deterministic(self):
        self.bse.create_belief("b_1", "Test", BeliefType.ENTITY, confidence=0.5)
        # Update up
        self.bse.update_belief("b_1", 0.1)
        self.assertAlmostEqual(self.bse.beliefs["b_1"].confidence, 0.6)
        # Max out
        self.bse.update_belief("b_1", 10.0)
        self.assertEqual(self.bse.beliefs["b_1"].confidence, 1.0)
        # Drop below threshold
        self.bse.update_belief("b_1", -1.0)
        self.assertEqual(self.bse.beliefs["b_1"].confidence, 0.0)
        self.assertEqual(self.bse.beliefs["b_1"].status, BeliefStatus.INACTIVE)

    def test_belief_reinforcement(self):
        self.bse.create_belief("b_1", "Test", BeliefType.EVENT, confidence=0.5)
        self.bse.reinforce_belief("b_1", "ev_1")
        
        belief = self.bse.beliefs["b_1"]
        self.assertAlmostEqual(belief.confidence, 0.5 + config.REINFORCEMENT_BOOST)
        self.assertIn("ev_1", belief.supporting_evidence_ids)

    def test_belief_weakening_explicit(self):
        self.bse.create_belief("b_1", "Test", BeliefType.EVENT, confidence=0.5)
        self.bse.weaken_belief("b_1", "ev_2")
        
        belief = self.bse.beliefs["b_1"]
        self.assertAlmostEqual(belief.confidence, 0.5 - config.CONTRADICTION_PENALTY)
        self.assertIn("ev_2", belief.contradicting_evidence_ids)

    def test_belief_merge(self):
        self.bse.create_belief("b_1", "Person talking", BeliefType.SPEECH, confidence=0.6, supporting_entities=["e_1"])
        self.bse.create_belief("b_2", "Human voice", BeliefType.SPEECH, confidence=0.7, supporting_events=["ev_1"])
        
        merged = self.bse.merge_beliefs("b_1", "b_2", "A person is actively speaking")
        
        self.assertEqual(merged.id, "b_2")
        self.assertNotIn("b_1", self.bse.beliefs)
        self.assertEqual(merged.statement, "A person is actively speaking")
        self.assertIn("e_1", merged.supporting_entities)
        self.assertIn("ev_1", merged.supporting_events)
        self.assertAlmostEqual(merged.confidence, min(1.0, 0.7 + config.REINFORCEMENT_BOOST))

    def test_belief_conflicts(self):
        self.bse.create_belief("b_1", "Approaching", BeliefType.EVENT, confidence=0.8)
        self.bse.weaken_belief("b_1", "ev_receding")
        
        conflicts = self.bse.get_conflicting_beliefs()
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].id, "b_1")

    def test_memory_cleanup_and_limits(self):
        for i in range(5):
            self.bse.create_belief(f"b_{i}", "Test", BeliefType.UNKNOWN)
            
        with self.assertRaises(BeliefLimitExceededError):
            self.bse.create_belief("b_overflow", "Test", BeliefType.UNKNOWN)

        # Deactivate one
        self.bse.deactivate_belief("b_0")
        # Now creation should trigger garbage collection and succeed
        new_b = self.bse.create_belief("b_overflow", "Test", BeliefType.UNKNOWN)
        self.assertEqual(new_b.id, "b_overflow")
        self.assertNotIn("b_0", self.bse.beliefs)

    def test_temporal_expiration_and_weakening(self):
        self.bse.create_belief("b_1", "Test", BeliefType.EVENT, confidence=0.5)
        # Simulate time moving past weaken time
        current_time = time.time() + config.WEAKEN_TIME_SECONDS + 0.1
        self.bse._decay_beliefs(current_time=current_time)
        
        belief = self.bse.beliefs["b_1"]
        self.assertEqual(belief.status, BeliefStatus.WEAKENING)
        self.assertAlmostEqual(belief.confidence, 0.5 - config.WEAKENING_RATE)

        # Simulate time moving past expiration
        current_time_exp = time.time() + config.EXPIRATION_TIME_SECONDS + 0.1
        self.bse._decay_beliefs(current_time=current_time_exp)
        
        self.assertNotIn("b_1", self.bse.beliefs)

    def test_invalid_updates(self):
        with self.assertRaises(InvalidBeliefUpdateError):
            self.bse.update_belief("non_existent", 0.1)

    def test_snapshot(self):
        self.bse.create_belief("b_1", "Snap", BeliefType.CONTEXT, confidence=0.9)
        snap = self.bse.snapshot()
        self.assertIn("b_1", snap)
        self.assertEqual(snap["b_1"]["statement"], "Snap")
        self.assertEqual(snap["b_1"]["confidence"], 0.9)
        self.assertEqual(snap["b_1"]["status"], "ACTIVE")

if __name__ == '__main__':
    unittest.main()
