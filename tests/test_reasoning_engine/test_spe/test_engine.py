import unittest
import time

from reasoning_engine.spe.engine import SituationProjectionEngine
from reasoning_engine.spe.models import SituationProjection, RiskLevel, UrgencyLevel, StabilityLevel
from reasoning_engine.wse.models import WorldState
from reasoning_engine.spe import config

class TestSituationProjectionEngine(unittest.TestCase):
    def setUp(self):
        self.spe = SituationProjectionEngine()
        
    def _create_mock_world_state(self, dominant: str, secondaries: list, conf: float, amb: float, entities: list) -> WorldState:
        return WorldState(
            id="ws_1",
            dominant_state=dominant,
            secondary_states=secondaries,
            confidence=conf,
            ambiguity_score=amb,
            active_entities=entities
        )

    def test_projection_generation_escalation(self):
        ws = self._create_mock_world_state(
            dominant="Conversation turning into Argument",
            secondaries=[],
            conf=0.9,
            amb=0.1,
            entities=["People"]
        )
        
        # Rule in config maps Argument -> Possible Distress
        proj = self.spe.project(ws, active_hypotheses=[], active_beliefs=[])
        
        self.assertEqual(proj.projected_state, "Possible Distress")
        self.assertEqual(proj.risk_level, RiskLevel.HIGH) # "distress" triggers HIGH risk
        self.assertEqual(proj.urgency, UrgencyLevel.HIGH)

    def test_projection_stabilization(self):
        ws = self._create_mock_world_state(
            dominant="Traffic Jam",
            secondaries=[],
            conf=0.8,
            amb=0.1,
            entities=["Cars"]
        )
        
        # Rule in config maps Traffic Jam -> Normal Traffic
        proj = self.spe.project(ws, active_hypotheses=[], active_beliefs=[])
        
        self.assertEqual(proj.projected_state, "Normal Traffic")
        self.assertEqual(proj.risk_level, RiskLevel.LOW)
        self.assertEqual(proj.stability, StabilityLevel.STABLE)

    def test_alternative_projections_from_ambiguity(self):
        ws = self._create_mock_world_state(
            dominant="Loud noise, possibly Argument",
            secondaries=["Traffic accident"],
            conf=0.7,
            amb=0.6, # High ambiguity triggers alternative projections
            entities=["Noise"]
        )
        
        proj = self.spe.project(ws, active_hypotheses=[], active_beliefs=[])
        
        self.assertEqual(proj.projected_state, "Possible Distress") # Argument -> Distress
        self.assertIn("Traffic Jam", proj.alternative_projections) # Traffic -> Traffic Jam

    def test_no_hallucination_rule(self):
        ws = self._create_mock_world_state(
            dominant="People eating lunch",
            secondaries=[],
            conf=0.9,
            amb=0.1,
            entities=["People", "Food"]
        )
        
        # There is no rule for "eating lunch". 
        # It should just project "Continued People eating lunch" and NOT make things up.
        proj = self.spe.project(ws, active_hypotheses=[], active_beliefs=[])
        
        self.assertEqual(proj.projected_state, "Continued People eating lunch")
        self.assertEqual(proj.risk_level, RiskLevel.LOW)

    def test_temporal_synchronization(self):
        ws = self._create_mock_world_state("Argument", [], 0.9, 0.1, [])
        self.spe.project(ws, active_hypotheses=[], active_beliefs=[])
        
        self.assertEqual(len(self.spe.history), 0)
        self.spe.project(ws, active_hypotheses=[], active_beliefs=[])
        self.assertEqual(len(self.spe.history), 1)
        
        # Expire history
        old_time = time.time() + config.PROJECTION_EXPIRATION_SECONDS + 10.0
        self.spe.synchronize(current_time=old_time)
        self.assertEqual(len(self.spe.history), 0)

if __name__ == '__main__':
    unittest.main()
