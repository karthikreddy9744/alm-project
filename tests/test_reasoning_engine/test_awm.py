import unittest
import time
from reasoning_engine.awm import (
    AuditoryWorldModel, EntityNode, EventNode, RelationshipEdge, 
    NodeState, Trajectory, RelationType, HierarchicalConfidence,
    DuplicateIDError, InvalidReferenceError, StateConsistencyError
)

class TestAuditoryWorldModel(unittest.TestCase):

    def setUp(self):
        self.awm = AuditoryWorldModel()
        self.conf_high = HierarchicalConfidence()
        self.conf_high.speech_recognition = 0.9
        self.conf_low = HierarchicalConfidence()
        self.conf_low.speech_recognition = 0.2
        
    # --- Validations (Exceptions) ---
    
    def test_invalid_confidence(self):
        with self.assertRaises(StateConsistencyError):
            c = HierarchicalConfidence()
            c.speech_recognition = 1.5
            c.__post_init__()
        with self.assertRaises(StateConsistencyError):
            c = HierarchicalConfidence()
            c.speech_recognition = -0.1
            c.__post_init__()
            
    def test_invalid_event_salience(self):
        with self.assertRaises(StateConsistencyError):
            EventNode(id="e1", class_map="Dog", trajectory=Trajectory.STATIC, acoustic_salience=2.0, confidence=self.conf_high)

    # --- Node Management ---

    def test_add_and_duplicate_entity(self):
        ent = EntityNode(id="ent1", entity_type="Human", state=NodeState.STATIONARY, confidence=self.conf_high)
        self.awm.add_entity(ent)
        self.assertIn("ent1", self.awm.entities)
        
        with self.assertRaises(DuplicateIDError):
            self.awm.add_entity(ent)

    def test_update_or_create_entity(self):
        ent = EntityNode(id="ent2", entity_type="Human", state=NodeState.STATIONARY, confidence=self.conf_high)
        self.awm.update_entity(ent)
        self.assertIn("ent2", self.awm.entities)
        
        ent.state = NodeState.MOVING
        self.awm.update_entity(ent)
        self.assertEqual(self.awm.entities["ent2"].state, NodeState.MOVING)

    # --- Relationship and Cascade Deletion ---

    def test_add_relationship_invalid_references(self):
        edge = RelationshipEdge(source_id="ent_none", target_id="evt_none", relation_type=RelationType.GENERATING)
        with self.assertRaises(InvalidReferenceError):
            self.awm.add_relationship(edge)

    def test_cascade_delete_relationships(self):
        # Create Nodes
        ent = EntityNode(id="h1", entity_type="Human", state=NodeState.STATIONARY, confidence=self.conf_high)
        evt = EventNode(id="v1", class_map="Car", trajectory=Trajectory.APPROACHING, acoustic_salience=0.8, confidence=self.conf_high)
        self.awm.add_entity(ent)
        self.awm.add_event(evt)
        
        # Create Edge
        edge = RelationshipEdge(source_id="h1", target_id="v1", relation_type=RelationType.REACTING)
        self.awm.add_relationship(edge)
        self.assertEqual(len(self.awm.relationships), 1)
        
        # Remove Entity -> Edge should be cascade deleted
        self.awm.remove_entity("h1")
        self.assertEqual(len(self.awm.relationships), 0)

    # --- Lifecycle and Expiration ---

    def test_expire_old_objects(self):
        current_time = 1000.0
        
        ent1 = EntityNode(id="ent1", entity_type="Human", state=NodeState.STATIONARY, confidence=self.conf_high, last_updated_timestamp=current_time - 10)
        ent2 = EntityNode(id="ent2", entity_type="Dog", state=NodeState.MOVING, confidence=self.conf_high, last_updated_timestamp=current_time - 2)
        
        self.awm.add_entity(ent1)
        self.awm.add_entity(ent2)
        
        # Add edge between them (updated recently)
        edge = RelationshipEdge(source_id="ent1", target_id="ent2", relation_type=RelationType.CO_OCCURRING, last_updated_timestamp=current_time)
        self.awm.add_relationship(edge)
        
        # Expire objects older than 5 seconds
        self.awm.expire_old_objects(current_time=current_time, timeout_s=5.0)
        
        # ent1 should be deleted, ent2 remains
        self.assertNotIn("ent1", self.awm.entities)
        self.assertIn("ent2", self.awm.entities)
        
        # Edge should be cascade deleted because ent1 was removed
        self.assertEqual(len(self.awm.relationships), 0)

    # --- Utility ---
    
    def test_snapshot_and_reset(self):
        ent = EntityNode(id="ent1", entity_type="Human", state=NodeState.STATIONARY, confidence=self.conf_high)
        self.awm.add_entity(ent)
        
        snap = self.awm.snapshot()
        self.assertIn("ent1", snap.entities)
        
        self.awm.reset()
        self.assertEqual(len(self.awm.entities), 0)
        # Snapshot should remain unaffected
        self.assertIn("ent1", snap.entities)

if __name__ == '__main__':
    unittest.main()
