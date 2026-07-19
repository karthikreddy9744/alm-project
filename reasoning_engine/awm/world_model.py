import time
import copy
from typing import Dict, List, Set, Optional

from reasoning_engine.awm.models import (
    EntityNode, EventNode, RelationshipEdge, BeliefObject, 
    WorldStateObject, ProjectionObject
)
from reasoning_engine.fusion.models import RecordingCharacterization

from reasoning_engine.awm.exceptions import (
    DuplicateIDError, InvalidReferenceError
)

class AuditoryWorldModel:
    """
    The persistent state repository for the CASRE engine.
    It manages entities, events, relationships, beliefs, and high-level states.
    It enforces referential integrity (e.g. cascading deletes for edges).
    """
    def __init__(self):
        # Base Graphs
        self.entities: Dict[str, EntityNode] = {}
        self.events: Dict[str, EventNode] = {}
        # List of edges
        self.relationships: List[RelationshipEdge] = []
        
        # High-level state
        self.beliefs: List[BeliefObject] = []
        self.current_world_state: Optional[WorldStateObject] = None
        self.projection: Optional[ProjectionObject] = None
        
        # Semantic Concepts (from CLAP)
        self.clap_concepts: List[str] = []
        
        # Neural Perception Characterization
        self.recording_characterization: Optional[RecordingCharacterization] = None


    def get_node(self, node_id: str):
        if node_id in self.entities:
            return self.entities[node_id]
        if node_id in self.events:
            return self.events[node_id]
        return None

    # -------------------------------------------------------------------------
    # Entity Management
    # -------------------------------------------------------------------------
    
    def add_entity(self, entity: EntityNode):
        if entity.id in self.entities:
            raise DuplicateIDError(f"Entity with ID '{entity.id}' already exists.")
        self.entities[entity.id] = entity

    def update_entity(self, entity: EntityNode):
        if entity.id not in self.entities:
            # According to rules: If it exists, update. Otherwise, create it.
            self.add_entity(entity)
        else:
            self.entities[entity.id] = entity

    def remove_entity(self, entity_id: str):
        if entity_id in self.entities:
            del self.entities[entity_id]
            self._cascade_delete_relationships(entity_id)

    # -------------------------------------------------------------------------
    # Event Management
    # -------------------------------------------------------------------------

    def add_event(self, event: EventNode):
        if event.id in self.events:
            raise DuplicateIDError(f"Event with ID '{event.id}' already exists.")
        self.events[event.id] = event

    def update_event(self, event: EventNode):
        if event.id not in self.events:
            self.add_event(event)
        else:
            self.events[event.id] = event

    def remove_event(self, event_id: str):
        if event_id in self.events:
            del self.events[event_id]
            self._cascade_delete_relationships(event_id)

    # -------------------------------------------------------------------------
    # Relationship Management
    # -------------------------------------------------------------------------

    def add_relationship(self, edge: RelationshipEdge):
        # Validate that source and target exist
        if edge.source_id not in self.entities and edge.source_id not in self.events:
            raise InvalidReferenceError(f"Source ID '{edge.source_id}' does not exist.")
        if edge.target_id not in self.entities and edge.target_id not in self.events:
            raise InvalidReferenceError(f"Target ID '{edge.target_id}' does not exist.")
            
        # Prevent duplicates
        for existing in self.relationships:
            if existing.source_id == edge.source_id and \
               existing.target_id == edge.target_id and \
               existing.relation_type == edge.relation_type:
                # Just update the timestamp if it already exists
                existing.last_updated_timestamp = edge.last_updated_timestamp
                return
                
        self.relationships.append(edge)

    def remove_relationship(self, source_id: str, target_id: str):
        self.relationships = [
            edge for edge in self.relationships
            if not (edge.source_id == source_id and edge.target_id == target_id)
        ]

    def _cascade_delete_relationships(self, node_id: str):
        self.relationships = [
            edge for edge in self.relationships
            if edge.source_id != node_id and edge.target_id != node_id
        ]

    # -------------------------------------------------------------------------
    # High-level States
    # -------------------------------------------------------------------------

    def update_beliefs(self, beliefs: List[BeliefObject]):
        # Validate that all beliefs point to valid nodes
        for b in beliefs:
            if b.node_id not in self.entities and b.node_id not in self.events:
                raise InvalidReferenceError(f"Belief references non-existent node ID '{b.node_id}'.")
        self.beliefs = beliefs

    def update_world_state(self, state: WorldStateObject):
        for node_id in state.active_nodes:
            if node_id not in self.entities and node_id not in self.events:
                raise InvalidReferenceError(f"World state references non-existent node ID '{node_id}'.")
        self.current_world_state = state

    def update_projection(self, projection: ProjectionObject):
        self.projection = projection

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def expire_old_objects(self, current_time: float, timeout_s: float):
        """Removes objects that haven't been updated within timeout_s."""
        cutoff_time = current_time - timeout_s
        
        # Gather expired node IDs
        expired_entities = [e.id for e in self.entities.values() if e.last_updated_timestamp < cutoff_time]
        expired_events = [e.id for e in self.events.values() if e.last_updated_timestamp < cutoff_time]
        
        for eid in expired_entities:
            self.remove_entity(eid)
            
        for eid in expired_events:
            self.remove_event(eid)
            
        # Edges will auto-cascade, but check explicitly for standalone edge timeout
        self.relationships = [r for r in self.relationships if r.last_updated_timestamp >= cutoff_time]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def snapshot(self) -> 'AuditoryWorldModel':
        """Creates a deep copy of the current state."""
        return copy.deepcopy(self)

    def reset(self):
        """Completely clears the world state."""
        self.entities.clear()
        self.events.clear()
        self.relationships.clear()
        self.beliefs.clear()
        self.current_world_state = None
        self.projection = None
