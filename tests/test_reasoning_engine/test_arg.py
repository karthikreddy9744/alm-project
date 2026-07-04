import unittest
from reasoning_engine.arg import (
    AcousticRelationshipGraph, ARGNode, ARGEdge, NodeType, EdgeType,
    ARGError, NodeNotFoundError, DuplicateNodeError, DuplicateEdgeError
)
from reasoning_engine.awm.models import HierarchicalConfidence

class TestAcousticRelationshipGraph(unittest.TestCase):

    def setUp(self):
        self.arg = AcousticRelationshipGraph()
        self.conf = HierarchicalConfidence()
        self.conf.semantic_embeddings = 0.9

    # --- Validation / Exception Tests ---

    def test_invalid_node_confidence(self):
        from reasoning_engine.awm.exceptions import StateConsistencyError
        with self.assertRaises(StateConsistencyError):
            c = HierarchicalConfidence()
            c.semantic_embeddings = 1.5
            c.__post_init__()

    def test_invalid_edge_references(self):
        edge = ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.APPROACHES, confidence=self.conf)
        with self.assertRaises(NodeNotFoundError):
            self.arg.add_edge(edge)

    def test_duplicate_node(self):
        n1 = ARGNode(id="n1", node_type=NodeType.PERSON, confidence=self.conf)
        self.arg.add_node(n1)
        with self.assertRaises(DuplicateNodeError):
            self.arg.add_node(n1)

    def test_duplicate_edge(self):
        n1 = ARGNode(id="n1", node_type=NodeType.PERSON, confidence=self.conf)
        n2 = ARGNode(id="n2", node_type=NodeType.VEHICLE, confidence=self.conf)
        self.arg.add_node(n1)
        self.arg.add_node(n2)
        
        e1 = ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.APPROACHES, confidence=self.conf)
        self.arg.add_edge(e1)
        # Note: We made ARGEdge duplicates reinforce the edge, so they don't raise DuplicateEdgeError anymore.
        # But wait! We'll just verify the strength increases instead.
        self.arg.add_edge(e1)
        edges = self.arg.outgoing_edges("n1")
        self.assertAlmostEqual(edges[0].strength, 0.6)

    # --- Graph Structural Tests ---

    def test_cascade_delete(self):
        n1 = ARGNode(id="n1", node_type=NodeType.PERSON, confidence=self.conf)
        n2 = ARGNode(id="n2", node_type=NodeType.VEHICLE, confidence=self.conf)
        self.arg.add_node(n1)
        self.arg.add_node(n2)
        self.arg.add_edge(ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.APPROACHES, confidence=self.conf))
        
        self.assertEqual(self.arg.node_count, 2)
        self.assertEqual(self.arg.edge_count, 1)
        
        self.arg.remove_node("n1")
        self.assertEqual(self.arg.node_count, 1)
        self.assertEqual(self.arg.edge_count, 0)
        self.assertEqual(len(self.arg.incoming_edges("n2")), 0)

    # --- Traversal & Query Tests ---

    def test_neighbors_and_edges(self):
        n1 = ARGNode(id="n1", node_type=NodeType.PERSON, confidence=self.conf)
        n2 = ARGNode(id="n2", node_type=NodeType.VEHICLE, confidence=self.conf)
        self.arg.add_node(n1)
        self.arg.add_node(n2)
        self.arg.add_edge(ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.APPROACHES, confidence=self.conf))
        
        self.assertIn("n2", self.arg.neighbors("n1"))
        self.assertIn("n1", self.arg.neighbors("n2"))
        
        out_e = self.arg.outgoing_edges("n1")
        self.assertEqual(len(out_e), 1)
        self.assertEqual(out_e[0].target_id, "n2")
        
        in_e = self.arg.incoming_edges("n2")
        self.assertEqual(len(in_e), 1)
        self.assertEqual(in_e[0].source_id, "n1")

    # --- Algorithm Tests ---

    def test_connected_components(self):
        for i in range(1, 5):
            self.arg.add_node(ARGNode(id=f"n{i}", node_type=NodeType.OBJECT, confidence=self.conf))
            
        # Group A: n1 - n2
        self.arg.add_edge(ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.CAUSES, confidence=self.conf))
        # Group B: n3 - n4
        self.arg.add_edge(ARGEdge(source_id="n3", target_id="n4", edge_type=EdgeType.CAUSES, confidence=self.conf))
        
        components = self.arg.connected_components()
        self.assertEqual(len(components), 2)
        self.assertIn({"n1", "n2"}, components)
        self.assertIn({"n3", "n4"}, components)

    def test_shortest_path(self):
        for i in range(1, 6):
            self.arg.add_node(ARGNode(id=f"n{i}", node_type=NodeType.OBJECT, confidence=self.conf))
            
        # Path: n1 -> n2 -> n3
        self.arg.add_edge(ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.CAUSES, confidence=self.conf))
        self.arg.add_edge(ARGEdge(source_id="n2", target_id="n3", edge_type=EdgeType.CAUSES, confidence=self.conf))
        
        # Dead end
        self.arg.add_edge(ARGEdge(source_id="n1", target_id="n4", edge_type=EdgeType.CAUSES, confidence=self.conf))
        
        # Unconnected
        # n5 is isolated
        
        path = self.arg.shortest_path("n1", "n3")
        self.assertEqual(path, ["n1", "n2", "n3"])
        
        # Undirected traversal supported by shortest path
        path_reverse = self.arg.shortest_path("n3", "n1")
        self.assertEqual(path_reverse, ["n3", "n2", "n1"])
        
        # No path
        path_none = self.arg.shortest_path("n1", "n5")
        self.assertEqual(path_none, [])

    def test_subgraph(self):
        for i in range(1, 4):
            self.arg.add_node(ARGNode(id=f"n{i}", node_type=NodeType.OBJECT, confidence=self.conf))
        self.arg.add_edge(ARGEdge(source_id="n1", target_id="n2", edge_type=EdgeType.CAUSES, confidence=self.conf))
        self.arg.add_edge(ARGEdge(source_id="n2", target_id="n3", edge_type=EdgeType.CAUSES, confidence=self.conf))
        
        sub = self.arg.subgraph(["n1", "n2"])
        self.assertEqual(sub.node_count, 2)
        self.assertEqual(sub.edge_count, 1)

if __name__ == '__main__':
    unittest.main()
