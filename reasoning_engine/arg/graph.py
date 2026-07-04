import copy
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

from reasoning_engine.arg.models import ARGNode, ARGEdge, EdgeType
from reasoning_engine.arg.exceptions import (
    NodeNotFoundError, DuplicateNodeError, DuplicateEdgeError, ARGError
)

class AcousticRelationshipGraph:
    """
    Deterministic, memory-efficient graph representation of auditory relationships.
    Uses native Python dicts for O(1) node and edge lookups.
    """
    def __init__(self):
        # Node storage: node_id -> ARGNode
        self._nodes: Dict[str, ARGNode] = {}
        
        # Adjacency lists for fast traversals
        # O(1) access to edges leaving or entering a node
        self._out_edges: Dict[str, Dict[Tuple[str, EdgeType], ARGEdge]] = defaultdict(dict)
        self._in_edges: Dict[str, Dict[Tuple[str, EdgeType], ARGEdge]] = defaultdict(dict)

    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------

    def add_node(self, node: ARGNode):
        if node.id in self._nodes:
            raise DuplicateNodeError(f"Node '{node.id}' already exists in ARG.")
        self._nodes[node.id] = node

    def update_node(self, node: ARGNode):
        """Updates an existing node or adds it if it doesn't exist."""
        self._nodes[node.id] = node

    def remove_node(self, node_id: str):
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
            
        # Safely remove all dependent outgoing edges
        out_targets = list(self._out_edges[node_id].keys())
        for target_id, edge_type in out_targets:
            self.remove_edge(node_id, target_id, edge_type)
            
        # Safely remove all dependent incoming edges
        in_sources = list(self._in_edges[node_id].keys())
        for source_id, edge_type in in_sources:
            self.remove_edge(source_id, node_id, edge_type)
            
        # Remove node
        del self._nodes[node_id]
        
        # Cleanup defaultdict memory
        if node_id in self._out_edges:
            del self._out_edges[node_id]
        if node_id in self._in_edges:
            del self._in_edges[node_id]

    def get_node(self, node_id: str) -> Optional[ARGNode]:
        return self._nodes.get(node_id)

    # -------------------------------------------------------------------------
    # Edge Management
    # -------------------------------------------------------------------------

    def add_edge(self, edge: ARGEdge):
        if edge.source_id not in self._nodes:
            raise NodeNotFoundError(f"Source node '{edge.source_id}' does not exist.")
        if edge.target_id not in self._nodes:
            raise NodeNotFoundError(f"Target node '{edge.target_id}' does not exist.")
            
        edge_key = (edge.target_id, edge.edge_type)
        if edge_key in self._out_edges[edge.source_id]:
            # Instead of failing, dynamically reinforce the relationship
            existing = self._out_edges[edge.source_id][edge_key]
            existing.evidence_count += 1
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.duration = edge.timestamp - existing.timestamp
            existing.last_update = edge.timestamp
            existing.confidence.fusion = min(1.0, max(existing.confidence.fusion, edge.confidence.fusion) + 0.05)
            return
            
        self._out_edges[edge.source_id][edge_key] = edge
        self._in_edges[edge.target_id][(edge.source_id, edge.edge_type)] = edge

    def update_edge(self, edge: ARGEdge):
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise NodeNotFoundError("Cannot update edge: referenced nodes do not exist.")
            
        edge_key = (edge.target_id, edge.edge_type)
        self._out_edges[edge.source_id][edge_key] = edge
        self._in_edges[edge.target_id][(edge.source_id, edge.edge_type)] = edge

    def remove_edge(self, source_id: str, target_id: str, edge_type: EdgeType):
        try:
            del self._out_edges[source_id][(target_id, edge_type)]
            del self._in_edges[target_id][(source_id, edge_type)]
        except KeyError:
            pass  # Safely ignore if it doesn't exist

    # -------------------------------------------------------------------------
    # Graph Queries & Traversals
    # -------------------------------------------------------------------------

    def neighbors(self, node_id: str) -> List[str]:
        """Returns all node IDs connected via either incoming or outgoing edges."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
        
        outgoing = [target_id for (target_id, _) in self._out_edges[node_id].keys()]
        incoming = [source_id for (source_id, _) in self._in_edges[node_id].keys()]
        return list(set(outgoing + incoming))

    def outgoing_edges(self, node_id: str) -> List[ARGEdge]:
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
        return list(self._out_edges[node_id].values())

    def incoming_edges(self, node_id: str) -> List[ARGEdge]:
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
        return list(self._in_edges[node_id].values())

    # -------------------------------------------------------------------------
    # Algorithms
    # -------------------------------------------------------------------------

    def connected_components(self) -> List[Set[str]]:
        """Returns a list of sets, where each set contains node IDs of a connected component (undirected)."""
        visited = set()
        components = []
        
        for node_id in self._nodes:
            if node_id not in visited:
                # BFS to find all connected nodes
                component = set()
                queue = deque([node_id])
                visited.add(node_id)
                
                while queue:
                    curr = queue.popleft()
                    component.add(curr)
                    for neighbor in self.neighbors(curr):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
                
        return components

    def shortest_path(self, start_id: str, end_id: str) -> List[str]:
        """Returns shortest path (undirected) between two nodes using BFS. Empty if no path."""
        if start_id not in self._nodes or end_id not in self._nodes:
            raise NodeNotFoundError("Start or end node not found.")
            
        if start_id == end_id:
            return [start_id]
            
        queue = deque([[start_id]])
        visited = {start_id}
        
        while queue:
            path = queue.popleft()
            curr = path[-1]
            
            for neighbor in self.neighbors(curr):
                if neighbor == end_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
                    
        return []

    def subgraph(self, node_ids: List[str]) -> 'AcousticRelationshipGraph':
        """Creates a new ARG containing only the specified nodes and their internal edges."""
        new_graph = AcousticRelationshipGraph()
        
        # Add nodes
        for nid in node_ids:
            if nid in self._nodes:
                new_graph.add_node(copy.deepcopy(self._nodes[nid]))
                
        # Add internal edges
        valid_ids = set(node_ids)
        for nid in valid_ids:
            if nid in self._out_edges:
                for (target_id, edge_type), edge in self._out_edges[nid].items():
                    if target_id in valid_ids:
                        new_graph.add_edge(copy.deepcopy(edge))
                        
        return new_graph

    # -------------------------------------------------------------------------
    # Lifecycle Operations
    # -------------------------------------------------------------------------

    def clear(self):
        """Empties the graph completely."""
        self._nodes.clear()
        self._out_edges.clear()
        self._in_edges.clear()

    def snapshot(self) -> 'AcousticRelationshipGraph':
        """Returns a complete, deep-copied instance of the graph."""
        return copy.deepcopy(self)

    # -------------------------------------------------------------------------
    # Properties / Stats
    # -------------------------------------------------------------------------
    
    @property
    def node_count(self) -> int:
        return len(self._nodes)
        
    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._out_edges.values())
