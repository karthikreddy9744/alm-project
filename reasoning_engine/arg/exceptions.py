"""
Custom exceptions for the Acoustic Relationship Graph (ARG).
"""

class ARGError(Exception):
    """Base exception for all ARG-related errors."""
    pass

class NodeNotFoundError(ARGError):
    """Raised when an operation references a node ID that does not exist in the graph."""
    pass

class DuplicateNodeError(ARGError):
    """Raised when attempting to add a node with an ID that already exists."""
    pass

class DuplicateEdgeError(ARGError):
    """Raised when attempting to add an edge that already exists exactly between source and target with the same type."""
    pass

class InvalidEdgeTypeError(ARGError):
    """Raised when an invalid edge type is used."""
    pass
