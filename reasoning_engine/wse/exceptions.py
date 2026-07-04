"""
Exceptions for World State Estimation Engine (WSE).
"""

class WorldStateEngineError(Exception):
    """Base exception for WSE."""
    pass

class InvalidWorldStateUpdateError(WorldStateEngineError):
    """Raised when an update violates determinism or typing rules."""
    pass
