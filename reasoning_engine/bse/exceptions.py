"""
Exceptions for Belief State Engine.
"""

class BeliefEngineError(Exception):
    """Base exception for BSE."""
    pass

class BeliefLimitExceededError(BeliefEngineError):
    """Raised when MAX_ACTIVE_BELIEFS is exceeded."""
    pass

class InvalidBeliefUpdateError(BeliefEngineError):
    """Raised when an update violates determinism or typing."""
    pass
