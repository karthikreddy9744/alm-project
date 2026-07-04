"""
Exceptions for Situation Projection Engine (SPE).
"""

class SituationProjectionError(Exception):
    """Base exception for SPE."""
    pass

class InvalidProjectionError(SituationProjectionError):
    """Raised when an update violates determinism or typing rules."""
    pass
