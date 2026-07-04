"""
Exceptions for Hypothesis Reasoning Engine (HRE).
"""

class HypothesisEngineError(Exception):
    """Base exception for HRE."""
    pass

class HypothesisLimitExceededError(HypothesisEngineError):
    """Raised when MAX_ACTIVE_HYPOTHESES is exceeded."""
    pass

class InvalidHypothesisUpdateError(HypothesisEngineError):
    """Raised when an update violates determinism or typing rules."""
    pass
