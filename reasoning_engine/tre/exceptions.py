"""
Exceptions for Transparent Reasoning Engine (TRE).
"""

class TransparentReasoningError(Exception):
    """Base exception for TRE."""
    pass

class InvalidTraceError(TransparentReasoningError):
    """Raised when an update violates determinism or typing rules."""
    pass
