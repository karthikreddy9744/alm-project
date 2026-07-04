"""
Custom exceptions for the Auditory World Model (AWM).
"""

class AWMError(Exception):
    """Base exception for all AWM-related errors."""
    pass

class InvalidReferenceError(AWMError):
    """Raised when an operation references a node or edge that does not exist."""
    pass

class DuplicateIDError(AWMError):
    """Raised when attempting to add an entity or event with an ID that already exists."""
    pass

class StateConsistencyError(AWMError):
    """Raised when an object violates internal validity invariants (e.g., probability out of bounds)."""
    pass
