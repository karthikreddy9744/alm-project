"""
Models for the Situation Intelligence Renderer (SIR).
"""
from enum import Enum, auto

class RenderMode(Enum):
    HUMAN = auto()
    COMPACT = auto()
    EMERGENCY = auto()
    DEVELOPER = auto()
    JSON = auto()
    API = auto()
