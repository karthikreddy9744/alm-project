from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any

class SpeechInfo(BaseModel):
    transcript: str
    language: str
    confidence: float
    speaker_count: int
    timestamps: Optional[List[Tuple[float, float]]] = None
    keywords: Optional[List[str]] = None
    duration: Optional[float] = None

class EventInfo(BaseModel):
    event_label: str
    salience: float
    confidence: float
    is_foreground: bool
    is_background: bool

class AudioEvidenceObject(BaseModel):
    speech: SpeechInfo
    environment: str
    temporal_events: List[str]
    primary_events: List[EventInfo]
    supporting_events: List[EventInfo]
    background_events: List[EventInfo]
    ignored_events: List[str]
    clap_concepts: List[str]
    clap_confidence: float
    metadata: Optional[Dict[str, Any]] = None
    processing_information: Optional[Dict[str, Any]] = None
