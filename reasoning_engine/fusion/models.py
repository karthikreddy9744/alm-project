from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any

class RecordingCharacterization(BaseModel):
    reverberation_profile: Optional[str] = None
    compression_level: Optional[str] = None
    post_processing_artifacts: Optional[bool] = None
    dynamic_range: Optional[str] = None
    noise_profile: Optional[str] = None
    recording_quality: Optional[str] = None

class SpeechInfo(BaseModel):
    transcript: str
    language: str
    confidence: float
    speaker_count: int
    timestamps: Optional[List[Tuple[float, float]]] = None
    keywords: Optional[List[str]] = None
    duration: Optional[float] = None

class EventInfo(BaseModel):
    id: str = "Unknown"
    event_label: str
    salience: float
    confidence: float
    is_foreground: bool
    is_background: bool
    start_time: float = 0.0
    end_time: float = 0.0
    detector: str = "Unknown"

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
    recording_characterization: Optional[RecordingCharacterization] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_information: Optional[Dict[str, Any]] = None
