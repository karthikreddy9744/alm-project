from reasoning_engine.fusion.models import AudioEvidenceObject, SpeechInfo, EventInfo
from reasoning_engine.awm.world_model import AuditoryWorldModel
from reasoning_engine.pse.models import SegregatedStreams
from typing import Any

class EvidenceFusionLayer:
    """
    Evidence Fusion Layer (ALM Phase 2).
    Merges Whisper, HTS-AT, and CLAP outputs into a structured AudioEvidenceObject.
    Performs ZERO reasoning.
    """
    def __init__(self):
        pass

    def fuse(self, awm: AuditoryWorldModel, streams: SegregatedStreams) -> AudioEvidenceObject:
        speech_info = SpeechInfo(transcript="", language="en", confidence=0.0, speaker_count=0)
        
        for e in awm.entities.values():
            if e.entity_type == "Speaker":
                speech_info = SpeechInfo(
                    transcript=getattr(e, 'transcript', ''),
                    language=getattr(e, 'language', 'en'),
                    confidence=e.confidence.speech_recognition,
                    speaker_count=1,
                    timestamps=getattr(e, 'timestamps', None),
                    duration=None
                )
                
        def map_events(events, is_foreground, is_background):
            info = []
            for ev in events:
                info.append(EventInfo(
                    id=ev.id,
                    event_label=ev.class_map,
                    salience=ev.acoustic_salience,
                    confidence=ev.confidence.sound_detection,
                    is_foreground=is_foreground,
                    is_background=is_background,
                    start_time=getattr(ev, 'start_time', 0.0),
                    end_time=getattr(ev, 'end_time', 0.0),
                    detector=getattr(ev, 'detector', 'Unknown')
                ))
            return info
            
        primary_events = map_events(streams.primary_events, True, False)
        supporting_events = map_events(streams.supporting_events, True, False)
        background_events = map_events(streams.background_events, False, True)
        ignored_events = [ev.class_map for ev in streams.ignored_events]
        
        temporal_events = [ev.class_map for ev in streams.primary_events + streams.supporting_events + streams.background_events]
        
        return AudioEvidenceObject(
            speech=speech_info,
            environment="Unknown Environment", # Will be reasoned over by Semantic Engine
            temporal_events=temporal_events,
            primary_events=primary_events,
            supporting_events=supporting_events,
            background_events=background_events,
            ignored_events=ignored_events,
            clap_concepts=getattr(awm, 'clap_concepts', []),
            clap_confidence=0.85,
            metadata={"source": "ALM_EvidenceFusionLayer"},
            processing_information={"status": "fused"}
        )
