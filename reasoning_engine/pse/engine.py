import logging
from typing import List, Dict
from reasoning_engine.awm.world_model import AuditoryWorldModel
from reasoning_engine.pse.models import SegregatedStreams

logger = logging.getLogger(__name__)

class PerceptualSegregationEngine:
    """
    ALM v12.17 Perceptual Segregation Engine.
    Implements true Selective Attention by segregating the Auditory World Model
    into Primary, Supporting, Background, and Ignored streams based on contextual masking.
    """
    def __init__(self, high_salience_threshold: float = 0.75, mask_threshold: float = 0.3):
        self.high_salience_threshold = high_salience_threshold
        self.mask_threshold = mask_threshold

    def segregate(self, awm: AuditoryWorldModel) -> SegregatedStreams:
        """
        Applies cognitive attention rules to segregate streams with ContextImportance.
        """
        streams = SegregatedStreams()
        
        high_tension_classes = ["Siren", "Gunshot", "Screaming", "Glass breaking", "Explosion"]
        
        has_high_tension = False
        for event in awm.events.values():
            if event.class_map in high_tension_classes:
                has_high_tension = True
                break
                
        # 1. Speech classification
        for entity in awm.entities.values():
            if "Speaker" in entity.entity_type:
                if has_high_tension:
                    streams.supporting_entities.append(entity)
                else:
                    streams.primary_entities.append(entity)
                    
        has_primary_entities = len(streams.primary_entities) > 0
        has_supporting_entities = len(streams.supporting_entities) > 0
        
        # 2. Filter Events (Selective Attention)
        for event in awm.events.values():
            is_high_tension = event.class_map in high_tension_classes
            
            if is_high_tension or (event.acoustic_salience >= self.high_salience_threshold and not has_primary_entities):
                # Extreme events, or very loud events when no primary speech
                streams.primary_events.append(event)
            elif event.acoustic_salience >= self.high_salience_threshold and has_primary_entities:
                streams.supporting_events.append(event)
            elif event.acoustic_salience >= self.mask_threshold:
                # Moderate sounds are background
                streams.background_events.append(event)
            else:
                # Low salience sounds
                if has_primary_entities or len(streams.primary_events) > 0:
                    streams.ignored_events.append(event)
                    streams.ignored_reasons[event.id] = f"Masked by primary focus (salience {event.acoustic_salience:.2f} < {self.mask_threshold})"
                else:
                    # Quiet environment
                    streams.background_events.append(event)
                    
        logger.debug(f"PSE Segregation: Primary Entities: {len(streams.primary_entities)}, Primary Events: {len(streams.primary_events)}")
        return streams

