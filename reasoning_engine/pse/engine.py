import logging
from typing import List, Tuple
from reasoning_engine.arg.graph import AcousticRelationshipGraph
from reasoning_engine.pse.models import SegregatedStreams

logger = logging.getLogger(__name__)

class PerceptualSegregationEngine:
    """
    ALM v10.8 Perceptual Segregation Engine.
    Filters the Auditory Relationship Graph (ARG) into Foreground and Background
    perceptual streams based on acoustic salience, confidence, and topological centrality.
    """
    def __init__(self, salience_threshold: float = 0.6):
        self.salience_threshold = salience_threshold

    def segregate(self, arg: AcousticRelationshipGraph) -> SegregatedStreams:
        """
        Processes the raw ARG and segregates nodes into distinct streams.
        Foreground: High salience / speech entities
        Background: Low salience / ambient beds
        """
        streams = SegregatedStreams()
        
        for node_id, node in arg._nodes.items():
            # Speech is almost always foreground in our curriculum
            from reasoning_engine.arg.models import NodeType
            if node.node_type == NodeType.SPEAKER and "Speaker" in node.metadata.get("label", ""):
                streams.foreground_nodes.append(node)
            else:
                # Events are separated by salience
                if node.metadata.get("acoustic_salience", 0.0) >= self.salience_threshold:
                    streams.foreground_nodes.append(node)
                else:
                    streams.background_nodes.append(node)
                    
        logger.debug(f"PSE Segregation: {len(streams.foreground_nodes)} Foreground, {len(streams.background_nodes)} Background")
        return streams
