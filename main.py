import time
import tracemalloc
import logging
import numpy as np
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from core_modules.inference_pipeline import ALMInferencePipeline
from reasoning_engine.awm.world_model import AuditoryWorldModel
from reasoning_engine.bse.engine import BeliefStateEngine
from reasoning_engine.bse.models import BeliefObject, BeliefType
from reasoning_engine.awm.models import HierarchicalConfidence
from reasoning_engine.arg.graph import AcousticRelationshipGraph
from reasoning_engine.arg.models import ARGNode, ARGEdge, EdgeType, NodeType
from reasoning_engine.pse.engine import PerceptualSegregationEngine
from reasoning_engine.hre.engine import HypothesisReasoningEngine
from reasoning_engine.wse.engine import WorldStateEngine
from reasoning_engine.spe.engine import SituationProjectionEngine
from reasoning_engine.tre.engine import TransparentReasoningEngine
from reasoning_engine.sir.engine import SituationIntelligenceRenderer
from reasoning_engine.sir.models import RenderMode

class UnifiedPipelineValidator:
    def __init__(self):
        logger.info("Initializing ALM v10.7 Unified Pipeline...")
        # Neural Layer
        self.neural_pipeline = ALMInferencePipeline()
        
        # Deterministic Logic Graph
        self.awm = AuditoryWorldModel()
        self.arg = AcousticRelationshipGraph()
        self.pse = PerceptualSegregationEngine()
        self.bse = BeliefStateEngine()
        self.hre = HypothesisReasoningEngine()
        self.wse = WorldStateEngine()
        self.spe = SituationProjectionEngine()
        self.tre = TransparentReasoningEngine()
        self.sir = SituationIntelligenceRenderer()
        
        self.latencies = {}
        
    def _measure_time(self, module_name, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        self.latencies[module_name] = (end - start) * 1000 # ms
        return result
        
    def run_pipeline(self, audio: np.ndarray, sr: int):
        self.awm = AuditoryWorldModel() # Reset AWM
        
        # 1. Neural Perception (Whisper -> HTS-AT -> CLAP -> Fusion -> SceneNet -> AWM)
        logger.info("Running Neural Perception...")
        self.awm = self._measure_time("NeuralPerception", self.neural_pipeline.process, audio, sr, self.awm)
        
        # 1.5 ARG & PSE (Auditory Relationship Graph & Perceptual Segregation)
        logger.info("Building ARG & Running PSE...")
        self.arg = AcousticRelationshipGraph()
        
        # Build ARG Nodes from AWM
        for event in self.awm.events.values():
            node = ARGNode(id=event.id, node_type=NodeType.ENVIRONMENTAL_EVENT, confidence=event.confidence, metadata={"label": event.class_map, "acoustic_salience": event.acoustic_salience})
            self.arg.add_node(node)
            
        for entity in self.awm.entities.values():
            node = ARGNode(id=entity.id, node_type=NodeType.SPEAKER, confidence=entity.confidence, metadata={"label": entity.entity_type, "acoustic_salience": 1.0})
            self.arg.add_node(node)
            
        # Build edges (e.g. Speech co-occurring with High Tension events)
        high_tension = ["Siren", "Gunshot", "Screaming", "Glass breaking"]
        for event_node in self.arg._nodes.values():
            if event_node.node_type == NodeType.ENVIRONMENTAL_EVENT and event_node.metadata.get("label") in high_tension:
                for entity_node in self.arg._nodes.values():
                    if entity_node.node_type == NodeType.SPEAKER and "Speaker" in entity_node.metadata.get("label", ""):
                        edge = ARGEdge(source_id=event_node.id, target_id=entity_node.id, edge_type=EdgeType.CO_OCCURS, confidence=HierarchicalConfidence(fusion=0.9))
                        self.arg.add_edge(edge)
                        
        # Segregate Streams
        streams = self._measure_time("PSE", self.pse.segregate, self.arg)
        
        # Build deterministic inputs from Streams
        awm_events = {node.id: node.metadata.get("label") for node in streams.background_nodes}
        awm_speech = {}
        for node in streams.foreground_nodes:
            if node.node_type == NodeType.SPEAKER and "Speaker" in node.metadata.get("label", ""):
                # We can't access transcript from ARGNode easily unless added to metadata, 
                # but we know it's speech. Let's fetch transcript from AWM.
                spk_entity = self.awm.entities.get(node.id)
                if spk_entity and hasattr(spk_entity, "transcript"):
                    awm_speech[node.id] = getattr(spk_entity, "transcript")
                    
        # Extract edge relationships for downstream modules
        arg_relationships = {}
        for src, edges in self.arg._out_edges.items():
            for (tgt, edge_type), edge_obj in edges.items():
                arg_relationships[f"{src}_to_{tgt}"] = str(edge_type.name)

        # 2. BSE
        logger.info("Running Belief State Engine...")
        b_conf = HierarchicalConfidence(beliefs=0.85)
        b1_stmt = "Complex overlapping scene detected"
        if awm_speech: b1_stmt += " with speech"
        b1 = BeliefObject(id="b_1", statement=b1_stmt, belief_type=BeliefType.EVENT, confidence=b_conf)
        
        # 3. HRE
        logger.info("Running Hypothesis Reasoning Engine...")
        h_conf = HierarchicalConfidence(beliefs=0.85, hypotheses=0.85)
        hyp = self._measure_time("HRE", self.hre.generate, 
            hypothesis_id="h_1", 
            statement="Environment is active", 
            supporting_beliefs=[b1.id], 
            confidence=h_conf
        )
        active_hyps = self.hre.rank()
        
        # 4. WSE
        logger.info("Running World State Engine...")
        world_state = self._measure_time("WSE", self.wse.estimate, active_hyps, [b1], awm_events, arg_relationships)
        
        # 5. SPE
        logger.info("Running Situation Projection Engine...")
        projection = self._measure_time("SPE", self.spe.project, world_state, active_hyps, [b1])
        
        # 6. TRE
        logger.info("Running Transparent Reasoning Engine...")
        trace = self._measure_time("TRE", self.tre.trace, 
            awm_events, 
            arg_relationships, 
            [b1], 
            active_hyps, 
            world_state, 
            projection
        )
        
        # 7. SIR
        logger.info("Running Situation Intelligence Renderer...")
        try:
            human_report = self._measure_time("SIR", self.sir.export,
                RenderMode.DEVELOPER, world_state, projection, trace, active_hyps, [b1]
            )
            return human_report
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return None

def main():
    logger.info("--- ALM v10.7 UNIFIED END-TO-END VALIDATION ---")
    tracemalloc.start()
    
    validator = UnifiedPipelineValidator()
    
    # Generate a dummy audio signal for end-to-end testing (e.g. 5 seconds of noise)
    logger.info("Generating test acoustic scene...")
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32) * 0.05
    
    try:
        report = validator.run_pipeline(dummy_audio, 16000)
    except Exception as e:
        logger.error(f"Fatal error during pipeline execution: {e}")
        return
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info("--- PERFORMANCE METRICS ---")
    logger.info(f"Total Latency: {sum(validator.latencies.values()):.2f} ms")
    for mod, lat in validator.latencies.items():
        logger.info(f"  {mod}: {lat:.2f} ms")
        
    logger.info(f"Peak Memory: {peak / 10**6:.4f} MB")
    
    if report:
        logger.info("--- SIR OUTPUT ---")
        logger.info(f"\\n{report}")

if __name__ == "__main__":
    main()
