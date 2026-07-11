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
from reasoning_engine.pse.engine import PerceptualSegregationEngine
from reasoning_engine.hre.engine import HypothesisReasoningEngine
from reasoning_engine.wse.engine import WorldStateEngine
from reasoning_engine.spe.engine import SituationProjectionEngine
from reasoning_engine.tre.engine import TransparentReasoningEngine
from reasoning_engine.sir.engine import SituationIntelligenceRenderer
from reasoning_engine.sir.models import RenderMode

from reasoning_engine.semantic.engine import SemanticInterpretationEngine

class UnifiedPipelineValidator:
    def __init__(self):
        logger.info("Initializing ALM v12.0 Unified Pipeline (Phase 1)...")
        # Neural Layer
        self.neural_pipeline = ALMInferencePipeline()
        
        # Semantic Interpretation Layer
        self.semantic_engine = SemanticInterpretationEngine()
        
        # Deterministic Logic Graph
        self.awm = AuditoryWorldModel()
        self.pse = PerceptualSegregationEngine()
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
        # Reset AWM and all stateful logic engines for independent evaluation
        self.awm = AuditoryWorldModel()
        self.pse = PerceptualSegregationEngine()
        self.hre = HypothesisReasoningEngine()
        self.wse = WorldStateEngine()
        self.spe = SituationProjectionEngine()
        self.tre = TransparentReasoningEngine()
        self.sir = SituationIntelligenceRenderer()
        
        # 1. Neural Perception (Whisper -> HTS-AT -> CLAP -> Fusion -> SceneNet -> AWM)
        logger.info("Running Neural Perception...")
        self.awm = self._measure_time("NeuralPerception", self.neural_pipeline.process, audio, sr, self.awm)
        
        # 2. PSE (Perceptual Segregation Engine - Contextual Masking)
        logger.info("Running Perceptual Segregation Engine...")
        streams = self._measure_time("PSE", self.pse.segregate, self.awm)
        
        # 2.5 Evidence Fusion Layer (Phase 2)
        logger.info("Running Evidence Fusion Layer...")
        from reasoning_engine.fusion.engine import EvidenceFusionLayer
        fusion_layer = EvidenceFusionLayer()
        audio_evidence = self._measure_time("EvidenceFusion", fusion_layer.fuse, self.awm, streams)

        # 3. Semantic Interpretation Engine (Phase 2)
        logger.info("Running Semantic Interpretation Engine...")
        semantic_json = self._measure_time("SemanticEngine", self.semantic_engine.generate_semantics, audio_evidence)
        
        # 3. HRE (Hypothesis Reasoning Engine)
        logger.info("Running Hypothesis Reasoning Engine...")
        self.hre.manage_hypotheses(
            semantic_json=semantic_json, 
            streams=streams, 
            temporal_history=self.wse.history
        )
        active_hyps = self._measure_time("HRE_rank", self.hre.rank)
        
        # 4. WSE (World State Engine)
        logger.info("Running World State Engine...")
        world_state = self._measure_time("WSE", self.wse.estimate, active_hyps, streams)
        
        # 5. SPE (Situation Projection Engine)
        logger.info("Running Situation Projection Engine...")
        projection = self._measure_time("SPE", self.spe.project, world_state)
        
        # Extract pointers for TRE
        awm_entities = {e.id: e.entity_type for e in self.awm.entities.values()}
        awm_events = {e.id: e.class_map for e in self.awm.events.values()}
        
        # 6. TRE (Transparent Reasoning Engine)
        logger.info("Running Transparent Reasoning Engine...")
        trace = self._measure_time("TRE", self.tre.trace, 
            awm_entities, 
            awm_events, 
            active_hyps, 
            world_state, 
            projection
        )
        
        # 7. SIR (Situation Intelligence Renderer)
        logger.info("Running Situation Intelligence Renderer...")
        try:
            sir_output = self._measure_time("SIR", self.sir.export,
                RenderMode.THREE_TIER_REPORT, world_state, projection, trace, active_hyps, streams
            )
            return {
                "speech": sir_output.get("speech"),
                "environment": sir_output.get("environment"),
                "situation": sir_output.get("situation"),
                "report": sir_output.get("full_report"),
                "world_state": world_state,
                "projection": projection,
                "trace": trace,
                "active_hyps": active_hyps,
                "streams": streams,
                "awm": self.awm,
                "audio_evidence": audio_evidence,
                "semantic_json": semantic_json,
                "latencies": self.latencies
            }
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return None

def main():
    logger.info("--- ALM v12.0 UNIFIED END-TO-END VALIDATION ---")
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
        logger.info(f"\n{report.get('report', 'No report generated')}")

if __name__ == "__main__":
    main()
