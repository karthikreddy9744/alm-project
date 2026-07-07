import os
import sys
import tracemalloc
import time
import librosa
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main import UnifiedPipelineValidator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing ALM Benchmark Suite...")
    
    start_load = time.perf_counter()
    validator = UnifiedPipelineValidator()
    load_time = time.perf_counter() - start_load
    logger.info(f"Model Loading Time: {load_time:.2f} seconds")
    
    sample_file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'YTDown_Shorts_Loki-s-Glorious-Purpose_Media_KPz1r8DCuH0_009_128k.mp3')
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample file not found: {sample_file}")
        return
        
    logger.info("Starting memory tracing...")
    tracemalloc.start()
    
    logger.info("Loading audio...")
    audio, sr = librosa.load(sample_file, sr=16000)
    
    logger.info("Running pipeline benchmark...")
    
    # Warmup
    logger.info("Running warmup pass...")
    validator.run_pipeline(audio[:sr*2], sr) # Just a short 2sec warmup
    
    tracemalloc.clear_traces()
    
    start_inference = time.perf_counter()
    report = validator.run_pipeline(audio, sr)
    total_inference = time.perf_counter() - start_inference
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info("\n" + "="*40)
    logger.info("       ALM v12.0 BENCHMARK REPORT       ")
    logger.info("="*40)
    
    logger.info(f"Model Loading Time: {load_time:.2f} s")
    logger.info(f"Total Pipeline Latency: {total_inference:.2f} s")
    
    logger.info("\nModule Breakdown:")
    for mod, lat in validator.latencies.items():
        logger.info(f"  - {mod}: {lat:.2f} ms")
        
    logger.info(f"\nPeak Memory (RAM): {peak / 10**6:.2f} MB")
    logger.info(f"Current Memory (RAM): {current / 10**6:.2f} MB")
    
    if report:
        logger.info("\nStatus: SUCCESS")
    else:
        logger.info("\nStatus: FAILED")
    logger.info("="*40)

if __name__ == "__main__":
    main()
