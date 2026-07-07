import os
import sys
import glob
import time
import argparse
import librosa
import logging
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import UnifiedPipelineValidator
from core_modules.export_utils import Exporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation(mock_inference=False):
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    audio_files = sorted(glob.glob(os.path.join(dataset_dir, "*.*")))
    
    if not audio_files:
        logger.error("No audio files found in research/dataset/")
        return
        
    logger.info(f"Starting evaluation runner on {len(audio_files)} samples...")
    
    if not mock_inference:
        validator = UnifiedPipelineValidator()
    else:
        logger.warning("Running with --mock-inference flag enabled.")
        
    reports = []
    
    for file in audio_files:
        filename = os.path.basename(file)
        logger.info(f"Processing {filename}...")
        
        try:
            start_t = time.perf_counter()
            audio, sr = librosa.load(file, sr=16000)
            
            if mock_inference:
                # Mock delay and result
                time.sleep(0.1)
                latency = (time.perf_counter() - start_t) * 1000
                
                # We need to simulate the structure expected by Exporter and statistical_analysis
                class MockWorldState:
                    def __init__(self):
                        self.dominant_state = "Mocked Situation State"
                        
                report = {
                    "filename": filename,
                    "speech": "Simulated transcript.",
                    "environment": "Simulated environment.",
                    "situation": "Simulated human-oriented situation.",
                    "world_state": MockWorldState(),
                    "latencies": {"inference": latency, "cognitive_pipeline": 1.2}
                }
            else:
                report = validator.run_pipeline(audio, sr)
                report["filename"] = filename
                
            reports.append(report)
            
        except Exception as e:
            logger.error(f"Error on {filename}: {e}")
            
    csv_path = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
    Exporter.export_to_csv(reports, csv_path)
    logger.info(f"Evaluation complete. Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-inference", action="store_true", help="Skip model load and return mock data for testing downstream scripts.")
    args = parser.parse_args()
    
    run_evaluation(args.mock_inference)
