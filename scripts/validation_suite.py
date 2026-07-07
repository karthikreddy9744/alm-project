import os
import sys
import glob
import csv
import time
import librosa
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main import UnifiedPipelineValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing ALM Validation Suite...")
    validator = UnifiedPipelineValidator()
    
    samples_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    audio_files = glob.glob(os.path.join(samples_dir, '*.mp3')) + glob.glob(os.path.join(samples_dir, '*.wav'))
    
    output_file = "validation_report.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Status", "Latency_ms", "Speech_Output", "Environment_Output", "Situation_Output"])
        
        for file in audio_files:
            filename = os.path.basename(file)
            logger.info(f"Evaluating {filename}...")
            
            try:
                start_t = time.perf_counter()
                audio, sr = librosa.load(file, sr=16000)
                report = validator.run_pipeline(audio, sr)
                latency = (time.perf_counter() - start_t) * 1000
                
                if report:
                    writer.writerow([
                        filename,
                        "SUCCESS",
                        f"{latency:.2f}",
                        report.get("speech", ""),
                        report.get("environment", ""),
                        report.get("situation", "")
                    ])
                    logger.info(f"SUCCESS: {filename}")
                else:
                    writer.writerow([filename, "FAILED - No Report", 0, "", "", ""])
                    logger.warning(f"FAILED: {filename}")
            except Exception as e:
                writer.writerow([filename, f"ERROR - {str(e)}", 0, "", "", ""])
                logger.error(f"ERROR on {filename}: {e}")

    logger.info(f"Validation complete. Report saved to {output_file}.")

if __name__ == "__main__":
    main()
