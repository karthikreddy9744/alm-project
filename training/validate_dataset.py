import os
import glob
import json
import librosa
from tqdm import tqdm
import argparse

def validate_audio_files(directory, target_sr=16000):
    files = glob.glob(os.path.join(directory, "**/*.wav"), recursive=True)
    report = {
        "total_files": len(files),
        "valid_files": 0,
        "corrupted_files": 0,
        "total_duration_seconds": 0.0,
        "invalid_sr_files": 0
    }
    
    corrupted_list = []
    
    print(f"Validating {len(files)} files in {directory}...")
    for f in tqdm(files):
        try:
            # We use librosa.get_duration to quickly check if it's readable
            duration = librosa.get_duration(path=f)
            sr = librosa.get_samplerate(f)
            
            report["valid_files"] += 1
            report["total_duration_seconds"] += duration
            
            if sr != target_sr:
                report["invalid_sr_files"] += 1
                
        except Exception as e:
            report["corrupted_files"] += 1
            corrupted_list.append((f, str(e)))
            
    return report, corrupted_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Real ALM Datasets")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw audio dataset")
    args = parser.parse_args()
    
    speech_dir = os.path.join(args.data_dir, "speech")
    events_dir = os.path.join(args.data_dir, "events")
    ambient_dir = os.path.join(args.data_dir, "ambient")
    
    final_report = {}
    
    for name, d in [("Speech", speech_dir), ("Events", events_dir), ("Ambient", ambient_dir)]:
        if os.path.exists(d):
            rep, corrupts = validate_audio_files(d)
            final_report[name] = rep
            if corrupts:
                print(f"WARNING: Found {len(corrupts)} corrupted files in {name}.")
        else:
            final_report[name] = {"error": f"Directory not found: {d}"}
            
    # Language detection is heuristic based on filenames (fleurs lang codes)
    if os.path.exists(speech_dir):
        speech_files = glob.glob(os.path.join(speech_dir, "*.wav"))
        langs = set()
        for sf in speech_files:
            base = os.path.basename(sf)
            if "librispeech" in base:
                langs.add("en")
            elif "fleurs" in base:
                # e.g. hi_in_fleurs_0.wav
                langs.add(base.split("_fleurs")[0])
        final_report["Speech"]["languages"] = list(langs)
        
    print("\n--- VALIDATION REPORT ---")
    print(json.dumps(final_report, indent=4))
    
    with open(os.path.join(args.data_dir, "dataset_metadata.json"), "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Metadata saved to {os.path.join(args.data_dir, 'dataset_metadata.json')}")
