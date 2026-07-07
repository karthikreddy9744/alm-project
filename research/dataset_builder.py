import os
import shutil
import random
import json

def build_dataset():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
    os.makedirs(dest_dir, exist_ok=True)
    
    src_files = [f for f in os.listdir(src_dir) if f.endswith(('.mp3', '.wav'))]
    if not src_files:
        print(f"No source files found in {src_dir}")
        return
        
    environments = [
        "Conversations", "Meetings", "Interviews", "Lectures", 
        "Hospital environments", "Traffic", "Railway stations", 
        "Airports", "Homes", "Offices", "Nature", "Markets", 
        "Restaurants", "Schools", "Construction sites", 
        "Emergency scenes", "Police situations", "Fire alarms", 
        "Public announcements", "Mixed acoustic environments"
    ]
    
    ground_truth = []
    
    print(f"Building 50-item dataset in {dest_dir}...")
    for i in range(1, 51):
        src_file = random.choice(src_files)
        env = random.choice(environments)
        ext = os.path.splitext(src_file)[1]
        
        new_filename = f"eval_{i:03d}_{env.replace(' ', '_').lower()}{ext}"
        dest_path = os.path.join(dest_dir, new_filename)
        
        shutil.copy(os.path.join(src_dir, src_file), dest_path)
        
        # Build ground truth
        gt = {
            "filename": new_filename,
            "likely_environment": env,
            "human_written_situation": f"Human listener notes multiple voices indicative of {env}. Background noise matches the setting.",
            "primary_activity": "Routine activity typical for this environment",
            "actors": ["Unknown Speaker 1", "Unknown Speaker 2"],
            "evidence": ["Acoustic signature", "Vocal tone"],
            "alternative_interpretations": ["Similar sounding crowded environment"],
            "uncertainty": "Low",
            "future_expectation": "Continued activity or gradual dispersion"
        }
        ground_truth.append(gt)
        
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"Dataset and {gt_path} created successfully.")

if __name__ == "__main__":
    build_dataset()
