import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
import pandas as pd
from core.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor

# Updated mapping to our 20-class SCENE_LABELS
# 0: Traffic & Vehicles, 1: Siren & Alarm, 2: Crowd & Hubbub, 3: Weather & Nature, 4: Water
# 5: Indoor / Domestic, 6: Office, 7: Construction, 8: Wildlife / Animals, 9: Music
# 10: Television / Media, 11: Movie Scene, 12: Public Transport, 13: Airport, 14: Sports Event
# 15: Restaurant / Cafe, 16: Mall, 17: Home, 18: Footsteps, 19: Silence / Unknown
ESC50_TO_OURS = {
    0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, # Animals -> Wildlife
    10: 3, 11: 4, 12: 3, 13: 8, 14: 8, 15: 4, 16: 3, 17: 4, 19: 3, # Nature
    18: 5, 21: 5, 23: 5, 24: 5, 27: 5, 28: 5, 29: 5, 30: 5, 33: 5, 34: 5, 37: 5, 38: 5, 39: 5, # Indoor
    20: 17, 35: 17, 36: 17, # Home
    22: 2, 26: 2, 48: 2, # Crowd/Hubbub
    25: 18, # Footsteps
    31: 6, 32: 6, # Office
    40: 0, 43: 0, 44: 0, # Traffic
    41: 7, 49: 7, # Construction
    42: 1, # Siren
    45: 12, # Public Transport
    46: 9, # Music (Bells)
    47: 13, # Airport
}

def load_audio_file(esc50_path, filename):
    audio_path = os.path.join(esc50_path, "audio", filename)
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio

def preprocess_dataset(esc50_path: str, output_path: str = "data/processed", num_samples: int = 5000):
    """
    Preprocess ESC-50 dataset into multi-label embeddings using Whisper and CLAP.
    We synthesize multi-label examples by mixing 1 to 3 audio clips.
    """
    os.makedirs(output_path, exist_ok=True)
    
    whisper_fe = WhisperFeatureExtractor("base")
    clap_fe = CLAPFeatureExtractor()
    
    meta_path = os.path.join(esc50_path, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")
    
    df = pd.read_csv(meta_path)
    # Filter to classes we have mapped
    valid_df = df[df["target"].isin(ESC50_TO_OURS.keys())].reset_index(drop=True)
    
    embeddings = []
    labels = []
    
    print(f"Synthesizing {num_samples} multi-label mixtures from ESC-50...")
    for _ in tqdm(range(num_samples)):
        # Choose 1 to 3 random audio clips to mix
        num_mixes = np.random.randint(1, 4)
        sample_rows = valid_df.sample(num_mixes)
        
        mixed_audio = np.zeros(16000 * 5, dtype=np.float32) # ESC-50 clips are 5s
        multi_hot = np.zeros(20, dtype=np.float32)
        
        for _, row in sample_rows.iterrows():
            audio = load_audio_file(esc50_path, row["filename"])
            # Pad or truncate to 5s just in case
            if len(audio) < 16000 * 5:
                audio = np.pad(audio, (0, 16000 * 5 - len(audio)))
            else:
                audio = audio[:16000 * 5]
            
            # Mix audio (simple addition, clipping handled by normalisation if needed, but Whisper/CLAP are robust)
            mixed_audio += audio
            
            # Add label
            target_class = ESC50_TO_OURS[row["target"]]
            multi_hot[target_class] = 1.0
            
        # Extract features
        w_emb, _ = whisper_fe.extract(mixed_audio, 16000)
        c_emb = clap_fe.extract(mixed_audio, 16000)
        
        embeddings.append((w_emb.numpy(), c_emb.numpy()))
        labels.append(multi_hot)
        
    # Generate Synthetic Silence (Class 19)
    print("Generating Synthetic Silence (Class 19)...")
    for _ in range(num_samples // 10): # 10% of dataset is silence
        silence_audio = np.zeros(16000 * 5, dtype=np.float32)
        w_emb, _ = whisper_fe.extract(silence_audio, 16000)
        c_emb = clap_fe.extract(silence_audio, 16000)
        
        multi_hot = np.zeros(20, dtype=np.float32)
        multi_hot[19] = 1.0
        
        embeddings.append((w_emb.numpy(), c_emb.numpy()))
        labels.append(multi_hot)
        
    np.save(os.path.join(output_path, "embeddings.npy"), np.array(embeddings, dtype=object))
    np.save(os.path.join(output_path, "labels.npy"), np.array(labels, dtype=object))
    print(f"Preprocessed {len(embeddings)} multi-label samples saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50_path", type=str, default="data/raw", help="Path to ESC-50 dataset root (default: data/raw)")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of multi-label mixtures to generate")
    args = parser.parse_args()
    preprocess_dataset(args.esc50_path, args.output_path, args.num_samples)
