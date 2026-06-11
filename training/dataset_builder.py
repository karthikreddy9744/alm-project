import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from core.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor

# ESC-50 class mapping to our 15 classes
ESC50_TO_OURS = {
    # 0: Emergency
    42: 0, # Siren
    12: 0, # Crackling fire

    # 1: Traffic
    40: 1, # Helicopter
    43: 1, # Car horn
    44: 1, # Engine
    45: 1, # Train
    47: 1, # Airplane

    # 2: Weather
    10: 2, # Rain
    16: 2, # Wind
    19: 2, # Thunderstorm

    # 3: Water
    11: 3, # Sea waves
    15: 3, # Water drops
    17: 3, # Pouring water

    # 4: Wildlife & Animals
    0: 4, # Dog
    1: 4, # Rooster
    2: 4, # Pig
    3: 4, # Cow
    4: 4, # Frog
    5: 4, # Cat
    6: 4, # Insects
    7: 4, # Sheep
    8: 4, # Crow
    9: 4, # Birds
    13: 4, # Crickets
    14: 4, # Chirping birds

    # 5: Indoor/Domestic
    18: 5, # Toilet flush
    30: 5, # Door knock
    33: 5, # Door, wood creaks
    34: 5, # Can opening
    37: 5, # Clock alarm
    38: 5, # Clock tick

    # 6: Home Appliances
    35: 6, # Washing machine
    36: 6, # Vacuum cleaner

    # 7: Office/Work
    31: 7, # Mouse click
    32: 7, # Keyboard typing

    # 8: Human Crowd
    22: 8, # Clapping
    26: 8, # Laughing

    # 9: Human Speech & Non-speech
    20: 9, # Crying baby
    21: 9, # Sneezing
    23: 9, # Breathing
    24: 9, # Coughing
    27: 9, # Brushing teeth
    28: 9, # Snoring
    29: 9, # Drinking/sipping

    # 10: Tools & Construction
    41: 10, # Chainsaw
    49: 10, # Hand saw

    # 11: Explosions & Weaponry
    39: 11, # Glass breaking
    48: 11, # Fireworks

    # 12: Music & Bells
    46: 12, # Church bells

    # 13: Footsteps
    25: 13, # Footsteps
    
    # 14: Silence/Unknown
    # No direct ESC-50 mapping, class reserved for inference.
}

def preprocess_dataset(esc50_path: str, output_path: str = "data/processed", mix_librispeech: bool = False):
    """
    Preprocess ESC-50 dataset into embeddings using Whisper and CLAP.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Load feature extractors
    whisper_fe = WhisperFeatureExtractor("base")
    clap_fe = CLAPFeatureExtractor()
    
    # Process each audio file
    embeddings = []
    labels = []
    
    meta_path = os.path.join(esc50_path, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")
    
    import pandas as pd
    df = pd.read_csv(meta_path)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        target_class = row["target"]
        if target_class not in ESC50_TO_OURS:
            continue  # Skip classes not in our mapping
        
        audio_path = os.path.join(esc50_path, "audio", row["filename"])
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract embeddings
        w_emb, _ = whisper_fe.extract(audio, sr)
        c_emb = clap_fe.extract(audio, sr)
        
        # Optional LibriSpeech Mixing (Simulated or via real dataset if implemented)
        if mix_librispeech:
            # Here we simulate mixing speech by blending Whisper embeddings with a generic "speech" distribution
            # In production, load a LibriSpeech wav, mix raw audio, and re-extract.
            speech_sim = np.random.normal(loc=0.5, scale=0.1, size=w_emb.numpy().shape)
            w_emb = torch.tensor((w_emb.numpy() + speech_sim) / 2.0, dtype=torch.float32)
        
        embeddings.append((w_emb.numpy(), c_emb.numpy()))
        labels.append(ESC50_TO_OURS[target_class])
    
    # Generate Synthetic Silence (Class 14)
    print("Generating Synthetic Silence (Class 14)...")
    for _ in range(80): # Match average class size
        silence_audio = np.zeros(16000 * 5, dtype=np.float32)
        w_emb, _ = whisper_fe.extract(silence_audio, 16000)
        c_emb = clap_fe.extract(silence_audio, 16000)
        embeddings.append((w_emb.numpy(), c_emb.numpy()))
        labels.append(14)
        
    # Save processed data
    np.save(os.path.join(output_path, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_path, "labels.npy"), labels)
    print(f"Preprocessed {len(embeddings)} samples saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50_path", type=str, default="data/raw", help="Path to ESC-50 dataset root (default: data/raw)")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--mix_librispeech", action="store_true", help="Enable optional LibriSpeech mixing")
    args = parser.parse_args()
    preprocess_dataset(args.esc50_path, args.output_path, args.mix_librispeech)
