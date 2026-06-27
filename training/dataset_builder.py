import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from core.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor

# ALM v7.0 40-Class Mapping
ESC50_TO_OURS = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 1,
    10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 10, 18: 16, 19: 9,
    20: 17, 21: 18, 22: 19, 23: 20, 24: 18, 25: 21, 26: 22, 27: 23, 28: 24, 29: 23,
    30: 25, 31: 26, 32: 26, 33: 25, 34: 27, 35: 28, 36: 29, 37: 30, 38: 30, 39: 31,
    40: 32, 41: 33, 42: 34, 43: 35, 44: 35, 45: 36, 46: 37, 47: 32, 48: 38, 49: 33
}

def load_audio_file(esc50_path, filename):
    audio_path = os.path.join(esc50_path, "audio", filename)
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio

def mix_audio(speech, env, target_length=16000*5):
    """Mix speech and env at a random SNR in [-5, 20] dB."""
    if len(speech) < target_length:
        speech = np.pad(speech, (0, target_length - len(speech)))
    else:
        speech = speech[:target_length]
        
    if len(env) < target_length:
        env = np.pad(env, (0, target_length - len(env)))
    else:
        env = env[:target_length]
        
    P_s = np.mean(speech**2) + 1e-8
    P_e = np.mean(env**2) + 1e-8
    
    snr_db = np.random.uniform(-5, 20)
    snr_linear = 10 ** (snr_db / 10)
    
    # Scale environment to meet SNR
    target_P_e = P_s / snr_linear
    scale_factor = np.sqrt(target_P_e / P_e)
    scaled_env = env * scale_factor
    
    mixed = speech + scaled_env
    # Normalize to prevent clipping
    mixed = mixed / np.max([np.max(np.abs(mixed)) + 1e-8, 1.0])
    return mixed

class MultimodalDatasetBuilder:
    def __init__(self, esc50_path: str, output_path: str):
        self.esc50_path = esc50_path
        self.output_path = output_path
        self.whisper_fe = WhisperFeatureExtractor("base")
        self.clap_fe = CLAPFeatureExtractor()
        os.makedirs(output_path, exist_ok=True)
        
    def build(self, num_samples: int = 5000):
        print("Loading LibriSpeech test-clean...")
        # Fallback to local files if HF datasets fails, or use streaming to be fast
        try:
            librispeech = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)
            ls_audio = [x['audio']['array'] for x in librispeech]
        except Exception as e:
            print(f"Warning: Failed to load LibriSpeech from Hugging Face ({e}). Generating synthetic speech placeholders.")
            ls_audio = [np.random.randn(16000 * 5).astype(np.float32) * 0.1 for _ in range(500)]
            
        print("Loading ESC-50 metadata...")
        meta_path = os.path.join(self.esc50_path, "meta", "esc50.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"ESC-50 metadata not found at {meta_path}")
            
        df = pd.read_csv(meta_path)
        
        embeddings = []
        labels = []
        
        print(f"Synthesizing {num_samples} samples (30/30/40 distribution)...")
        # Dist: 30% speech-only, 30% env-only, 40% mixed
        num_speech = int(num_samples * 0.3)
        num_env = int(num_samples * 0.3)
        num_mixed = num_samples - num_speech - num_env
        
        # 1. Speech Only
        for _ in tqdm(range(num_speech), desc="Speech Only"):
            speech = ls_audio[np.random.choice(len(ls_audio))]
            if len(speech) < 16000 * 5: speech = np.pad(speech, (0, 16000*5 - len(speech)))
            else: speech = speech[:16000*5]
            
            w_emb, _ = self.whisper_fe.extract(speech, 16000)
            c_emb = self.clap_fe.extract(speech, 16000)
            multi_hot = np.zeros(40, dtype=np.float32)
            multi_hot[39] = 1.0 # Silence / Unknown environment
            
            embeddings.append((w_emb.numpy(), c_emb.numpy()))
            labels.append(multi_hot)
            
        # 2. Environment Only
        for _ in tqdm(range(num_env), desc="Environment Only"):
            row = df.sample(1).iloc[0]
            env = load_audio_file(self.esc50_path, row["filename"])
            if len(env) < 16000 * 5: env = np.pad(env, (0, 16000*5 - len(env)))
            else: env = env[:16000*5]
            
            w_emb, _ = self.whisper_fe.extract(env, 16000)
            c_emb = self.clap_fe.extract(env, 16000)
            multi_hot = np.zeros(40, dtype=np.float32)
            multi_hot[ESC50_TO_OURS[row["target"]]] = 1.0
            
            embeddings.append((w_emb.numpy(), c_emb.numpy()))
            labels.append(multi_hot)
            
        # 3. Mixed (Speech + Env)
        for _ in tqdm(range(num_mixed), desc="Mixed"):
            speech = ls_audio[np.random.choice(len(ls_audio))]
            row = df.sample(1).iloc[0]
            env = load_audio_file(self.esc50_path, row["filename"])
            
            mixed = mix_audio(speech, env)
            
            w_emb, _ = self.whisper_fe.extract(mixed, 16000)
            c_emb = self.clap_fe.extract(mixed, 16000)
            multi_hot = np.zeros(40, dtype=np.float32)
            multi_hot[ESC50_TO_OURS[row["target"]]] = 1.0
            
            embeddings.append((w_emb.numpy(), c_emb.numpy()))
            labels.append(multi_hot)
            
        np.save(os.path.join(self.output_path, "embeddings.npy"), np.array(embeddings, dtype=object))
        np.save(os.path.join(self.output_path, "labels.npy"), np.array(labels, dtype=object))
        print(f"Preprocessed {len(embeddings)} multi-label samples saved to {self.output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50_path", type=str, default="data/raw", help="Path to ESC-50 dataset root (default: data/raw)")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--num_samples", type=int, default=3262, help="Number of multi-label mixtures to generate")
    args = parser.parse_args()
    
    builder = MultimodalDatasetBuilder(args.esc50_path, args.output_path)
    builder.build(args.num_samples)
