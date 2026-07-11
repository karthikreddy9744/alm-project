import os
import random
import glob
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple
from tqdm import tqdm

from core_modules.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor, HTSATFeatureExtractor
from core_modules.scene_network import SCENE_LABELS

# ESC-50 to ALM taxonomy mapping
ESC50_TO_OURS = {k: i for i, k in enumerate(SCENE_LABELS)}
if "Silence / Unknown" not in ESC50_TO_OURS:
    ESC50_TO_OURS["Silence / Unknown"] = 39

class UnifiedDatasetBuilder:
    def __init__(self, raw_audio_path: str, output_dir: str, target_sr: int = 16000, max_events: int = 6, use_mock_data: bool = False):
        self.raw_audio_path = raw_audio_path
        self.speech_dir = os.path.join(raw_audio_path, "speech")
        self.env_dir = os.path.join(raw_audio_path, "events")
        self.ambient_dir = os.path.join(raw_audio_path, "ambient")
        self.output_dir = output_dir
        self.target_sr = target_sr
        self.max_events = max_events
        self.use_mock_data = use_mock_data
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.speech_dir, exist_ok=True)
        os.makedirs(self.env_dir, exist_ok=True)
        os.makedirs(self.ambient_dir, exist_ok=True)
        
        if self.use_mock_data:
            self._ensure_mock_data_exists()
        else:
            self._verify_real_data_exists()
            
        # Enable multilingual directories by using recursive globbing
        self.speech_files = glob.glob(os.path.join(self.speech_dir, "**/*.wav"), recursive=True)
        self.ambient_files = glob.glob(os.path.join(self.ambient_dir, "**/*.wav"), recursive=True)
        
        print("Initializing Frozen Foundation Extractors...")
        self.whisper_fe = WhisperFeatureExtractor("base")
        self.clap_fe = CLAPFeatureExtractor()
        self.htsat_fe = HTSATFeatureExtractor()

    def _ensure_mock_data_exists(self):
        """Generates mock speech, distinct events, and continuous ambient beds if missing."""
        if not glob.glob(os.path.join(self.speech_dir, "**/*.wav"), recursive=True):
            print("No speech wavs found. Generating mock speech files...")
            for i in range(5):
                t = np.linspace(0, 3.0, int(self.target_sr * 3.0))
                mock_speech = np.sin(2 * np.pi * 300 * t) * np.exp(-t)
                sf.write(os.path.join(self.speech_dir, f"mock_speech_{i}.wav"), mock_speech, self.target_sr)
                
        for label in SCENE_LABELS:
            if label == "Silence / Unknown": continue
            class_dir = os.path.join(self.env_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            if not glob.glob(os.path.join(class_dir, "*.wav")):
                t = np.linspace(0, 2.0, int(self.target_sr * 2.0))
                mock_env = np.random.randn(len(t)) * np.exp(-t)
                safe_label = label.replace(' ', '_').replace('/', '_')
                sf.write(os.path.join(class_dir, f"mock_env_{safe_label}.wav"), mock_env, self.target_sr)
                
        if not glob.glob(os.path.join(self.ambient_dir, "*.wav")):
            print("No ambient beds found. Generating mock ambient beds...")
            for bed_type in ["indoor_hum", "outdoor_wind"]:
                t = np.linspace(0, 15.0, int(self.target_sr * 15.0))
                if bed_type == "indoor_hum":
                    bed = np.sin(2 * np.pi * 50 * t) * 0.05 + np.random.randn(len(t)) * 0.01
                else:
                    bed = np.random.randn(len(t)) * 0.03
                sf.write(os.path.join(self.ambient_dir, f"mock_{bed_type}.wav"), bed.astype(np.float32), self.target_sr)

    def _verify_real_data_exists(self):
        """Ensures that the real dataset structure exists and contains wav files."""
        has_speech = len(glob.glob(os.path.join(self.speech_dir, "**/*.wav"), recursive=True)) > 0
        has_env = len(glob.glob(os.path.join(self.env_dir, "**/*.wav"), recursive=True)) > 0
        has_ambient = len(glob.glob(os.path.join(self.ambient_dir, "**/*.wav"), recursive=True)) > 0
        
        if not (has_speech and has_env and has_ambient):
            raise FileNotFoundError(
                "Missing real dataset files! Either place real data in "
                f"{self.raw_audio_path} (speech/, events/, ambient/) or run "
                "dataset_builder with --use_mock_data (not recommended for production)."
            )

    def _sample_duration(self) -> float:
        """3-5s (20%), 5-8s (50%), 8-12s (30%)"""
        choice = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
        if choice == 0: return random.uniform(3.0, 5.0)
        elif choice == 1: return random.uniform(5.0, 8.0)
        else: return random.uniform(8.0, 12.0)

    def _sample_snr(self) -> float:
        """-5 to 0 (15%), 0 to 5 (20%), 5 to 10 (30%), 10 to 15 (20%), 15 to 20 (15%)"""
        choice = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.20, 0.30, 0.20, 0.15])
        if choice == 0: return random.uniform(-5.0, 0.0)
        elif choice == 1: return random.uniform(0.0, 5.0)
        elif choice == 2: return random.uniform(5.0, 10.0)
        elif choice == 3: return random.uniform(10.0, 15.0)
        else: return random.uniform(15.0, 20.0)

    def _apply_snr(self, foreground: np.ndarray, background: np.ndarray, target_snr_db: float) -> np.ndarray:
        power_fg = np.mean(foreground**2) + 1e-10
        power_bg = np.mean(background**2) + 1e-10
        current_snr_db = 10 * np.log10(power_fg / power_bg)
        scale_factor = 10 ** ((target_snr_db - current_snr_db) / 20)
        return foreground * scale_factor

    def _load_and_process_speech(self, file_path: str, max_length: int) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.target_sr)
        
        # Time-stretch (speaking rate variance: 0.8x to 1.2x)
        rate = random.uniform(0.8, 1.2)
        if rate != 1.0:
            audio = librosa.effects.time_stretch(y=audio, rate=rate)
            
        if len(audio) > max_length:
            return audio[:max_length]
        return audio

    def _load_and_process_event(self, file_path: str, max_length: int) -> np.ndarray:
        audio, _ = librosa.load(file_path, sr=self.target_sr)
        
        # Reverb
        intensity = random.uniform(0.1, 0.6)
        decay = np.exp(-np.linspace(0, 5, int(self.target_sr * 0.3)))
        reverb = np.convolve(audio, decay, mode='full')[:len(audio)]
        audio = (audio * (1 - intensity)) + (reverb * intensity * 0.1)
        
        if len(audio) > max_length:
            return audio[:max_length]
        return audio
        
    def _get_curriculum_specs(self, stage: int) -> Tuple[int, int]:
        """Returns (num_speakers, num_events) based on stage"""
        if stage == 1: return (1, 1)                  # Stage 1: 1 Spk + 1 Env
        elif stage == 2: return (1, random.randint(2,3)) # Stage 2: 1 Spk + 2-3 Env
        elif stage == 3: return (2, random.randint(2,3)) # Stage 3: 2 Spk + 2-3 Env
        elif stage == 4: return (3, random.randint(3, self.max_events)) # Stage 4: 3 Spk + 3-max Env
        elif stage == 5: return (0, random.randint(1, self.max_events)) # Stage 5: Env Only
        elif stage == 6: return (random.randint(1,3), 0) # Stage 6: Speech Only
        return (1, 1)

    def generate_scene(self, stage: int) -> Tuple[np.ndarray, np.ndarray, int]:
        duration_s = self._sample_duration()
        scene_length = int(duration_s * self.target_sr)
        final_mix = np.zeros(scene_length, dtype=np.float32)
        multi_hot = np.zeros(40, dtype=np.float32)
        
        num_speakers, num_events = self._get_curriculum_specs(stage)
        
        # 1. Base Ambient Bed
        if self.ambient_files:
            ambient_file = random.choice(self.ambient_files)
            ambient_audio, _ = librosa.load(ambient_file, sr=self.target_sr)
            if len(ambient_audio) > scene_length:
                start = random.randint(0, len(ambient_audio) - scene_length)
                ambient_audio = ambient_audio[start:start+scene_length]
            else:
                ambient_audio = np.pad(ambient_audio, (0, scene_length - len(ambient_audio)))
            final_mix += ambient_audio * random.uniform(0.1, 0.5)

        # 2. Multi-speaker Speech
        if num_speakers > 0 and self.speech_files:
            # Note: We do NOT map speech into the 40-class multi_hot array, as speech is 
            # exclusively handled by the Whisper path, and indexing here overwrites index 0.
            selected_speakers = random.choices(self.speech_files, k=num_speakers)
            
            for spk_file in selected_speakers:
                spk_audio = self._load_and_process_speech(spk_file, scene_length)
                
                # Interrupted Conversation (30% chance)
                if random.random() < 0.3 and len(spk_audio) > self.target_sr:
                    cut_start = random.randint(0, len(spk_audio) // 2)
                    cut_end = cut_start + int(self.target_sr * random.uniform(0.5, 1.5))
                    spk_audio[cut_start:min(cut_end, len(spk_audio))] = 0.0
                
                start_idx = random.randint(0, scene_length - len(spk_audio))
                # Apply SNR
                target_snr = self._sample_snr()
                bg_segment = final_mix[start_idx:start_idx+len(spk_audio)]
                if np.mean(bg_segment**2) > 0:
                    spk_audio = self._apply_snr(spk_audio, bg_segment, target_snr)
                    
                final_mix[start_idx:start_idx+len(spk_audio)] += spk_audio

        # 3. Environmental Events
        if num_events > 0:
            available_classes = [c for c in SCENE_LABELS if c != "Silence / Unknown" and c != "Speech (English)"]
            
            for _ in range(num_events):
                event_class = random.choice(available_classes)
                class_dir = os.path.join(self.env_dir, event_class)
                event_files = glob.glob(os.path.join(class_dir, "*.wav"))
                
                if event_files:
                    evt_file = random.choice(event_files)
                    evt_audio = self._load_and_process_event(evt_file, scene_length)
                    
                    start_idx = random.randint(0, scene_length - len(evt_audio))
                    target_snr = self._sample_snr()
                    
                    bg_segment = final_mix[start_idx:start_idx+len(evt_audio)]
                    if np.mean(bg_segment**2) > 0:
                        evt_audio = self._apply_snr(evt_audio, bg_segment, target_snr)
                        
                    final_mix[start_idx:start_idx+len(evt_audio)] += evt_audio
                    multi_hot[ESC50_TO_OURS[event_class]] = 1.0

        # Normalize and apply Random Gain to prevent clipping and improve robustness
        max_val = np.max(np.abs(final_mix))
        if max_val > 0:
            target_peak = random.uniform(0.4, 0.9) # Random Gain augmentation
            final_mix = final_mix * (target_peak / max_val)
            
        return final_mix, multi_hot, stage

    def build_and_extract(self, total_samples: int = 140000, shard_size: int = 1000, batch_size: int = 32):
        print(f"Synthesizing {total_samples} samples across 6 Curriculum Stages with {shard_size} samples per shard (Batch size: {batch_size})...")
        
        # Exact ratios required by ALM v12.7.1
        proportions = {
            1: 0.214, # Stage 1 (30k/140k)
            2: 0.250, # Stage 2 (35k/140k)
            3: 0.214, # Stage 3 (30k/140k)
            4: 0.143, # Stage 4 (20k/140k)
            5: 0.107, # Stage 5 (15k/140k)
            6: 0.072  # Stage 6 (10k/140k)
        }
        
        stage_pool = []
        for stg, prop in proportions.items():
            stage_pool.extend([stg] * int(total_samples * prop))
            
        # Pad to exactly total_samples if rounding caused a mismatch
        while len(stage_pool) < total_samples:
            stage_pool.append(1)
            
        random.shuffle(stage_pool)
        
        shard_idx = 1
        current_embeddings = []
        current_labels = []
        
        # We will iterate through stage_pool in chunks of batch_size
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
            batch_stages = stage_pool[batch_start:min(batch_start + batch_size, total_samples)]
            
            batch_audio = []
            batch_multi_hot = []
            batch_cur_stage = []
            
            for stage in batch_stages:
                audio, multi_hot, cur_stage = self.generate_scene(stage)
                batch_audio.append(audio)
                batch_multi_hot.append(multi_hot)
                batch_cur_stage.append(cur_stage)
                
            # Batch extraction using frozen models
            w_embs = self.whisper_fe.batch_extract(batch_audio, self.target_sr)
            c_embs = self.clap_fe.batch_extract(batch_audio, self.target_sr)
            h_embs = self.htsat_fe.batch_extract(batch_audio, self.target_sr)
            
            for j in range(len(batch_stages)):
                w_emb = w_embs[j].numpy()
                c_emb = c_embs[j].numpy()
                h_emb = h_embs[j].numpy()
                
                current_embeddings.append((w_emb, c_emb, h_emb))
                label_with_stage = np.append(batch_multi_hot[j], [batch_cur_stage[j]])
                current_labels.append(label_with_stage)
                
                global_idx = batch_start + j
                if (global_idx + 1) % shard_size == 0 or (global_idx + 1) == total_samples:
                    emb_path = os.path.join(self.output_dir, f"embeddings_shard_{shard_idx}.npy")
                    lbl_path = os.path.join(self.output_dir, f"labels_shard_{shard_idx}.npy")
                    
                    np.save(emb_path, np.array(current_embeddings, dtype=object))
                    np.save(lbl_path, np.array(current_labels, dtype=object))
                    
                    print(f"Saved Shard {shard_idx} ({len(current_embeddings)} samples) to disk.")
                    
                    current_embeddings = []
                    current_labels = []
                    shard_idx += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ALM v12.8 Curriculum Dataset Builder")
    parser.add_argument("--audio_dir", type=str, default="data/raw", help="Path to raw audio files")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Path to save processed shards")
    parser.add_argument("--num_samples", type=int, default=100000, help="Total number of acoustic scenes")
    parser.add_argument("--shard_size", type=int, default=1000, help="Samples per file shard")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--max_events", type=int, default=6, help="Maximum simultaneous environmental events")
    parser.add_argument("--use_mock_data", action="store_true", help="Allow fallback to mock synthetic data")
    args = parser.parse_args()
    
    builder = UnifiedDatasetBuilder(args.audio_dir, args.output_dir, max_events=args.max_events, use_mock_data=args.use_mock_data)
    builder.build_and_extract(total_samples=args.num_samples, shard_size=args.shard_size, batch_size=args.batch_size)
