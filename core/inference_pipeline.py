# core/inference_pipeline.py — ALM v2.0
import numpy as np
import torch
import librosa
from core.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor
from core.fusion_layer import FusionLayer
from core.scene_network import SceneContextNetwork
from core.context_builder import generate_response, SCENE_LABELS

def preprocess_audio_array(audio: np.ndarray, sr: int,
                           target_sr: int = 16000,
                           max_seconds: int = 60) -> np.ndarray:
    # TRUNCATE FIRST to prevent DoS attacks via massive file uploads
    max_original_len = sr * max_seconds
    if len(audio) > max_original_len:
        audio = audio[:max_original_len]
        
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Enforce exact max length on target sample rate
    max_len = target_sr * max_seconds
    audio = audio[:max_len]
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

class ALMInferencePipeline:
    def __init__(self):
        # Feature extractors
        self.whisper_fe = WhisperFeatureExtractor('base')
        self.clap_fe = CLAPFeatureExtractor()
        
        # Custom trained modules
        self.fusion = FusionLayer()
        self.scene_net = SceneContextNetwork()
        
        # Load trained weights
        try:
            checkpoint = torch.load('models/scene_model.pt', map_location='cpu')
            self.fusion.load_state_dict(checkpoint['fusion'])
            self.scene_net.load_state_dict(checkpoint['scene_net'])
        except FileNotFoundError:
            print("Warning: 'models/scene_model.pt' not found. Using untrained weights.")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}. Using untrained weights.")
            
        self.fusion.eval()
        self.scene_net.eval()
        # NOTE: No LLM loaded — CASRE requires zero model loading

    def _sanitize_transcript(self, transcript: str, audio: np.ndarray):
        """Removes hallucinations and scores transcript quality based on audio energy."""
        # 1. Silence Detection (RMS Energy)
        rms_energy = np.sqrt(np.mean(audio**2))
        if rms_energy < 0.001:  # Absolute silence or static
            return "", 0.0
            
        words = transcript.lower().split()
        if not words:
            return "", 0.0
            
        # Detect repetition hallucination (e.g. "oh oh oh oh")
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        # If highly repetitive or mostly non-speech sounds
        if len(words) > 4 and repetition_ratio < 0.3:
            return "", 0.1 # Suppress hallucination
            
        quality_score = max(0.1, min(1.0, repetition_ratio + 0.2))
        return transcript, quality_score

    def run(self, audio: np.ndarray, sr: int):
        # Step 1: Preprocess
        audio = preprocess_audio_array(audio, sr)
        
        # Step 2: Feature extraction (parallel)
        w_emb, transcript = self.whisper_fe.extract(audio)
        c_emb = self.clap_fe.extract(audio)
        
        # Step 3: Fusion
        with torch.no_grad():
            fused = self.fusion(w_emb.unsqueeze(0), c_emb.unsqueeze(0))
            logits = self.scene_net(fused)
            probs = torch.softmax(logits, dim=-1).squeeze()
            
        # Step 4: Scene classification
        scene_idx = probs.argmax().item()
        confidence = probs[scene_idx].item()
        scene_class = SCENE_LABELS[scene_idx]
        
        # Threshold enforcement (e.g., if highest probability is less than 30%)
        if confidence < 0.30:
            scene_class = 'Silence/Unknown'
            
        # Transcript sanitization and quality scoring
        transcript, t_conf = self._sanitize_transcript(transcript, audio)
        
        # Step 5: CASRE — natural language understanding
        ai_response = generate_response(
            transcript, scene_class, confidence, probs.tolist(), t_conf
        )
        
        return transcript, scene_class, confidence, ai_response
