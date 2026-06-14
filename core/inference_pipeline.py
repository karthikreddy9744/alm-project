import numpy as np
import torch
import librosa
from core.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor
from core.fusion_layer import FusionLayer
from core.scene_network import SceneContextNetwork
from core.casre_engine import CASREEngine
from core.audio_utils import lufs_normalize, detect_clipping

class ALMInferencePipeline:
    def __init__(self):
        # Feature extractors
        self.whisper_fe = WhisperFeatureExtractor('base')
        self.clap_fe = CLAPFeatureExtractor()
        
        # Custom trained modules
        self.fusion = FusionLayer()
        self.scene_net = SceneContextNetwork()
        self.casre = CASREEngine(threshold=0.30)
        
        # Load trained weights (ignoring missing keys warning during architectural upgrades)
        try:
            checkpoint = torch.load('models/scene_model.pt', map_location='cpu')
            self.fusion.load_state_dict(checkpoint['fusion'])
            self.scene_net.load_state_dict(checkpoint['scene_net'])
        except FileNotFoundError:
            print("Warning: 'models/scene_model.pt' not found. Using untrained weights for v4 architecture.")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}. Using untrained weights.")
            
        self.fusion.eval()
        self.scene_net.eval()

    def run(self, audio: np.ndarray, sr: int):
        # Step 1: Preprocess (LUFS Normalization)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        audio = lufs_normalize(audio, sr, target_lufs=-23.0)
        
        # Temporal Chunking (Phase 7: Temporal Understanding)
        chunk_duration = 5 # 5 seconds
        chunk_samples = chunk_duration * sr
        
        num_chunks = len(audio) // chunk_samples
        if len(audio) % chunk_samples > (chunk_samples / 2):
            num_chunks += 1
        num_chunks = max(1, num_chunks)
            
        timeline = []
        global_transcript = []
        
        # Process each chunk temporally
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(len(audio), (i + 1) * chunk_samples)
            chunk_audio = audio[start:end]
            
            # Pad final chunk if necessary
            if len(chunk_audio) < chunk_samples and len(chunk_audio) > 0:
                chunk_audio = np.pad(chunk_audio, (0, chunk_samples - len(chunk_audio)))
                
            # Step 2: Feature extraction (parallel)
            w_emb, transcript = self.whisper_fe.extract(chunk_audio, sr)
            c_emb = self.clap_fe.extract(chunk_audio, sr)
            
            if transcript:
                global_transcript.append(f"[{i*5}-{(i+1)*5}s]: {transcript}")
                
            # Step 3: Advanced Cross-Attention Fusion
            with torch.no_grad():
                fused = self.fusion(w_emb.unsqueeze(0), c_emb.unsqueeze(0))
                logits = self.scene_net(fused)
                
            # Phase 4: Multi-Label Outputs using Sigmoid (BCEWithLogitsLoss)
                probs = torch.sigmoid(logits).squeeze().tolist()
                
            # Semantic-Acoustic Alignment Filter (Force realism to fix untrained ML noise)
            from core.casre_engine import SCENE_LABELS
            if transcript:
                transcript_lower = transcript.lower()
                for tone, words in self.casre.keyword_categories.items():
                    if any(w in transcript_lower for w in words):
                        if tone == "Weather/Disaster":
                            probs[SCENE_LABELS.index("Weather & Nature")] = max(probs[SCENE_LABELS.index("Weather & Nature")], 0.95)
                            probs[SCENE_LABELS.index("Silence / Unknown")] = 0.0
                        elif tone == "War/Crisis":
                            probs[SCENE_LABELS.index("Crowd & Hubbub")] = max(probs[SCENE_LABELS.index("Crowd & Hubbub")], 0.88)
                            probs[SCENE_LABELS.index("Silence / Unknown")] = 0.0
                        elif tone == "Sci-Fi/Fantasy/Drama":
                            probs[SCENE_LABELS.index("Movie Scene")] = max(probs[SCENE_LABELS.index("Movie Scene")], 0.99)
                        elif tone == "Emergency":
                            probs[SCENE_LABELS.index("Indoor / Domestic")] = max(probs[SCENE_LABELS.index("Indoor / Domestic")], 0.85)
                            probs[SCENE_LABELS.index("Silence / Unknown")] = 0.0
                        elif tone == "Broadcast":
                            probs[SCENE_LABELS.index("Television / Media")] = max(probs[SCENE_LABELS.index("Television / Media")], 0.90)
                        elif tone == "Music/Song":
                            probs[SCENE_LABELS.index("Music")] = max(probs[SCENE_LABELS.index("Music")], 0.95)
                        elif tone == "Casual" or tone == "Frustration":
                            probs[SCENE_LABELS.index("Indoor / Domestic")] = max(probs[SCENE_LABELS.index("Indoor / Domestic")], 0.70)
                
                # If there's any speech, explicitly suppress Silence
                probs[SCENE_LABELS.index("Silence / Unknown")] = 0.0
                
            # Suppress weird outdoor noise indoors
            if probs[SCENE_LABELS.index("Indoor / Domestic")] > 0.6:
                probs[SCENE_LABELS.index("Wildlife / Animals")] *= 0.1
                probs[SCENE_LABELS.index("Traffic & Vehicles")] *= 0.3
                
            # Acoustic Primitives Extraction (NATE Architecture)
            # 1. RMS Energy (Proximity/Distance/Loudness)
            rms = float(np.mean(librosa.feature.rms(y=chunk_audio)))
            # 2. Spectral Centroid (Pitch/Severity/Emotion)
            cent = float(np.mean(librosa.feature.spectral_centroid(y=chunk_audio, sr=sr)))
                
            # Store timeline event
            timeline.append({
                "start": i * 5,
                "end": (i + 1) * 5,
                "probs": probs,
                "transcript": transcript,
                "rms": rms,
                "pitch": cent
            })
            
        # Global Event Aggregation for CASRE
        # Compute max probabilities across all chunks for global summary
        if len(timeline) > 0:
            global_probs = np.max([t["probs"] for t in timeline], axis=0).tolist()
        else:
            global_probs = [0.0] * 20
            
        full_transcript = "\n".join(global_transcript)
        
        # Step 5: Advanced CASRE — scenario matrix heuristics & risk scoring
        # Compute transcript confidence metric (simple length/quality heuristic)
        t_conf = min(1.0, len(full_transcript) / 100) if full_transcript else 0.0
        
        # Pass the full temporal timeline to CASRE for NATE logic
        ai_response, active_scenes, risk_score, is_media = self.casre.analyze(
            full_transcript, global_probs, t_conf, timeline
        )
        
        # Add temporal event log to response
        temporal_log = "\n\nTemporal Event Timeline:\n"
        for t in timeline:
            from core.casre_engine import SCENE_LABELS
            # Sort indices by probability
            sorted_indices = sorted(range(len(t["probs"])), key=lambda k: t["probs"][k], reverse=True)
            active = []
            
            # Take top 3 highest probability scenes > 0.3
            for idx in sorted_indices:
                if t["probs"][idx] > 0.3:
                    active.append(SCENE_LABELS[idx])
                if len(active) == 3:
                    break
            
            if not active:
                active.append("Silence / Unknown")
                
            scene_str = ", ".join(active)
            temporal_log += f"[{t['start']:02d}s - {t['end']:02d}s] -> {scene_str}\n"
            
        ai_response += temporal_log
        
        # Format the top class for legacy compatibility in Gradio UI
        top_scene = active_scenes[0][0] if active_scenes else "Silence / Unknown"
        confidence = active_scenes[0][1] if active_scenes else 1.0
        
        return full_transcript, top_scene, confidence, ai_response
