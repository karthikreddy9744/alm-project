import torch
import numpy as np
import librosa
from core_modules.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor, HTSATFeatureExtractor
from core_modules.fusion_layer import FusionLayer
from core_modules.scene_network import SceneContextNetwork, SCENE_LABELS
from reasoning_engine.awm.world_model import AuditoryWorldModel
from reasoning_engine.awm.models import EntityNode, EventNode, NodeState, Trajectory, HierarchicalConfidence

class ALMInferencePipeline:
    """
    ALM v10.7 Unified Inference Pipeline.
    Bridges the Neural Perception Layer (Whisper/CLAP/HTS-AT + Fusion) 
    with the Deterministic Cognitive Graph (AWM).
    """
    def __init__(self, model_path="models/alm_v10_final.pt"):
        print("Initializing Neural Perception Extractors...")
        self.whisper_fe = WhisperFeatureExtractor('base')
        self.clap_fe = CLAPFeatureExtractor()
        self.htsat_fe = HTSATFeatureExtractor()
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print("Loading Unified Fusion Architecture...")
        self.fusion = FusionLayer().to(self.device)
        self.scene_net = SceneContextNetwork(num_classes=40).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.fusion.load_state_dict(checkpoint['fusion'])
            self.scene_net.load_state_dict(checkpoint['scene_net'])
        except Exception as e:
            print(f"Warning: Could not load trained weights from {model_path}. Using untrained initialization. Error: {e}")
            
        self.fusion.eval()
        self.scene_net.eval()
        
    def process(self, audio: np.ndarray, sr: int, awm: AuditoryWorldModel) -> AuditoryWorldModel:
        """
        Runs the neural perception pipeline and populates the given AuditoryWorldModel.
        """
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # 1. Feature Extraction
        w_emb, transcript, dominant_lang = self.whisper_fe.extract(audio, sr, extract_text=True)
        c_emb = self.clap_fe.extract(audio, sr)
        h_emb = self.htsat_fe.extract(audio, sr)
        
        # 2. Neural Fusion & Scene Context
        with torch.no_grad():
            w_emb = w_emb.to(self.device)
            c_emb = c_emb.to(self.device)
            h_emb = h_emb.to(self.device)
            fused = self.fusion(w_emb, c_emb, h_emb)
            logits = self.scene_net(fused)
            probs = torch.sigmoid(logits).squeeze().tolist()
            
        # 3. Deterministic AWM Population (prob > 0.5 threshold)
        active_scenes = []
        for idx, prob in enumerate(probs):
            if prob > 0.5:
                active_scenes.append((SCENE_LABELS[idx], prob))
                
        # 4. Insert into World Model
        # Insert Speech Entity if present
        if transcript:
            speaker_conf = HierarchicalConfidence(speech_recognition=0.9, sound_detection=1.0)
            speaker = EntityNode(
                id="spk_1",
                entity_type="Speaker",
                state=NodeState.UNKNOWN,
                confidence=speaker_conf
            )
            # Add attributes manually if we need to track transcript inside AWM
            # Though strictly not part of the base EntityNode fields, Python objects allow it
            speaker.transcript = transcript 
            speaker.language = dominant_lang
            awm.add_entity(speaker)
            
            speech_event_conf = HierarchicalConfidence(sound_detection=1.0)
            speech_event = EventNode(
                id="evt_speech_1",
                class_map="Speech Activity",
                trajectory=Trajectory.UNKNOWN,
                acoustic_salience=0.8,
                confidence=speech_event_conf
            )
            awm.add_event(speech_event)
            
        # Insert Environmental Events
        for i, (scene_label, conf) in enumerate(active_scenes):
            env_event_conf = HierarchicalConfidence(sound_detection=conf)
            env_event = EventNode(
                id=f"evt_env_{i}",
                class_map=scene_label,
                trajectory=Trajectory.UNKNOWN,
                acoustic_salience=min(1.0, conf + 0.1),
                confidence=env_event_conf
            )
            awm.add_event(env_event)
            
        return awm
