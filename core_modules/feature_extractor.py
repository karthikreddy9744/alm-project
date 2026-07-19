import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)
from transformers import (
    WhisperProcessor,
    WhisperModel,
    ClapModel,
    ClapProcessor
)
from core_modules.audio_utils import SileroVADWrapper

class WhisperFeatureExtractor:
    def __init__(self, model_size='base'):
        model_name = f'openai/whisper-{model_size}'
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        
        # Explicitly freeze Whisper
        for param in self.model.parameters():
            param.requires_grad = False
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            device_str = "auto"
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model.to(self.device)
        
        # Initialize Dual-Whisper Transcription Engine (Faster-Whisper Large-v3-Turbo)
        from faster_whisper import WhisperModel as FasterWhisperModel
        compute_precision = "int8_float16" if device_str == "cuda" else "default"
        self.transcriber = FasterWhisperModel(
            "large-v3", 
            device=device_str, 
            compute_type=compute_precision
        )
        logger.info(f"Faster-Whisper initialized with model='large-v3', device='{device_str}', compute_type='{compute_precision}'")
        
        # Initialize Silero VAD
        self.vad = SileroVADWrapper(threshold=0.3)

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000, extract_text: bool = True, extract_emb: bool = True):
        # 1. Voice Activity Detection (VAD)
        timestamps = self.vad.get_timestamps(audio, sr)
        
        transcript_parts = []
        detected_languages = []
        # 2. VAD-Guided Transcription with Faster-Whisper
        if extract_text and len(timestamps) > 0:
            for t in timestamps:
                start_sample = t['start']
                end_sample = t['end']
                
                if (end_sample - start_sample) / sr > 0.2:
                    speech_chunk = audio[start_sample:end_sample]
                    segments, info = self.transcriber.transcribe(
                        speech_chunk,
                        beam_size=5,
                        condition_on_previous_text=False,
                        task="translate"
                    )
                    
                    text = " ".join([s.text.strip() for s in segments]).strip()
                    if info.language:
                        detected_languages.append(info.language)
                    
                    # 3. Hallucination Post-Filtering
                    if text:
                        words = text.lower().split()
                        if len(words) > 0:
                            unique_words = set(words)
                            rep_ratio = len(unique_words) / len(words)
                            # Suppress pathological loops
                            if rep_ratio > 0.3 or len(words) < 5:
                                # Truncate absurdly long continuous spam
                                if len(text) > 200 and rep_ratio < 0.5:
                                    text = text[:200] + "..."
                                transcript_parts.append(text)
        
        transcript = " ".join(transcript_parts) if extract_text else ""
        
        # 4. Extract Acoustic Embeddings using Transformers (Preserves FusionLayer 512-dim compatibility)
        embedding = None
        if extract_emb:
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            encoder_output = self.model.model.encoder(input_features)
            encoder_last_hidden = encoder_output.last_hidden_state
            embedding = encoder_last_hidden.mean(dim=1).squeeze(0).cpu() # [512]
        
        dominant_lang = "en"
        if detected_languages:
            dominant_lang = max(set(detected_languages), key=detected_languages.count)
        
        return embedding, transcript, dominant_lang, timestamps

    @torch.inference_mode()
    def batch_extract(self, audio_list: list, sr: int = 16000):
        """Batch extraction of acoustic embeddings only (disables VAD/text for speed)"""
        padded_audio = []
        for a in audio_list:
            if len(a) < 480000:
                padded_audio.append(np.pad(a, (0, 480000 - len(a))))
            else:
                padded_audio.append(a[:480000])
        inputs = self.processor(padded_audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        encoder_output = self.model.model.encoder(input_features)
        embeddings = encoder_output.last_hidden_state.mean(dim=1).cpu() # [batch, 512]
        return embeddings
class CLAPFeatureExtractor:
    def __init__(self):
        model_name = 'laion/clap-htsat-fused'
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.eval()
        
        # Explicitly freeze CLAP
        for param in self.model.parameters():
            param.requires_grad = False
            
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model.to(self.device)

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000):
        if sr != 48000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
        inputs = self.processor(audio=audio, sampling_rate=sr, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.get_audio_features(**inputs)
            if hasattr(output, 'pooler_output'):
                embedding = output.pooler_output
            elif isinstance(output, tuple):
                embedding = output[0]
            else:
                embedding = output
        
        return embedding.squeeze(0).cpu() # [512]

    @torch.inference_mode()
    def get_nearest_concepts(self, audio: np.ndarray, sr: int, concepts: list) -> dict:
        """Returns similarity scores for a list of concepts."""
        if sr != 48000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
            
        inputs = self.processor(text=concepts, audio=audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            audio_embeds = outputs.audio_embeds
            text_embeds = outputs.text_embeds
            
            audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Compute cosine similarity between audio and text embeddings
            cosine_sim = torch.matmul(audio_embeds, text_embeds.t()).squeeze(0).cpu().numpy()
            
            # Apply Softmax Temperature Calibration to suppress long-tail hallucinations
            temperature = 0.15
            scaled_sim = cosine_sim / temperature
            # Stable softmax
            max_sim = np.max(scaled_sim)
            exp_sim = np.exp(scaled_sim - max_sim)
            softmax_probs = exp_sim / np.sum(exp_sim)
            
        return {concept: float(prob) for concept, prob in zip(concepts, softmax_probs)}

    @torch.inference_mode()
    def batch_extract(self, audio_list: list, sr: int = 16000):
        if sr != 48000:
            import librosa
            audio_list = [librosa.resample(a, orig_sr=sr, target_sr=48000) for a in audio_list]
            sr = 48000
        
        # Determine max length for padding within the processor
        inputs = self.processor(audio=audio_list, sampling_rate=sr, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model.get_audio_features(**inputs)
            if hasattr(output, 'pooler_output'):
                embeddings = output.pooler_output
            elif isinstance(output, tuple):
                embeddings = output[0]
            else:
                embeddings = output
                
        return embeddings.cpu() # [batch, 512]

class HTSATFeatureExtractor:
    """
    HTS-AT (Hierarchical Token-Semantic Audio Transformer)
    Used for high-resolution Polyphonic Event Detection (AudioSet).
    """
    def __init__(self):
        # We use AST as the functional equivalent available in HuggingFace 
        # for AudioSet polyphonic event detection.
        from transformers import ASTModel, ASTFeatureExtractor
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        
        self.processor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)
        self.model.eval()
        
        # Explicitly freeze HTS-AT
        for param in self.model.parameters():
            param.requires_grad = False
            
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model.to(self.device)

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000):
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Pool across time to get a single vector per audio chunk [768]
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            
        return embedding

    @torch.inference_mode()
    def batch_extract(self, audio_list: list, sr: int = 16000):
        if sr != 16000:
            import librosa
            audio_list = [librosa.resample(a, orig_sr=sr, target_sr=16000) for a in audio_list]
            sr = 16000
            
        inputs = self.processor(audio_list, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu() # [batch, 768]
            
        return embeddings