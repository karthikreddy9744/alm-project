import torch
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperModel,
    ClapModel,
    ClapProcessor
)

class WhisperFeatureExtractor:
    def __init__(self, model_size='base'):
        model_name = f'openai/whisper-{model_size}'
        from transformers import WhisperForConditionalGeneration
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model.to(self.device)
        
        # Initialize transcription pipeline once using the SAME model instance to save memory
        from transformers import pipeline
        self.transcriber = pipeline(
            'automatic-speech-recognition',
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device=self.device,
            generate_kwargs={
                "task": "transcribe"
            }
        )

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sr: int = 16000):
        transcript = self.transcriber(audio)['text'].strip()
        
        # Extract embeddings using only the encoder
        inputs = self.processor(audio, sampling_rate=sr, return_tensors='pt')
        input_features = inputs.input_features.to(self.device)
        
        with torch.no_grad():
            encoder_output = self.model.get_encoder()(input_features)
        
        # Get encoder last hidden state and mean pool
        encoder_last_hidden = encoder_output.last_hidden_state
        embedding = encoder_last_hidden.mean(dim=1).squeeze(0).cpu() # [512]
        
        return embedding, transcript

class CLAPFeatureExtractor:
    def __init__(self):
        model_name = 'laion/clap-htsat-fused'
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.eval()
        
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