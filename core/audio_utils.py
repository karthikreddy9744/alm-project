import numpy as np
import pyloudnorm as pyln
import torch
import warnings

def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """Detect if the audio is clipped (too loud)."""
    return np.max(np.abs(audio)) >= threshold

def lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Normalize audio to target LUFS."""
    try:
        # Pyloudnorm expects (samples, channels) or (samples,)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        
        if np.isinf(loudness):
            # If silence, return original
            return audio
            
        normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        return normalized_audio
    except Exception as e:
        warnings.warn(f"LUFS normalization failed: {e}. Falling back to peak normalization.")
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak
        return audio

class SileroVADWrapper:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Silero VAD from torch hub
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          trust_repo=True)
        self.model = self.model.to(self.device)
        self.get_speech_timestamps = utils[0]

    def get_timestamps(self, audio: np.ndarray, sr: int = 16000):
        """Returns list of dicts: [{'start': int, 'end': int}] in frames."""
        tensor_audio = torch.FloatTensor(audio).to(self.device)
        speech_timestamps = self.get_speech_timestamps(
            tensor_audio, 
            self.model, 
            sampling_rate=sr,
            threshold=self.threshold
        )
        return speech_timestamps
