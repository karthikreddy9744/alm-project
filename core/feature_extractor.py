import numpy as np
from faster_whisper import WhisperModel
import librosa

class WhisperFeatureExtractor:

    def __init__(
        self,
        model_size="base",
        device="cpu",
        compute_type="int8"
    ):

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def extract_features(self, audio_path):

        # ---------- Transcript ----------
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        )

        # ---------- Audio Features ----------
        audio, sr = librosa.load(audio_path,sr=16000,mono=True)
        audio_features = self.model.feature_extractor(audio)

        encoder_output = self.model.encode(audio_features)

        encoder_output = np.array(encoder_output)

        embedding = encoder_output.mean(axis=1).squeeze(0)

        return {
            "language": info.language,
            "transcript": transcript,
            "embedding": embedding
            }


if __name__ == "__main__":

    extractor = WhisperFeatureExtractor()

    result = extractor.extract_features(
        "samples/test.wav"
    )

    print("\nLanguage:")
    print(result["language"])

    print("\nTranscript:")
    print(result["transcript"])

    print("\nEmbedding Shape:")
    print(result["embedding"].shape)

    print("\nFirst 10 Values:")
    print(result["embedding"][:10])