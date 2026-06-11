import unittest
import numpy as np
from core.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor

class TestFeatureExtractors(unittest.TestCase):
    def setUp(self):
        # Create a dummy audio array (1 second, 16kHz)
        self.dummy_audio = np.random.randn(16000).astype(np.float32)
        self.sr = 16000
    
    def test_whisper_extractor(self):
        extractor = WhisperFeatureExtractor("base")
        embedding, transcript = extractor.extract(self.dummy_audio, self.sr)
        self.assertEqual(embedding.shape, (512,))
        self.assertIsInstance(transcript, str)
    
    def test_clap_extractor(self):
        extractor = CLAPFeatureExtractor()
        embedding = extractor.extract(self.dummy_audio, self.sr)
        self.assertEqual(embedding.shape, (512,))

if __name__ == "__main__":
    unittest.main()
