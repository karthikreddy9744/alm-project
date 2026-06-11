import unittest
import numpy as np
from core.inference_pipeline import ALMInferencePipeline, preprocess_audio_array

from core.context_builder import SCENE_LABELS

class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = ALMInferencePipeline()
        # Create a dummy audio array (2 seconds, 16kHz)
        self.dummy_audio = np.random.randn(32000).astype(np.float32)
        self.sr = 16000
    
    def test_preprocess_audio(self):
        processed = preprocess_audio_array(self.dummy_audio, self.sr)
        self.assertEqual(processed.shape, (32000,))
        # Check normalization
        self.assertLessEqual(np.max(np.abs(processed)), 1.0)
    
    def test_pipeline_run(self):
        transcript, scene_class, confidence, ai_response = self.pipeline.run(self.dummy_audio, self.sr)
        self.assertIsInstance(transcript, str)
        self.assertIn(scene_class, SCENE_LABELS)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(ai_response, str)
        self.assertGreater(len(ai_response), 0)

if __name__ == "__main__":
    unittest.main()
