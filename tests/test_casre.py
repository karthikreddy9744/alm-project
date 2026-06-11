import unittest
from core.context_builder import (
    _analyze_transcript,
    _get_confidence_tone,
    _build_speech_context,
    _build_cross_modal_insight,
    generate_response,
    SCENE_LABELS
)

class TestCASRE(unittest.TestCase):
    def test_analyze_transcript_empty(self):
        result = _analyze_transcript("")
        self.assertEqual(result["type"], "empty")
        self.assertEqual(result["urgency"], "none")
    
    def test_analyze_transcript_emergency(self):
        result = _analyze_transcript("Help! Fire!")
        self.assertEqual(result["type"], "distress")
        self.assertEqual(result["urgency"], "high")
    
    def test_analyze_transcript_question(self):
        result = _analyze_transcript("What is that sound?")
        self.assertEqual(result["type"], "question")
        self.assertEqual(result["urgency"], "low")
    
    def test_analyze_transcript_calm(self):
        result = _analyze_transcript("Okay, thanks!")
        self.assertEqual(result["type"], "calm")
        self.assertEqual(result["urgency"], "none")
    
    def test_get_confidence_tone(self):
        self.assertEqual(_get_confidence_tone(0.9), "high")
        self.assertEqual(_get_confidence_tone(0.6), "medium")
        self.assertEqual(_get_confidence_tone(0.3), "low")
    
    def test_build_speech_context(self):
        ctx = _build_speech_context("Hello there", {"type": "calm", "urgency": "none"})
        self.assertIn("Hello there", ctx)
    
    def test_build_cross_modal_insight(self):
        insight = _build_cross_modal_insight("Emergency", {"type": "distress", "urgency": "high"}, 0.95)
        self.assertIn("emergency", insight.lower())
    
    def test_generate_response(self):
        response = generate_response(
            transcript="Help me!",
            scene_class="Emergency",
            confidence=0.9,
            scene_probs=[0.9, 0.05, 0.03, 0.01, 0.01] + [0.0] * 10
        )
        self.assertIn("Emergency", response)
        self.assertIn("recommended action", response.lower())

if __name__ == "__main__":
    unittest.main()
