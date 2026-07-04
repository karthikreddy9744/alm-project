import logging
import re
from langdetect import detect, DetectorFactory

# Ensure consistent language detection
DetectorFactory.seed = 0

class MultilingualSpeechNormalizationLayer:
    """
    Multilingual Speech Normalization Layer (MSNL) for ALM v7.0.
    Since Whisper is now handling both transcription AND native translation (task='translate'),
    this layer acts as a semantic pass-through, preserving the detected language metadata
    extracted natively from the Whisper pipeline.
    """
    def __init__(self):
        logging.info("Initializing MSNL...")

    def normalize(self, transcript: str, detected_lang: str = "en"):
        """
        Takes a raw transcript (which is already translated to English by Whisper), 
        and the language detected natively by Whisper.
        
        Returns:
            dict: Contains 'original_transcript' (English translation), 'detected_language' (Source lang code), 
                  'semantic_transcript' (English), 'translation_confidence', and 'reasoning_language'
        """
        if not transcript or not transcript.strip():
            return {
                "original_transcript": transcript,
                "detected_language": "en",
                "semantic_transcript": transcript,
                "translation_confidence": 1.0,
                "reasoning_language": "en"
            }

        return {
            "original_transcript": transcript,
            "detected_language": detected_lang,
            "semantic_transcript": transcript,
            "translation_confidence": 1.0 if detected_lang == "en" else 0.95,
            "reasoning_language": "en"
        }
