import logging
import re
from langdetect import detect, DetectorFactory
from transformers import pipeline

# Ensure consistent language detection
DetectorFactory.seed = 0

class MultilingualSpeechNormalizationLayer:
    """
    Multilingual Speech Normalization Layer (MSNL) for ALM v7.0.
    Detects the language of the provided transcript and translates it to English
    using Hugging Face's Helsinki-NLP/opus-mt models if it is non-English.
    """
    def __init__(self):
        self.translation_pipelines = {}
        logging.info("Initializing MSNL...")

    def _get_translation_pipeline(self, src_lang: str):
        if src_lang not in self.translation_pipelines:
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
            try:
                logging.info(f"Loading translation model {model_name}...")
                from transformers import MarianMTModel, MarianTokenizer
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                self.translation_pipelines[src_lang] = (tokenizer, model)
            except Exception as e:
                logging.error(f"Failed to load translation model for {src_lang}: {e}")
                self.translation_pipelines[src_lang] = None
        return self.translation_pipelines.get(src_lang)

    def normalize(self, transcript: str):
        """
        Takes a raw transcript, detects its language, and translates to English if necessary.
        
        Returns:
            dict: Contains 'original_transcript', 'detected_language', 'semantic_transcript' (English), 
                  'translation_confidence', and 'reasoning_language'
        """
        if not transcript or not transcript.strip():
            return {
                "original_transcript": transcript,
                "detected_language": "en",
                "semantic_transcript": transcript,
                "translation_confidence": 1.0,
                "reasoning_language": "en"
            }

        # Remove obvious noise words before detection (optional, but helps langdetect)
        clean_transcript = re.sub(r'\[.*?\]', '', transcript).strip()
        
        if not clean_transcript:
            clean_transcript = transcript

        try:
            detected_lang = detect(clean_transcript)
        except Exception as e:
            logging.warning(f"Language detection failed: {e}. Defaulting to English.")
            detected_lang = "en"

        if detected_lang == "en":
            return {
                "original_transcript": transcript,
                "detected_language": "en",
                "semantic_transcript": transcript,
                "translation_confidence": 1.0,
                "reasoning_language": "en"
            }
            
        # Needs translation
        translator = self._get_translation_pipeline(detected_lang)
        if translator:
            try:
                tokenizer, model = translator
                inputs = tokenizer(transcript, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                return {
                    "original_transcript": transcript,
                    "detected_language": detected_lang,
                    "semantic_transcript": translated_text,
                    "translation_confidence": 0.95, # High confidence if pipeline succeeded
                    "reasoning_language": "en"
                }
            except Exception as e:
                logging.error(f"Translation failed: {e}")
                
        # Fallback if translation fails or model isn't available
        return {
            "original_transcript": transcript,
            "detected_language": detected_lang,
            "semantic_transcript": transcript,
            "translation_confidence": 0.0,
            "reasoning_language": detected_lang
        }
