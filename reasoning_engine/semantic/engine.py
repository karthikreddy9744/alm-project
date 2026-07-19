import json
import json_repair
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import ValidationError
from reasoning_engine.semantic.models import SemanticSceneObject
from reasoning_engine.fusion.models import AudioEvidenceObject
from reasoning_engine.semantic.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

class SemanticInterpretationEngine:
    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        logger.info(f"Loading Semantic Engine Model: {self.model_id} (4-bit)")
        try:
            # We use bitsandbytes 4-bit quantization if cuda is available. On Mac MPS, bnb isn't fully supported, so we fallback to float16 or bfloat16.
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                # MPS fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info("Semantic Engine Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Semantic Engine Model: {e}")
            raise e

    def generate_semantics(self, input_data: AudioEvidenceObject) -> SemanticSceneObject:
        user_prompt = build_user_prompt(input_data.model_dump_json(indent=2))
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        try:
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return self._parse_json(response)
        except Exception as e:
            logger.error(f"Error during semantic generation: {e}")
            return self._fallback_interpretation()
            
    def _parse_json(self, response_text: str) -> SemanticSceneObject:
        import re
        # Find the first { and last } to extract the JSON block
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
        else:
            clean_text = response_text.strip()
        
        try:
            data = json_repair.loads(clean_text)
            
            # Handle edge cases where json_repair returns a string (double encoded) or list
            if isinstance(data, str):
                try:
                    data = json_repair.loads(data)
                except:
                    data = {}
            if isinstance(data, list):
                # Find the first dictionary inside the list
                dict_items = [item for item in data if isinstance(item, dict)]
                data = dict_items[0] if dict_items else {}
                
            if not isinstance(data, dict):
                data = {}
                
            try:
                # Try strict Pydantic parse
                return SemanticSceneObject(**data)
            except ValidationError as ve:
                logger.warning(f"Pydantic validation failed, salvaging dictionary... {ve}")
                
                # Salvage Operation: Coerce known list fields if the model hallucinated a string
                for key in ["actors", "human_goals"]:
                    if key in data and isinstance(data[key], str):
                        data[key] = [data[key]]
                        
                # Aggressive Salvage: Scrub hallucinated Enums
                for error in ve.errors():
                    if error.get("type") == "enum" or "Input should be" in error.get("msg", ""):
                        loc = error.get("loc", [])
                        
                        field_name = loc[-1] if loc else ""
                        fallback = "Unknown"
                        if field_name == "dominant_modality":
                            fallback = "Balanced"
                        elif field_name == "agreement_level":
                            fallback = "Low"
                        elif field_name == "verification_status":
                            fallback = "Inconclusive"
                        elif field_name == "relationship_to_hypothesis":
                            fallback = "LowConfidence"
                        elif field_name == "influence":
                            fallback = "Ignored"

                        curr = data
                        for i, key in enumerate(loc):
                            if i == len(loc) - 1:
                                if isinstance(curr, dict) and key in curr:
                                    logger.warning(f"Scrubbing invalid enum at {loc} (was '{curr[key]}'). Forcing '{fallback}'.")
                                    curr[key] = fallback
                                elif isinstance(curr, list) and isinstance(key, int) and key < len(curr):
                                    logger.warning(f"Scrubbing invalid enum in list at {loc}. Forcing '{fallback}'.")
                                    curr[key] = fallback
                            elif isinstance(curr, dict) and key in curr:
                                curr = curr[key]
                            elif isinstance(curr, list) and isinstance(key, int) and key < len(curr):
                                curr = curr[key]
                            else:
                                break
                                
                # Try strict parse again
                try:
                    return SemanticSceneObject(**data)
                except ValidationError as e:
                    logger.warning(f"Second pass Pydantic validation failed: {e}. Returning safe fallback.")
                    return self._fallback_interpretation()
                    
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON: {e}\nRaw output: {response_text}")
            return self._fallback_interpretation()
            
    def _fallback_interpretation(self) -> SemanticSceneObject:
        from reasoning_engine.semantic.models import (
            SpeechUnderstanding, AuditoryObservation, CrossModalAssessment,
            AgreementLevel, VerificationStatus, DominantModality,
            AudioProvenanceReasoning, SourceType, RepresentationType, ProvenanceReliability
        )
        return SemanticSceneObject(
            speech_understanding=SpeechUnderstanding(
                summary="Failed to parse JSON.",
                topic="Unknown",
                speaker_intent="Unknown",
                speaker_role="Unknown",
                emotional_tone="Unknown",
                confidence=0.0
            ),
            auditory_observations=[],
            audio_provenance_reasoning=AudioProvenanceReasoning(
                source_type=SourceType.UNKNOWN,
                representation_type=RepresentationType.UNKNOWN,
                confidence=0.0,
                provenance_reliability=ProvenanceReliability.LOW,
                supporting_evidence=[],
                remaining_uncertainty="JSON parsing failed."
            ),
            cross_modal_assessment=CrossModalAssessment(
                agreement_level=AgreementLevel.LOW,
                verification_status=VerificationStatus.INCONCLUSIVE,
                dominant_modality=DominantModality.BALANCED,
                major_supports=[],
                major_conflicts=[],
                remaining_uncertainty="JSON parsing failed.",
                overall_assessment="System error during semantic parsing."
            ),
            primary_situation="Unknown",
            environmental_context="Unknown",
            actors=[],
            human_goals=[],
            alternative_hypotheses=[],
            missing_evidence=[],
            likely_next_state="Unknown",
            interpretation_confidence=0.1,
            human_oriented_summary="Unknown Situation"
        )
