# 12: Project Decisions

## Why Whisper?
OpenAI's Whisper (Large-v3) is universally recognized as the most robust zero-shot ASR model available. Its multi-lingual capabilities and timestamp accuracy are required for the `AudioEvidenceObject`.

## Why Qwen?
Qwen3-4B-Instruct provides the perfect balance of semantic logic capability and VRAM efficiency. Massive 70B models cannot run locally, and smaller 1B models lack the intelligence required for complex Provenance deduction.

## Why CLAP / HTS-AT?
Instead of training a custom sound classifier for thousands of arbitrary labels, CLAP provides a zero-shot textual embedding space, allowing the pipeline to match sounds dynamically to semantic descriptions.

## Why Schemas (Pydantic)?
LLMs hallucinate structural formats. If the pipeline relies on JSON passing between engines, a missing comma crashes the execution. Pydantic enforces strict structural compliance, acting as the logic constraint firewall.

## Why Zero-Shot (No Custom Training / No `.pt`)?
Attempting to fine-tune massive foundation models on local datasets inevitably causes catastrophic forgetting. The models lose their vast, generalized knowledge. By freezing the models and guiding them with schemas, ALM leverages their maximum potential.

## Why Google Colab?
The Mac MPS backend is structurally incapable of the low-precision compute required to run ALM's pipeline efficiently. Colab provides free/cheap access to L4 GPUs which handle the CUDA workloads natively.
