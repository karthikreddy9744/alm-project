# 07: Implementation

## Execution Flow

1. **Initialization (`main.py`):** The user invokes `main.py` with an audio file path. 
2. **Orchestration (`inference_pipeline.py`):** The `UnifiedPipelineValidator` is spun up to manage the sequential execution.
3. **Perception Execution:** The audio is sent to `core_modules/feature_extractor.py`. Whisper and CLAP models are loaded onto the GPU (or MPS), inference is performed, and weights are immediately offloaded to prevent VRAM overflow.
4. **Data Fusion:** The raw features are passed to `fusion_layer.py` which instantiates the `AudioEvidenceObject` via Pydantic. If validation fails, execution halts.
5. **Logic Sequence:** The `AudioEvidenceObject` is handed sequentially to the `reasoning_engine` directories (`semantic` -> `hre` -> `tre` -> `wse` -> `spe` -> `sir`).
6. **LLM Inference:** Each engine dynamically prompts Qwen3-4B-Instruct, appending the prior JSON states into the prompt context to ensure chronological reasoning.
7. **Final Output:** The `sir` engine yields the final HOASU Markdown report back to `main.py` to be printed or saved.

## Schema Flow
The primary mechanism for preventing hallucination is Schema Flow. The `AudioEvidenceObject` acts as an immutable ledger. Once perception writes the transcript and acoustic classes into the object, the LLM logic engines are structurally forced to reference that object in their JSON responses. They cannot invent new events because they must cite a timestamp from the AEO.
