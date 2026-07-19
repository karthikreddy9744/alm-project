# ALM v12.0: Master Technical & Research Specification
> **Status:** Definitive Final Specification
> **Compilation:** Generated via Knowledge Base Synthesis
---

# 01: Project Inventory

## Overview
This document maps the entire ALM v12.0 directory structure, providing explicit purposes and responsibilities for all major files and folders. 

## Folder Tree
```text
alm-project/
├── core_modules/        # [Active] Physical sensory extraction and pipeline execution.
├── reasoning_engine/    # [Active] High-level LLM logic constraints (HRE, TRE, etc.).
├── evaluation/          # [Active] Evaluation datasets and generated statistical results.
├── research/            # [Active] Benchmarking scripts and LaTeX rendering utilities.
├── archive/             # [Archive] Deprecated custom PyTorch weights and early scripts.
├── literature_survey/   # [Active] Markdown summaries of related academic works.
├── datasets/            # [Active] The physical .mp3 and .wav audio files for inference.
├── documentation/       # [Active] Master specifications and this knowledge base.
├── colab_setup.ipynb    # [Active] GPU bootstrapper for L4/A100 compute.
└── main.py              # [Active] Application entry point.
```

## Detailed File Analysis

### `core_modules/`
- **`feature_extractor.py`**: [Critical] Uses `faster-whisper` and `CLAP` to extract raw acoustic properties (transcripts, embeddings). Depends on `torchaudio`.
- **`inference_pipeline.py`**: [Critical] The master orchestrator that receives the `AudioEvidenceObject` and executes the sequential reasoning layers.

### `reasoning_engine/`
- **`fusion/`**: Formats literal audio text and acoustic data into the strict `AudioEvidenceObject` Pydantic schema.
- **`semantic/`**: The Qwen3 Semantic Interpretation Engine.
- **`tre/`**: The Transparent Reasoning Engine for Cross-Modal Verification.
- **`hre/`**: The Hypothesis Reasoning Engine for deducing entities.
- **`wse/`**: The World State Engine for mapping environment.
- **`spe/`**: The Situation Projection Engine for predictive futures.
- **`sir/`**: The Situation Intelligence Renderer for natural language output.

### `research/`
- **`evaluation_runner.py`**: [Critical] Feeds `hoasu_bench.json` into the pipeline automatically. Outputs 6 CSV/JSON artifacts.
- **`statistical_analysis.py`**: [Active] Calculates Fleiss' Kappa and Wilcoxon scores for paper generation.

### Current Status
All zero-shot folders (`core_modules`, `reasoning_engine`, `evaluation`) are active and frozen. All end-to-end deep learning scripts are deprecated and sequestered to `archive/`.


---

# 02: Project History

## ALM v1 - v4 (End-to-End CNNs)
- **Main Idea:** Custom PyTorch CNNs trained locally on the ESC-50 dataset.
- **Why it changed:** Catastrophically brittle.
- **Lessons Learned:** Local, narrow datasets are too small for real-world generalization; deep learning audio classifiers act as black boxes.
- **Reason for Moving On:** Hit the "Explainability Wall".

## ALM v5 - v8 (Audio-LLM Hybrids)
- **Main Idea:** Projecting audio embeddings (CLAP) directly into an LLM's token space.
- **Why it changed:** Severe, uncontrollable hallucinations. 
- **Lessons Learned:** LLMs conflate audio features with visual data from their training distribution (e.g., guessing a speaker's shirt color from their voice).
- **Reason for Moving On:** Hit the "Hallucination Wall".

## ALM v9 - v11 (Hybrid Neuro-Symbolic)
- **Main Idea:** Fusion of Whisper ASR with a custom-trained local `scene_model.pt`.
- **Why it changed:** The local scene model became a massive computational and linguistic bottleneck.
- **Lessons Learned:** Custom training cannot compete with massive foundation models for semantic intelligence.
- **Reason for Moving On:** Hit the "Compute Wall".

## ALM v12 (Zero-Shot Structured Reasoning - FINAL)
- **Main Idea:** Total deprecation of custom models. 100% reliance on Whisper, CLAP, and Qwen3 chained sequentially via strict JSON schemas (`AudioEvidenceObject`).
- **Why it became final:** Architecturally sound, highly explainable, entirely zero-shot, and resolves all hallucination issues through Schema-Constrained Reasoning.


---

# 03: Project Philosophy

## Vision
To pioneer a transparent, neuro-symbolic standard for machine listening that replaces opaque black-box deep learning classification with auditable, deductive, evidence-based reasoning architectures.

## Motivation
Modern AI is plagued by the "black-box" problem. In high-stakes environments, systems that rely on end-to-end deep learning frequently hallucinate context when presented with ambiguous data. Furthermore, they lack **Provenance Reasoning**—the ability to distinguish between the physical occurrence of a sound and a media representation of it (e.g., a real explosion vs. a movie explosion).

## Problem Statement
Current audio systems treat speech and environmental sounds as isolated domains, mapping raw waveforms to literal text without understanding context. There is a profound absence of architectures capable of interpreting audio streams with the contextual awareness, temporal logic, and provenance differentiation inherent to human cognition.

## Research Gap
Existing foundation models (Whisper, CLAP) handle perception flawlessly, but lack structured semantic interpretation. Current Audio-LLM systems fail to evaluate provenance, resolve cross-modal contradictions, and generate empathetic summaries.

## Objectives
- Achieve Acoustic-Semantic Fusion.
- Replace black-box classification with transparent JSON logic chains.
- Implement explicit Probabilistic Provenance Awareness.
- Eradicate hallucinations through schema-constrained logic.

## Scope
- High-fidelity audio processing (Live, Broadcast, Media).
- Zero-shot inference without fine-tuning.
- Multi-modal fusion.
- Desktop (MPS) and Cloud GPU (CUDA) execution.

## Out of Scope
- End-to-end neural weight training.
- Cryptographic deepfake digital forensics.
- Real-time ultra-low-latency streaming.

## Research & Design Principles
1. **Evidence Dominates Assumptions:** ALM is explicitly forbidden from assuming unproven visual or situational contexts not verified by acoustic or transcript evidence.
2. **Reasoning State Exposure:** 8 explicit states of logic are serialized to disk to ensure 100% auditability.
3. **Human-Oriented Auditory Situation Understanding (HOASU):** Machine intelligence must be translated into empathetic, jargon-free narratives for human operators.


---

# 04: Literature Foundation

## Influencing Literature

### 1. Sci-Phi (Scientific Philosophy in AI)
- **Influence:** Heavily inspired ALM's transition away from End-to-End deep learning toward Neuro-Symbolic logic. Sci-Phi proved that structured symbolic logic (schemas) applied to neural outputs drastically reduces hallucinations.
- **Adopted:** The concept of explicit intermediate logic verification layers.

### 2. SLAM-LLM
- **Influence:** An industry standard for injecting audio embeddings directly into an LLM.
- **Adopted:** Validated the use of CLAP embeddings for environmental understanding.
- **Rejected:** SLAM-LLM's core thesis—mapping embeddings directly into token space—was ultimately rejected by ALM due to its inability to produce a transparent logic trace. ALM instead opted for the `AudioEvidenceObject` middle-ground.

### 3. "Can We Trust AI With Our Ears?"
- **Influence:** This foundational survey on auditory hallucinations highlighted the critical lack of "Provenance Reasoning" in modern classifiers.
- **Adopted:** ALM directly addresses this gap by implementing the `tre` (Transparent Reasoning Engine) specifically tasked with Cross-Modal Verification and Provenance deduction.

## Research Positioning
ALM positions itself at the intersection of **Machine Listening** and **Explainable AI (XAI)**. It is not competing to be the fastest acoustic event detector; it is competing to be the most cognitively robust and transparent auditory reasoning engine. 

## Current Novelty
The primary novelty lies in ALM's **Schema-Constrained Provenance Reasoning**. While models like Whisper transcribe speech, and models like HTS-AT tag sounds, ALM is the first architecture to explicitly fuse them, cross-reference them for contradictions (e.g., calm speech overlapping with sirens = Media/Synthetic), and serialize the logic.


---

# 05: Repository Analysis

## Deep Dive: Repository Folders

### `core_modules/`
- **Purpose:** The physical sensory organs and pipeline orchestration.
- **Responsibilities:** Extracts literal transcripts and acoustic arrays via Whisper and CLAP/HTS-AT.
- **Interactions:** Exclusively feeds data UP to the `fusion_layer`, never accepts logic DOWN.
- **Importance:** High. Any failure in extraction kills the entire pipeline.
- **Future Maintenance:** Must be updated when new foundation models (like Whisper-v4) release.

### `reasoning_engine/`
- **Purpose:** The brain of ALM. Houses the LLM prompts and strict schemas.
- **Responsibilities:** Executes the sequential engine logic (SIE, HRE, TRE, WSE, SPE, SIR).
- **Interactions:** Receives the `AudioEvidenceObject`, mutates the state JSON, passes to next engine.
- **Importance:** Critical. This is where 100% of the project's intellectual novelty lives.

### `research/`
- **Purpose:** Automation and academic output generation.
- **Responsibilities:** Executes `evaluation_runner.py` across batches of audio and computes Fleiss' Kappa.
- **Outputs:** Generates `.csv` statistical tables for papers.

### `evaluation/`
- **Purpose:** Standardized benchmarking storage.
- **Responsibilities:** Houses `hoasu_bench.json` (the golden evaluation dataset) and the `results/` output directory.

### `archive/`
- **Purpose:** Project traceability and historical preservation.
- **Responsibilities:** Sequestering legacy `.pt` weights and obsolete PyTorch CNN scripts from execution paths.

### `datasets/` & `samples/`
- **Purpose:** Physical storage of `.wav` and `.mp3` files injected into the pipeline during evaluation.


---

# 06: Architecture

## Entire Architecture Flow
ALM v12.0 is a strictly linear, neuro-symbolic pipeline. Raw audio enters the Neural Perception Layer, is serialized into an `AudioEvidenceObject`, and then passes through 6 sequential Large Language Model (Qwen3) Logic Engines.

## Core Modules & Schemas

### 1. Neural Perception (`core_modules/feature_extractor.py`)
- **Inputs:** Audio waveform.
- **Outputs:** Text Transcript, 512-dim Acoustic Embedding.
- **Dependencies:** Whisper Large-v3, CLAP.

### 2. Evidence Fusion (`reasoning_engine/fusion`)
- **Purpose:** Validates the raw outputs and casts them into the Pydantic schema.
- **Output Schema:** `AudioEvidenceObject` (Strict JSON).

### 3. Semantic Interpretation Engine (`reasoning_engine/semantic`)
- **Purpose:** Analyzes the literal transcription for intent, tone, language, and identifies narrator vs. participant dynamics.
- **Input:** `AudioEvidenceObject`
- **Output:** `SemanticState` JSON.

### 4. Hypothesis Reasoning Engine (`reasoning_engine/hre`)
- **Purpose:** Generates the initial situational baseline (Who is present? What are they doing?).
- **Input:** `SemanticState`
- **Output:** `HypothesisState` JSON.

### 5. Transparent Reasoning Engine (`reasoning_engine/tre`)
- **Purpose:** Executes Cross-Modal Verification and Audio Provenance Reasoning. Detects contradictions.
- **Input:** `HypothesisState`
- **Output:** `ProvenanceState` JSON (Live, Media, Synthetic, Broadcast).

### 6. World State Engine (`reasoning_engine/wse`)
- **Purpose:** Deduces the macro-environment using acoustic metadata (e.g., reverb = indoors).
- **Input:** `ProvenanceState`
- **Output:** `WorldState` JSON.

### 7. Situation Projection Engine (`reasoning_engine/spe`)
- **Purpose:** Predicts immediate future developments based on the World State.
- **Input:** `WorldState`
- **Output:** `ProjectionState` JSON.

### 8. Situation Intelligence Renderer (`reasoning_engine/sir`)
- **Purpose:** Formats the 7 previous JSON states into a cohesive, human-empathetic Markdown report (HOASU).


---

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


---

# 08: Research

## Research Methodology
The ALM project follows an Evaluation-First research methodology. Architectural modules are not built in a vacuum; they are designed specifically to solve failure modes identified during formal benchmarking.

## Evaluation Strategy
Due to the qualitative, logic-driven nature of ALM, standard loss metrics (e.g., Cross-Entropy) are irrelevant. The evaluation strategy relies on:
1. **Procedural Benchmarking:** Running the pipeline automatically over hundreds of complex scenarios defined in `hoasu_bench.json`.
2. **Human Evaluation:** Using domain experts to grade the final HOASU outputs on a scale of 1-5 for accuracy, empathy, and logic consistency.
3. **Statistical Aggregation:** Calculating Fleiss' Kappa for human inter-rater reliability.

## Experimental Design & Ablations
To scientifically prove the necessity of ALM's massive architecture, experiments are run against baselines.
- **Baseline 1:** Raw Whisper + LLM (No CLAP, No strict schemas).
- **Baseline 2:** ALM without the Transparent Reasoning Engine (TRE).
- **ALM Full:** The complete v12.0 pipeline.
By ablating the TRE, researchers can mathematically prove the pipeline's inability to detect deepfakes or resolve cross-modal contradictions without explicit symbolic layers.

## Publication Strategy
The research is targeted at high-tier journals. The methodology section relies heavily on the transparency granted by "Reasoning State Exposure," allowing reviewers to audit the exact JSON traces produced by the model during the evaluation phase.


---

# 09: Datasets

## Overview of Datasets
ALM does not use datasets for neural weight training. Datasets are exclusively used for evaluation, benchmarking, and failure analysis. 

### 1. The Legacy Training Dataset (ESC-50)
- **Purpose:** Used in ALM v1 - v4 for training custom PyTorch CNNs.
- **Status:** Deprecated and abandoned. 
- **Why it failed:** It only contains 50 narrow classes of environmental sounds, completely failing to capture the complexity of real-world acoustic scenes and linguistic overlaps.

### 2. The Final Evaluation Dataset (HOASU-Bench)
- **Purpose:** The definitive dataset for evaluating ALM v12.0.
- **Creation:** Procedurally generated and curated to test specific logic traps.
- **Expected Structure:** Controlled `.wav` and `.mp3` files mapped via `hoasu_bench.json`.
- **Ground Truth:** `ground_truth_template.json` outlines exactly what the ALM pipeline *should* deduce for each file (e.g., expected Provenance, expected Entities).

### 3. Future Dataset Extensions
Future datasets (slated for ALM v13) will focus heavily on cryptographic deepfake samples and live-stream acoustic injection to test the real-time latency thresholds of the pipeline.


---

# 10: Evaluation

## The Evaluation Pipeline
The evaluation pipeline is entirely automated via `research/evaluation_runner.py`. It requires Google Colab (L4/A100 GPUs) for batch execution due to the massive VRAM footprint of running multiple LLM inferences per audio file.

## Execution and CSV Generation
1. The runner loads `hoasu_bench.json`.
2. It feeds each audio file into `inference_pipeline.py`.
3. It intercepts the intermediate JSON logic traces (Reasoning State Exposure).
4. It computes the execution latency for each stage.
5. It writes exactly 6 scientific artifacts, including `latency_report.csv`, `evaluation_results.csv`, and `execution_log.md`.

## Metrics and Human Evaluation
Because ALM generates qualitative intelligence reports, standard automated metrics (like BLEU or WER) are insufficient. 
- **Primary Metric:** Human Evaluation. Domain experts review the generated reports against the ground truth.
- **Statistical Analysis:** The `statistical_analysis.py` script computes Fleiss' Kappa to determine inter-rater reliability among the human graders, and Wilcoxon Signed-Rank tests to prove ALM's superiority over the Whisper-Only baseline.

## Future Evaluation Goals
Expanding the benchmark to include 1,000+ adversarial samples specifically designed to trick the Semantic Interpretation Engine.


---

# 11: Execution

## Hardware Requirements and Routing
ALM requires significant computational horsepower. 

### Local Execution (MacBook / Apple Silicon)
- **Use Case:** Code development, schema testing, and single-file mock evaluation.
- **Backend:** MPS (Metal Performance Shaders).
- **Limitation:** MPS does not support `int8_float16` quantization required by `faster-whisper`, meaning execution is extremely slow and memory-intensive. `compute_precision` must fall back to `"auto"` or `"float32"`.

### Production Execution (Google Colab / Cloud GPU)
- **Use Case:** Full 250-sample HOASU-Bench evaluation and scientific CSV generation.
- **Backend:** CUDA (L4 or A100 GPU).
- **Advantage:** Native support for `float16` and flash-attention, reducing inference times from minutes to seconds.
- **Required Commands:** Executing the `colab_setup.ipynb` notebook handles environment setup, pip installs, and GitHub cloning automatically.

## Expected Outputs and Logging
Running `python main.py samples/test_audio.wav` will stream logging directly to the console. The user will see:
1. `[INFO] Neural Perception... Complete.`
2. `[INFO] Validating AudioEvidenceObject... Passed.`
3. `[INFO] Executing WSE...`
Finally, the HOASU Markdown report is printed to standard out and saved to the disk.

## Common Errors
- **CUDA OOM (Out of Memory):** Occurs if Whisper and Qwen are loaded simultaneously without proper offloading. **Solution:** Ensure `del model` and `torch.cuda.empty_cache()` are called sequentially in the pipeline.


---

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


---

# 13: Publication

## Target Journals
The final ALM v12.0 architecture and methodology is targeted for submission to:
1. **IEEE Transactions on Audio, Speech, and Language Processing**
2. **Elsevier Artificial Intelligence**

## Research Contribution Outline
To pass peer-review, the paper will explicitly delineate contributions:
- **Algorithmic:** Forcing multi-modal LLM reasoning through the strict `AudioEvidenceObject` schema.
- **Scientific:** Formalizing Provenance Reasoning (distinguishing live audio from synthetic/media).
- **Evaluation:** Introduction of the `hoasu_bench.json` dataset as a superior metric over ESC-50.

## Limitations and Future Work
- **Limitations:** ALM currently struggles with 20+ speaker overlaps (cocktail party problem) due to Whisper's diarization limitations. Real-time streaming is currently impossible due to the sequential LLM inference latency.
- **Future Work (ALM v13):** Integration of Cryptographic Digital Forensics layers for hard deepfake detection, bypassing purely semantic inference. 

## Patent Discussion
Due to the use of MIT/Apache licensed open-source foundation models (Whisper, Qwen, CLAP), the core perceptual logic is unpatentable. However, the specific Neuro-Symbolic architectural pipeline and schema enforcement mechanisms may be considered for defensive publication.


---

# 14: Appendix

## Glossary of Terms
- **HOASU:** Human-Oriented Auditory Situation Understanding.
- **AEO:** Audio Evidence Object. The central data schema bridging perception and cognition.
- **Provenance:** The representational nature of the audio (Live, Broadcast, Media, Synthetic).
- **Neuro-Symbolic:** A hybrid AI approach combining neural networks (Perception) with explicit logic constraints (Reasoning Engines).
- **Reasoning State Exposure:** The methodology of serializing intermediate logic conclusions to disk for auditability.

## Version History
- **v1 - v4:** E2E CNNs.
- **v5 - v8:** Audio-LLM parameter projection.
- **v9 - v11:** Hybrid PyTorch Scene models + Whisper.
- **v12.0:** Final Zero-Shot Cognitive Pipeline.

## Directory Reference Map
```text
alm-project/
├── core_modules/        # Neural perception and pipeline execution
├── reasoning_engine/    # Logic modules (HRE, TRE, WSE, SPE, SIR, Semantic)
├── evaluation/          # Final datasets and generated CSV results
├── research/            # Evaluation scripts and ablation definitions
├── archive/             # Cold-storage for legacy .pt files
├── literature_survey/   # Markdown analysis of competing models
├── datasets/            # Physical .mp3 and .wav audio files
├── documentation/       # Master specifications
└── colab_setup.ipynb    # GPU execution environment bootstrapper
```


---
