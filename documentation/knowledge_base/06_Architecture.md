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
