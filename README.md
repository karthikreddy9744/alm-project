# ALM v10.8 (Acoustic Language Model)

ALM v10.8 represents a major breakthrough in audio artificial intelligence. It introduces a **Unified Architecture** that bridges state-of-the-art Neural Perception Models with a Deterministic Cognitive Graph, resolving the hallucinations and inconsistencies typically associated with purely neural audio LLMs.

## 🏗 Architecture Overview

The system is strictly divided into two distinct halves:

### 1. Neural Perception Layer (The "Ears")
We utilize a combination of mathematically frozen foundation models to extract acoustic features:
- **Whisper Large-v3 Turbo**: Extracts linguistic and acoustic embeddings (512d).
- **CLAP**: Extracts semantic audio embeddings (512d).
- **HTS-AT**: Extracts polyphonic event embeddings (768d).

These features are passed into a trainable **Fusion Layer** (1792d $\rightarrow$ 256d) and mapped to 40 scene classes via the **Scene Context Network**. *Only the Fusion Layer and Scene Context Network receive gradients during training.*

### 2. Deterministic Cognitive Graph (The "Brain")
Neural outputs are thresholded (`> 0.5`) and injected as discrete `EventNode` and `EntityNode` objects into the **Auditory World Model (AWM)**.
The reasoning engine then applies strict, logic-based, deterministic pipelines:
- **ARG (Auditory Relationship Graph)**: Explicitly connects events and entities based on co-occurrence and causality.
- **PSE (Perceptual Segregation Engine)**: Separates the scene into salient foreground and ambient background streams.
- **HRE (Hypothesis Reasoning Engine)**: Generates logical deductions from active streams.
- **BSE (Belief State Engine)**: Assigns hierarchical confidence.
- **WSE (World State Engine)**: Tracks the global state (e.g., Normal vs. Elevated Threat).
- **SPE (Situation Projection Engine)**: Projects short-term futures.
- **TRE (Transparent Reasoning Engine)**: Generates the JSON-auditable trace.
- **SIR (Situation Intelligence Renderer)**: Formats the human-readable report.

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Recommended: Python 3.10+
pip install -r requirements.txt
```

### 2. Data Preparation
Place your raw `.wav` files into the following directories. The Dataset Builder will dynamically mix them into realistic acoustic scenes (handling overlap, SNR, reverb, and random gain). Multilingual speech directories are loaded recursively.
- `data/raw/speech/<language>/` (e.g., english/, hindi/, telugu/)
- `data/raw/environment/<Class_Name>/`

### 3. Training (Transfer Learning)
To comply with Kaggle hardware limits, ALM v10.8 uses "Option B" (Precompute & Cache).
```bash
# Step 1: Synthesize scenes and precompute Whisper/CLAP/HTS-AT embeddings
python -m training.dataset_builder --max_events 6

# Step 2: Train the Fusion Layer & Scene Context Network
python -m training.train --epochs 50 --batch_size 32
```

### 4. End-to-End Inference
Launch the UI to run the full unified pipeline, complete with reasoning latencies, active events, and the JSON reasoning trace.
```bash
python application/app.py
```
*(Runs on `http://0.0.0.0:7860`)*

## 🧠 Why Deterministic Reasoning?
Traditional Audio-LLMs output next-token probabilities, making them susceptible to hallucinations when exposed to complex, overlapping environments. ALM v10.8 uses neural networks *only* for perception. All logic, physics tracking, and deductions occur in the Deterministic Graph, guaranteeing 100% interpretability and reliability.

## 📁 Model Artifacts & Reproducibility
The `models/` directory contains the production-ready weights and comprehensive metadata for scientific reproducibility:
- `alm_v10_final.pt`: The trained production model (Fusion Layer & Scene Context Network).
- `model_card.md`: Hugging Face-style model card with architecture, metrics, and capabilities.
- `training_config.json`: Hyperparameters and configuration used during the Kaggle training run.
- `dataset_manifest.json`: Detailed breakdown of the curriculum dataset (languages, classes, duration).
- `training_metrics.json`: Final empirical metrics on the validation hold-out set.
- `version.json` & `best_checkpoint_info.json`: Release engineering and checkpointing metadata.
