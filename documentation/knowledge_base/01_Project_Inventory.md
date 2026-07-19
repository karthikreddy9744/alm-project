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
