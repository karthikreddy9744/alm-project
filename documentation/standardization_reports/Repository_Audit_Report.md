# Repository Audit Report
**Objective:** Global line-by-line classification of all repository folders and files.

## 1. Folder Classification

| Folder | Classification | Justification |
| :--- | :--- | :--- |
| `core_modules/` | **ACTIVE** | Contains `feature_extractor.py` and `inference_pipeline.py`. Core to v12 zero-shot execution. |
| `reasoning_engine/` | **ACTIVE** | Houses all logic constraints (SIE, TRE, HRE, WSE, SPE, SIR). Required for execution. |
| `documentation/` | **ACTIVE** | Contains `ALM_MiniProject.md` and `knowledge_base/`. The definitive source of truth. |
| `evaluation/` | **ACTIVE** | Houses `hoasu_bench.json`. |
| `evaluation/results/` | **GENERATED** | Destination for CSV benchmarking outputs. |
| `research/` | **ACTIVE** | Contains `evaluation_runner.py` and statistical scripts. |
| `archive/` | **ARCHIVE** | Cold storage for legacy PyTorch checkpoint files (`.pt`). |
| `datasets/` | **ACTIVE** | Physical audio files for pipeline evaluation. |
| `samples/` | **TEMPORARY / DEPRECATED** | Seems to duplicate `datasets/`. Contains test MP3/WAV files (`Loki.mp3`, `test.wav`). |
| `scripts/` | **EXPERIMENTAL** | Contains `benchmark.py` and `validation_suite.py`. Overlaps significantly with `research/`. |
| `application/` | **EXPERIMENTAL** | Contains `app.py`. A Gradio GUI for the pipeline. |
| `configuration/` | **DEPRECATED** | Contains an isolated `requirements.txt`. Duplicates the root `requirements.txt`. |
| `models/` | **EMPTY** | Originally intended for model weights, but Whisper and Qwen are downloaded to HF cache. |
| `data/` | **LEGACY / EMPTY** | Contains `processed/` and `training/`. Remnant of v1 End-to-End training. |

## 2. File Classification (Root)

| File | Classification | Justification |
| :--- | :--- | :--- |
| `main.py` | **ACTIVE** | Primary entry point for ALM. |
| `colab_setup.ipynb` | **ACTIVE** | Environment bootstrapper for L4/A100 evaluation. |
| `requirements.txt` | **ACTIVE** | Global dependency list. |
| `README.md` | **LEGACY / DEPRECATED** | Outdated instructions that contradict `ALM_MiniProject.md`. |
| `COMMANDS` | **TEMPORARY** | A scratchpad of CLI commands. Should be merged into documentation. |
| `Dockerfile` | **ACTIVE** | Containerization definition. |
