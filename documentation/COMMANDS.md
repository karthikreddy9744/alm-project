# ALM v12.0 Command Manual

This document provides a detailed breakdown of all terminal commands required to operate, evaluate, and develop ALM v12.0.

## 1. Environment Setup & Dependency Installation
### Command
```bash
pip install -r requirements.txt
```
### Purpose
Installs all necessary Python dependencies (PyTorch, HuggingFace Transformers, Gradio, Librosa, etc.) for running ALM.
### Expected Behaviour
Downloads packages and installs them into your active Python environment.
### Common Failures
- `ModuleNotFoundError` later on implies you forgot this step.
- Ensure you are on Python 3.10+ to avoid typing syntax errors.

---

## 2. Running Inference & Validation Pipeline (Headless)
### Command
```bash
python main.py
```
### Purpose
Runs the unified ALM validation pipeline (Whisper -> Fusion -> Qwen -> HRE -> WSE -> SPE -> TRE -> SIR) directly in the terminal over a default test sample.
### Expected Behaviour
- Loads Whisper Large-v3, CLAP, HTS-AT, and Qwen2.5-3B.
- Prints a structured dictionary directly to the terminal encompassing the `speech`, `environment`, and `situation` outputs.
### Developer Notes
Useful for verifying that the complete architecture executes without crashing before loading the UI.

---

## 3. Launching the User Interface
### Command
```bash
python application/app.py
```
### Purpose
Starts the Gradio web interface, which is the primary way human users interact with ALM.
### Expected Behaviour
- Binds to `http://localhost:7860`.
- Provides a drag-and-drop audio interface.
- Exposes the **Three-Tier Intelligence Report** (Speech, Environment, Situation).
- Provides a **Developer Mode** accordion to view raw JSON traces.
### Common Failures
- **Port Conflict:** If 7860 is in use, Gradio will silently increment to 7861. Check terminal output.

---

## 4. Running Benchmarks
### Command
```bash
python scripts/benchmark.py
```
### Purpose
Executes latency and tracemalloc memory profiling on the ALM pipeline.
### Expected Behaviour
- Prints Model Load Time, Total Pipeline Latency, module breakdown (in ms), Peak RAM, and Current RAM usage.
### Developer Notes
Use this when modifying the `reasoning_engine` logic to ensure you have not introduced performance regressions. ALM v12.0 Cognitive Pipeline overhead should remain strictly under 50ms.

---

## 5. Running the Validation Suite (Research Phase)
### Command
```bash
python scripts/validation_suite.py
```
### Purpose
Evaluates all samples inside the `samples/` directory and dumps results into a CSV format.
### Expected Behaviour
- Produces a `validation_report.csv` file mapping filenames to success/failure states and latency markers.
### Developer Notes
Essential for ablation studies or reproducing research evaluation metrics across large datasets.

---

## 6. Cleanup Model Cache
### Command
```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct
```
### Purpose
Frees up disk space if you no longer intend to run ALM locally.
### Expected Behaviour
- Reclaims approximately ~6-8GB of local storage space.
### Common Failures
- If you run `python main.py` immediately after this, the system will have to re-download the massive Qwen tensor files.
