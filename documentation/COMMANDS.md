# ALM v10.4: Quick Reference Commands

This document contains all the essential terminal commands you will need to operate, test, evaluate, and deploy the ALM (Audio Language Model) v10 Cognitive Audio Scene Reasoning Engine (CASRE).

> [!TIP]
> **Important Note on `PYTHONPATH`:** 
> For scripts inside subdirectories (like `tests/` or `evaluation/`), you must prepend the command with `PYTHONPATH=.` so Python knows to look for the core modules in the root directory.

---

## 1. Environment Setup

Before running anything, ensure your virtual environment is active and all required libraries are installed.

**Activate Virtual Environment:**
```bash
source venv/bin/activate
```
*Why it's useful: Activates your local Python environment isolating dependencies like PyTorch, Torchaudio, and Transformers.*

**Install/Update Dependencies:**
```bash
pip install -r requirements.txt
```
*Why it's useful: Ensures you have the exact packages required to run the cognitive reasoning engine.*

---

## 2. Running the Pipeline (Deployment)

**Execute the E2E Cognitive Pipeline:**
```bash
python main.py
```
*Why it's useful: Triggers the main entry point for the ALM v10 pipeline. It routes mocked acoustic inputs through the Auditory World Model (AWM), runs all six cognitive stages (BSE -> HRE -> WSE -> SPE -> TRE -> SIR), and logs the deterministic Situation Report in <1ms.*

**Build the Docker Container:**
```bash
docker build -t alm-v10 .
```
*Why it's useful: Packages the entire repository into a production-ready `python:3.11-slim` container.*

**Run the Docker Container:**
```bash
docker run alm-v10
```
*Why it's useful: Executes the `main.py` pipeline in a fully headless, deterministic sandbox.*

---

## 3. Testing & Validation

**Run the Entire Test Suite:**
```bash
PYTHONPATH=. python -m unittest discover tests
```
*Why it's useful: Automatically executes all integration tests in the `tests/` directory to ensure the logic within the reasoning engine (BSE, HRE, WSE, etc.) remains fully operational and mathematically sound.*

**Run a Specific Unit Test (e.g. HRE):**
```bash
PYTHONPATH=. python -m unittest tests.test_reasoning_engine.test_hre
```
*Why it's useful: Focuses testing on a single cognitive module when actively debugging.*

---

## 4. Model & HuggingFace Commands

*Note: ALM v10 strictly uses external frozen models for perception. You must manage them separately if running live streams.*

**Download Whisper Large-v3 Turbo (INT8):**
```bash
huggingface-cli download openai/whisper-large-v3-turbo --local-dir models/whisper
```
*Why it's useful: Caches the INT8 quantized Whisper model locally to bypass network latency during live acoustic tracking.*

**Download CLAP (Environmental Audio):**
```bash
huggingface-cli download laion/clap-htsat-unfused --local-dir models/clap
```
*Why it's useful: Retrieves the LAION contrastive audio-text model for zero-shot environmental classification.*

---

## 5. Repository Maintenance

**Clear Python Caches (`__pycache__`):**
```bash
find . -type d -name "__pycache__" -exec rm -r {} +
```
*Why it's useful: Cleans up compiled python bytecode to ensure you are executing the absolute latest versions of the codebase without stale cache interference.*

**Generate System Logs:**
```bash
python main.py > system_run.log 2>&1
```
*Why it's useful: Dumps the structured `logging` output from a full pipeline execution directly into a log file for later review.*
