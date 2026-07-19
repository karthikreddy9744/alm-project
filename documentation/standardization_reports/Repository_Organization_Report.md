# Repository Organization Report
**Objective:** Verify folder responsibilities and flag overlaps.

## Responsibility Matrix & Overlap Analysis

### 1. `scripts/` vs `research/`
- **Current State:** `research/` contains `evaluation_runner.py`. `scripts/` contains `benchmark.py` and `validation_suite.py`.
- **Overlap:** Both folders attempt to run inference across batches of audio for benchmarking.
- **Recommendation:** **MERGE**. Move all contents of `scripts/` into `research/`. Delete `scripts/` to enforce a single source of truth for all experimental execution.

### 2. `samples/` vs `datasets/`
- **Current State:** Both folders contain `.mp3` and `.wav` files used for testing.
- **Overlap:** Total. 
- **Recommendation:** **MERGE**. All audio assets must live in `datasets/`. `samples/` should be deprecated.

### 3. `configuration/` vs Root Level
- **Current State:** `configuration/` contains only `requirements.txt`, which duplicates the root `requirements.txt`.
- **Overlap:** Total. Duplicate dependency management creates extreme reproducibility risks (split environments).
- **Recommendation:** **DELETE** `configuration/requirements.txt` and the `configuration/` folder entirely. 

### 4. `data/` and `models/`
- **Current State:** Both folders are effectively empty or contain unused subdirectories (`processed`, `training`).
- **Overlap:** None, but they are "Repository Noise".
- **Recommendation:** **ARCHIVE / DELETE**. Since ALM v12.0 relies on HuggingFace cache for frozen models and `datasets/` for audio, these folders serve no purpose.

## Summary of Validated Folders
The final, strictly delineated folder structure should be:
1. `core_modules/` (Perception)
2. `reasoning_engine/` (Logic)
3. `research/` (Execution & Benchmarking)
4. `evaluation/` (Results & Schemas)
5. `datasets/` (Audio Files)
6. `documentation/` (Specs & Knowledge Base)
7. `archive/` (Legacy Storage)
