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
