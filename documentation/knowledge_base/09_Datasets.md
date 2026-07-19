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
