# Dataset Report
**Objective:** Analyze the audio and JSON assets to determine purpose and validity.

## Asset Analysis

| Asset Location | Purpose | Verdict |
| :--- | :--- | :--- |
| `evaluation/hoasu_bench.json` | The golden list of 250 test schemas. | **KEEP**. Essential for the paper. |
| `evaluation/results/` | Output directory for CSV artifacts. | **KEEP (Generated)**. Safe to ignore in git via `.gitignore`. |
| `datasets/` | Destination for final, curated `.wav` files used by `hoasu_bench.json`. | **KEEP**. Essential. |
| `samples/` | Contains arbitrary files (`Loki.mp3`, `cyclone.mp3`, `test.wav`). Used historically for manual testing. | **MERGE & DEPRECATE**. These files should be moved into `datasets/` if they are part of the benchmark, or archived if they are just manual test files. |
| `data/training/` | Legacy artifact from ALM v1. | **ARCHIVE**. ALM v12 does not train custom weights. |
