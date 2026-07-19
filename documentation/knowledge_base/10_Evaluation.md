# 10: Evaluation

## The Evaluation Pipeline
The evaluation pipeline is entirely automated via `research/evaluation_runner.py`. It requires Google Colab (L4/A100 GPUs) for batch execution due to the massive VRAM footprint of running multiple LLM inferences per audio file.

## Execution and CSV Generation
1. The runner loads `hoasu_bench.json`.
2. It feeds each audio file into `inference_pipeline.py`.
3. It intercepts the intermediate JSON logic traces (Reasoning State Exposure).
4. It computes the execution latency for each stage.
5. It writes exactly 6 scientific artifacts, including `latency_report.csv`, `evaluation_results.csv`, and `execution_log.md`.

## Metrics and Human Evaluation
Because ALM generates qualitative intelligence reports, standard automated metrics (like BLEU or WER) are insufficient. 
- **Primary Metric:** Human Evaluation. Domain experts review the generated reports against the ground truth.
- **Statistical Analysis:** The `statistical_analysis.py` script computes Fleiss' Kappa to determine inter-rater reliability among the human graders, and Wilcoxon Signed-Rank tests to prove ALM's superiority over the Whisper-Only baseline.

## Future Evaluation Goals
Expanding the benchmark to include 1,000+ adversarial samples specifically designed to trick the Semantic Interpretation Engine.
