# 08: Research

## Research Methodology
The ALM project follows an Evaluation-First research methodology. Architectural modules are not built in a vacuum; they are designed specifically to solve failure modes identified during formal benchmarking.

## Evaluation Strategy
Due to the qualitative, logic-driven nature of ALM, standard loss metrics (e.g., Cross-Entropy) are irrelevant. The evaluation strategy relies on:
1. **Procedural Benchmarking:** Running the pipeline automatically over hundreds of complex scenarios defined in `hoasu_bench.json`.
2. **Human Evaluation:** Using domain experts to grade the final HOASU outputs on a scale of 1-5 for accuracy, empathy, and logic consistency.
3. **Statistical Aggregation:** Calculating Fleiss' Kappa for human inter-rater reliability.

## Experimental Design & Ablations
To scientifically prove the necessity of ALM's massive architecture, experiments are run against baselines.
- **Baseline 1:** Raw Whisper + LLM (No CLAP, No strict schemas).
- **Baseline 2:** ALM without the Transparent Reasoning Engine (TRE).
- **ALM Full:** The complete v12.0 pipeline.
By ablating the TRE, researchers can mathematically prove the pipeline's inability to detect deepfakes or resolve cross-modal contradictions without explicit symbolic layers.

## Publication Strategy
The research is targeted at high-tier journals. The methodology section relies heavily on the transparency granted by "Reasoning State Exposure," allowing reviewers to audit the exact JSON traces produced by the model during the evaluation phase.
