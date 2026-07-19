# ALM v12.0 Scientific Evaluation Methodology

**Target Venues:** IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), ICASSP, Elsevier Neural Networks.

## 1. Introduction and Scope
This document outlines the rigorous, reproducible scientific evaluation methodology for the ALM (Auditory Language Model) v12.0 architecture. Unlike traditional end-to-end Audio Large Language Models (Audio-LLMs) which are evaluated primarily on text-generation metrics (e.g., BLEU, ROUGE), ALM is a structured Cognitive Reasoning Engine. 

Therefore, this evaluation framework is specifically designed to quantitatively and qualitatively measure **Perception Quality, Reasoning Quality, Explainability, Robustness, and Transparency**. The core hypothesis is that explicit separation of perception and reasoning reduces hallucination and provides mathematically provable traceability compared to black-box baselines.

---

## 2. Metrics: Perception and Reasoning Quality

### 2.1 Neural Perception Metrics
Validates that the objective sensory layer extracts accurate environmental and semantic representations before any semantic reasoning occurs.
- **Word Error Rate (WER):** Evaluates the Whisper subsystem on E2E audio. Calculated as `(Substitutions + Deletions + Insertions) / Total Words`.
- **Mean Average Precision (mAP):** Evaluates the HTS-AT environmental event detection across the 40-class scene taxonomy.
- **Semantic Alignment Score (SAS):** Evaluates the zero-shot accuracy of CLAP acoustic concept mapping. Measured as the Top-1 and Top-3 accuracy against ground-truth labels.

### 2.2 Cognitive Reasoning Metrics
Validates the Semantic Interpretation Engine and Hypothesis Reasoning Engine.
- **Provenance Accuracy (PA):** The classification accuracy of the Engine’s `source_type` inference (e.g., Broadcast, Real-world, Movie). Proves the model correctly contextualizes audio rather than merely describing sound waves.
- **Cross-Modal Agreement Rate (CMAR):** Measures how often the engine correctly tags conflicting multi-modal streams (e.g., Speech vs. Environment) as *Contradictory* rather than artificially forcing alignment.
- **Situation Accuracy (SA):** A hybrid metric evaluating the correctness of the final `primary_situation` against the HOASU-Bench benchmark.

---

## 3. Metrics: Explainability and Transparency

ALM’s primary architectural claim is the elimination of opaque logic. These novel metrics quantify explainability:
- **Traceability Index (TI):** 
  - *Definition:* The percentage of high-level semantic claims generated in the final report that hold a direct pointer (ID) back to a raw sensory `AudioEvidenceObject`. 
  - *Goal:* A TI of 100% indicates zero hallucination.
- **Hallucination Rate (HR):**
  - *Definition:* The frequency of outputs violating the "Evidence Dominates Assumptions" rule. 
  - *Example:* Assuming the specific identity of a speaker or a copyrighted movie title without explicit spoken evidence.

---

## 4. Evaluation Protocol

- **Dataset:** The evaluation will strictly use the **HOASU-Bench** (Human-Oriented Auditory Situation Understanding Benchmark), containing 250 curated audio scenarios balancing Provenance, Cross-Modal Conflict, and Environments.
- **Inference Condition:** **Zero-shot Deterministic E2E**. The system must evaluate the full dataset in a single E2E pass without any prompt-tuning, fine-tuning, or parameter updates allowed between samples.
- **Hardware Profile:** Standardized on Apple Silicon (MPS) or NVIDIA RTX 4090/A100 (CUDA) to report deterministic latency (ms/token) and Memory Footprint (VRAM usage).

---

## 5. Human Evaluation Framework

Because the Situation Intelligence Renderer (SIR) produces an empathetic, human-oriented narrative, standard NLP metrics (ROUGE, BLEU) fail to capture nuance. ALM utilizes a rigorous Human Evaluation Protocol.

- **Blind A/B Testing:** Expert annotators will be presented with the generated outputs of ALM and Baseline models randomized side-by-side without labels.
- **Annotator Pool:** 3 independent expert reviewers per audio sample.
- **Scoring System (5-Point Likert Scale):**
  1. *Empathetic Narrative Quality:* Does the summary sound like a human experiencing/interpreting the event (5) or a dry list of sounds (1)?
  2. *Jargon Avoidance:* Is the report entirely free of technical terms (e.g., "CLAP score", "HTS-AT")? (5 = Yes, 1 = No).
  3. *Contextual Accuracy:* Does the summary capture the *true meaning* of the event?

---

## 6. Statistical Analysis

To satisfy rigorous peer-review standards, all results will be subjected to the following statistical treatments:
- **Inter-Annotator Agreement:** Measured using **Fleiss' Kappa** ($\kappa$). A threshold of $\kappa > 0.70$ is required to prove that subjective ratings are reliable and not random noise.
- **Significance Testing:** Because Likert scale data is ordinal and often non-normally distributed, the **Wilcoxon signed-rank test** will be utilized to determine statistical significance between ALM and Baseline models ($p < 0.05$).
- **Confidence Intervals (CI):** 95% Confidence Intervals will be computed for all quantitative metrics (PA, CMAR, TI) using **bootstrapping** (N=10,000 resamples) over the HOASU-Bench test set.
- **Cognitive Correlation:** **Pearson's Correlation Coefficient ($r$)** will be used to measure the relationship between the engine's internal `uncertainty` score and the actual ground-truth error rate, proving that the model "knows what it doesn't know."

---

## 7. Baselines

To prove the superiority of structured cognitive reasoning over brute-force scaling, ALM will be benchmarked against:
1. **Qwen-Audio / SALMONN:** State-of-the-art E2E Audio-LLMs. These models serve as the "Opaque Black-Box" baselines to highlight ALM's superiority in Traceability Index (TI) and Provenance Accuracy.
2. **Naive Pipeline (Whisper + CLAP + LLM Prompting):** A standard sequential pipeline lacking the Hypothesis Reasoning Engine (HRE) and World State Engine (WSE). This ablation baseline proves that ALM's intelligence stems from its deterministic cognitive graph, not just prompt engineering.

---

## 8. Error and Failure Analysis

A mandatory inclusion for scientific transparency. Reviewers will be provided with an explicit breakdown of ALM's limitations:
- **Perceptual Masking Failures:** Analysis of E2E failures where the Perceptual Segregation Engine (PSE) incorrectly flags foreground speech as 'Background' due to extreme high-tension events (e.g., alarms, explosions).
- **Cross-Modal False Positives:** Instances where ALM identifies Evidence as *Contradictory* when it is actually a rare but true real-world acoustic overlap (e.g., heavy rain sounds inside a dry studio due to a prop).
- **Latency Trade-offs:** Quantitative analysis of the time overhead introduced by the Multi-Agent Hypothesis Competition vs. standard Single-pass LLM generation.
