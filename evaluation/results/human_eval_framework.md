# ALM v12.0 Human Evaluation Framework

This document outlines the protocol for human evaluators to score ALM outputs against baselines.

## Protocol
1. **Listen:** The participant listens to a single audio clip (drawn from the 50-item evaluation dataset).
2. **Review Ground Truth:** Participant identifies the *Ground Truth Situation* (what a human infers).
3. **Compare Outputs:** Participant reads the output from 3 blinded systems:
   - System A (Raw Qwen3-4B-Instruct-2507 text-only prompt)
   - System B (ALM v12.0 Output)
   - System C (Whisper Transcript only)

## Scoring Rubric (1-5 Likert Scale)

### 1. Situation Quality (Accuracy)
- 1: Completely wrong context/hallucinated.
- 3: Partially correct but misses key elements.
- 5: Perfectly captures the environmental and semantic situation.

### 2. Human Plausibility
- 1: Reads like an AI log (e.g. "I detect 85% probability of siren").
- 3: Acceptable but sterile.
- 5: Reads like a human describing the event ("There is a siren approaching, likely an emergency").

### 3. Explainability
- 1: Opaque reasoning; cannot tell how the conclusion was reached.
- 3: Shows some evidence but logic leaps are present.
- 5: Transparent connection between audio facts and final situation.

### 4. Completeness
- 1: Misses major audio events.
- 5: Captures all relevant auditory events.

### 5. Uncertainty Handling
- 1: Confidently incorrect or fails to acknowledge ambiguity.
- 5: Appropriately flags uncertainty when audio is unclear using natural language hedging (e.g. "There is a remote possibility...").

### 6. Naturalness
- 1: Robotic, overly verbose, repetitive.
- 5: Concise, natural, and highly readable.

## Ablation Scoring
In Phase 2 of evaluation, evaluators compare ALM vs ALM without specific layers (e.g., removing the Perceptual Segregation Engine). Evaluators use forced-choice: "Which output provides a better human-oriented understanding of the scene?"
