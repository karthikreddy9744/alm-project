# 02: Project History

## ALM v1 - v4 (End-to-End CNNs)
- **Main Idea:** Custom PyTorch CNNs trained locally on the ESC-50 dataset.
- **Why it changed:** Catastrophically brittle.
- **Lessons Learned:** Local, narrow datasets are too small for real-world generalization; deep learning audio classifiers act as black boxes.
- **Reason for Moving On:** Hit the "Explainability Wall".

## ALM v5 - v8 (Audio-LLM Hybrids)
- **Main Idea:** Projecting audio embeddings (CLAP) directly into an LLM's token space.
- **Why it changed:** Severe, uncontrollable hallucinations. 
- **Lessons Learned:** LLMs conflate audio features with visual data from their training distribution (e.g., guessing a speaker's shirt color from their voice).
- **Reason for Moving On:** Hit the "Hallucination Wall".

## ALM v9 - v11 (Hybrid Neuro-Symbolic)
- **Main Idea:** Fusion of Whisper ASR with a custom-trained local `scene_model.pt`.
- **Why it changed:** The local scene model became a massive computational and linguistic bottleneck.
- **Lessons Learned:** Custom training cannot compete with massive foundation models for semantic intelligence.
- **Reason for Moving On:** Hit the "Compute Wall".

## ALM v12 (Zero-Shot Structured Reasoning - FINAL)
- **Main Idea:** Total deprecation of custom models. 100% reliance on Whisper, CLAP, and Qwen3 chained sequentially via strict JSON schemas (`AudioEvidenceObject`).
- **Why it became final:** Architecturally sound, highly explainable, entirely zero-shot, and resolves all hallucination issues through Schema-Constrained Reasoning.
