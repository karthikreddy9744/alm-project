# 04: Literature Foundation

## Influencing Literature

### 1. Sci-Phi (Scientific Philosophy in AI)
- **Influence:** Heavily inspired ALM's transition away from End-to-End deep learning toward Neuro-Symbolic logic. Sci-Phi proved that structured symbolic logic (schemas) applied to neural outputs drastically reduces hallucinations.
- **Adopted:** The concept of explicit intermediate logic verification layers.

### 2. SLAM-LLM
- **Influence:** An industry standard for injecting audio embeddings directly into an LLM.
- **Adopted:** Validated the use of CLAP embeddings for environmental understanding.
- **Rejected:** SLAM-LLM's core thesis—mapping embeddings directly into token space—was ultimately rejected by ALM due to its inability to produce a transparent logic trace. ALM instead opted for the `AudioEvidenceObject` middle-ground.

### 3. "Can We Trust AI With Our Ears?"
- **Influence:** This foundational survey on auditory hallucinations highlighted the critical lack of "Provenance Reasoning" in modern classifiers.
- **Adopted:** ALM directly addresses this gap by implementing the `tre` (Transparent Reasoning Engine) specifically tasked with Cross-Modal Verification and Provenance deduction.

## Research Positioning
ALM positions itself at the intersection of **Machine Listening** and **Explainable AI (XAI)**. It is not competing to be the fastest acoustic event detector; it is competing to be the most cognitively robust and transparent auditory reasoning engine. 

## Current Novelty
The primary novelty lies in ALM's **Schema-Constrained Provenance Reasoning**. While models like Whisper transcribe speech, and models like HTS-AT tag sounds, ALM is the first architecture to explicitly fuse them, cross-reference them for contradictions (e.g., calm speech overlapping with sirens = Media/Synthetic), and serialize the logic.
