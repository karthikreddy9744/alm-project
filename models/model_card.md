---
language:
- en
- hi
- te
- ta
- bn
- gu
- mr
- pa
- kn
- ml
license: mit
tags:
- audio-classification
- auditory-world-model
- zero-shot
- multimodal
- environmental-sound-classification
---

# ALM v10: Auditory Language Model

**ALM (Auditory Language Model)** is a unified multimodal acoustic-cognitive architecture that bridges high-fidelity neural sound perception with a deterministic symbolic reasoning graph (the Auditory World Model).

## Model Details
- **Model Name:** ALM v10 Unified Architecture
- **Version:** v10.13 (Production Candidate)
- **Architecture:** Neural Perception Extractors + Fusion Layer + Scene Context Network + Deterministic Reasoning Engine.
- **Languages Supported:** 13+ (English, Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Spanish, French, German).
- **Sampling Rate:** 16kHz

## Foundation Models Used (Frozen Extractors)
- **Whisper (OpenAI, Base):** Provides native zero-shot multilingual speech transcription and language identification.
- **CLAP (LAION):** Contrastive Language-Audio Pretraining provides generalized semantic audio embeddings and edge-case zero-shot understanding.
- **HTS-AT (AudioSet/ESC-50):** An Audio Spectrogram Transformer explicitly trained on environmental sound events (ESC-50) for high-impulse event detection.

## Trainable Components
The core trainable components in this repository constitute a lightweight bridging mechanism to unify the massive foundation models:
1. **Fusion Layer:** A fast 3-layer MLP that projects the concatenated embeddings (Whisper 512 + CLAP 512 + HTS-AT 768) into a unified 256-dimensional semantic space.
2. **Scene Context Network:** A multi-label linear classifier that maps the 256D fusion space into 40 distinct auditory events.

## Training Metrics (Kaggle Production Run)
- **Dataset:** 100,000 real-world overlapping samples (80k Train / 20k Val).
- **Strategy:** 4-stage Curriculum Learning (Clean -> Polyphonic -> Multi-Speaker -> Extreme Overlap).
- **Epochs:** 40
- **Best F1-Score:** 0.8429
- **Precision:** 0.9367
- **Recall:** 0.7662
- **Validation Loss:** 0.0406

## Known Strengths
- **Zero-Shot Multilingualism:** Natively processes and understands spoken intent across diverse Indic languages without explicit retraining.
- **High-Impulse Reliability:** Exceptionally high precision (93.7%) for critical events (glass breaking, sirens, dog barking).
- **Ecological Plausibility:** The downstream Reasoning Engine (AWM) guarantees that impossible acoustic states are filtered out deterministically.

## Known Limitations
- **Severe Acoustic Masking:** Extremely crowded auditory scenes (e.g., sirens overlapping with loud music and screaming at -5dB SNR) may cause the HTS-AT filter to drop weaker signals (like footsteps).
- **Echo Dependency:** Heavy reverberation can slightly degrade the Whisper transcription WER.

## Intended Use
ALM v10 is designed for ambient acoustic intelligence, smart home auditory monitoring, cognitive robotics, and real-time auditory threat assessment. 

## Unsupported Scenarios
- Medical diagnostic audio (e.g., heart murmur detection).
- Music structural analysis (e.g., chord progression detection).
- Forensic audio restoration.

## Future Improvements
- Integration of a dedicated Source Separation (Demucs) layer prior to extraction to prevent acoustic masking in heavy overlap.
- Dynamic chunking based on Voice Activity Detection (VAD) to improve Whisper inference latency.
