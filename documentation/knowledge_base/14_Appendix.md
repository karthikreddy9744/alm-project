# 14: Appendix

## Glossary of Terms
- **HOASU:** Human-Oriented Auditory Situation Understanding.
- **AEO:** Audio Evidence Object. The central data schema bridging perception and cognition.
- **Provenance:** The representational nature of the audio (Live, Broadcast, Media, Synthetic).
- **Neuro-Symbolic:** A hybrid AI approach combining neural networks (Perception) with explicit logic constraints (Reasoning Engines).
- **Reasoning State Exposure:** The methodology of serializing intermediate logic conclusions to disk for auditability.

## Version History
- **v1 - v4:** E2E CNNs.
- **v5 - v8:** Audio-LLM parameter projection.
- **v9 - v11:** Hybrid PyTorch Scene models + Whisper.
- **v12.0:** Final Zero-Shot Cognitive Pipeline.

## Directory Reference Map
```text
alm-project/
├── core_modules/        # Neural perception and pipeline execution
├── reasoning_engine/    # Logic modules (HRE, TRE, WSE, SPE, SIR, Semantic)
├── evaluation/          # Final datasets and generated CSV results
├── research/            # Evaluation scripts and ablation definitions
├── archive/             # Cold-storage for legacy .pt files
├── literature_survey/   # Markdown analysis of competing models
├── datasets/            # Physical .mp3 and .wav audio files
├── documentation/       # Master specifications
└── colab_setup.ipynb    # GPU execution environment bootstrapper
```
