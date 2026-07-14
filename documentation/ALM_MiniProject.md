# Auditory Language Model (ALM): Technical Specification

## 1. Project Overview
The Auditory Language Model (ALM) is an advanced multi-modal cognitive architecture designed to achieve **Human-Oriented Auditory Situation Understanding**. Unlike traditional sound classifiers or speech recognizers that output discrete labels, ALM functions as a complete auditory reasoning engine. It fuses neural perception (speech and environmental sound) with semantic reasoning to reconstruct the real-world situation represented by an audio stream, explicitly accounting for provenance, recording characteristics, and cross-modal consistency.

## 2. Problem Statement
Current audio understanding systems treat speech and environmental sounds as isolated domains. They map waveforms to literal labels (e.g., "Explosion", "Music") or transcripts without understanding the *context* or the *nature of the representation*. A system that detects "explosion" and "screaming" might falsely alert emergency services when it is actually listening to an action movie. There is a lack of systems capable of interpreting audio streams with the contextual awareness and provenance reasoning inherent to human cognition.

## 3. Motivation
Humans do not interpret every sound literally. Subconsciously, a human listener determines whether they are experiencing a real event or a media representation (e.g., a movie, a song, a podcast) and filters their understanding accordingly. To build robust, real-world AI systems capable of monitoring complex acoustic environments, AI must emulate this process—separating objective perceptual observation from subjective semantic interpretation.

## 4. Research Gap
While foundation models exist for isolated perceptual tasks (Whisper for ASR, CLAP for zero-shot audio classification, HTS-AT for event detection), there is no standardized architecture that bridges the semantic gap between literal audio events and abstract situational context. Existing models fail to reason about *provenance* (the origin of the audio) and frequently hallucinate context when evidence is ambiguous.

## 5. Research Objectives
- **Acoustic-Semantic Fusion**: Combine discrete perceptual streams into a unified semantic representation.
- **Explainable Reasoning**: Replace black-box classification with transparent, step-by-step reasoning chains.
- **Provenance Awareness**: Implement explicit mechanisms to identify the representational nature of audio (e.g., Live vs. Media).
- **Robustness to Ambiguity**: Enforce the principle that "Evidence Dominates Assumptions," preventing the system from guessing unknown context.

## 6. Design Philosophy
ALM is governed by three fundamental principles:
1. **Separation of Perception and Reasoning**: Perception observes; reasoning interprets. Neural models (Whisper, CLAP, HTS-AT) provide objective facts. The LLM consumes these facts and reasons over them.
2. **Evidence Dominates Assumptions**: The reasoning engine is forbidden from hallucinating context, specific identities, or visual details that cannot be deduced directly from the audio.
3. **Provenance is Probabilistic**: The system determines the likelihood of audio being a live event versus a media representation before making literal interpretations.

## 7. Human-Oriented Auditory Situation Understanding
The ultimate goal of ALM is not to list what sounds are present, but to explain *what is happening*. It seeks to answer: What is the situation? Who is involved? What is the context? It translates raw data into a human-empathetic narrative that captures the true essence of the acoustic scene.

## 8. System Architecture
ALM utilizes an 8-step Cognitive Pipeline that flows from raw audio arrays to complex situational reports:
1. **Neural Perception Layer**: Extracts speech and environmental data.
2. **AudioEvidenceObject Serialization**: Packages observations into JSON.
3. **Speech Understanding**: Analyzes the transcript.
4. **Auditory Observation Analysis**: Assesses environmental cues.
5. **Audio Provenance Reasoning**: Estimates the recording's origin.
6. **Cross-Modal Evidence Verification**: Resolves conflicts between speech and sound.
7. **Evidence Influence Assessment**: Weights the importance of observations.
8. **Situation Interpretation**: Generates the final human-oriented summary.

## 9. Neural Perception Layer
The foundation of ALM's pipeline. It provides objective observations without attempting to interpret them contextually.
- **Whisper**: Provides Automatic Speech Recognition (ASR).
- **HTS-AT**: Provides high-temporal-resolution event detection.
- **CLAP**: Provides zero-shot semantic mapping and environmental context.
- **Recording Characterization**: Analyzes acoustic features (reverb, clipping, background music) and explicitly maps them in the perception layer to inform downstream provenance reasoning.

## 10. AudioEvidenceObject
The `AudioEvidenceObject` is the strict Pydantic JSON schema that bridges the perception and reasoning layers. It contains:
- Transcripts and speaker data.
- Environmental observations with timestamps and acoustic salience.
- `RecordingCharacterization` detailing the objective acoustic qualities of the file.

## 11. Speech Understanding
The first reasoning step where the semantic engine analyzes the Whisper transcripts to determine the intent, language, and emotional tone of the speaker(s).

## 12. Auditory Observation Analysis
The semantic engine reviews the HTS-AT/CLAP outputs to determine how environmental sounds contribute to the overarching hypothesis.

## 13. Audio Provenance Reasoning
A critical cognitive step where ALM asks: "What kind of audio am I listening to?" It assesses whether the audio represents a Real-World Event, a Media Production (Movie, Song), a Broadcast, or a Synthetic/AI generation. It calculates a `provenance_reliability` metric to express confidence in this assessment.

## 14. Initial Semantic Hypothesis
Based on speech, observations, and provenance, the engine forms a baseline hypothesis of the situation.

## 15. Cross-Modal Evidence Verification
The engine actively checks for alignment or contradiction between the speech stream and the environmental audio stream. If a person is speaking calmly but explosions are heard, the engine flags a contradiction or assumes a media provenance (e.g., action movie).

## 16. Evidence Influence Assessment
Not all sounds are equally important. This step assigns weights (PrimarySupport, SecondarySupport, Background, Contradictory) to each observation based on its relevance to the final hypothesis.

## 17. Situation Interpretation
The engine synthesizes all prior steps into a cohesive, abstract understanding of the scene, logging missing evidence and remaining uncertainty.

## 18. Human-Oriented Auditory Situation Understanding (Expanded)
A dedicated summary generated by the engine that mimics human empathetic understanding. It strips away technical jargon (e.g., "CLAP score 0.8") and explains the scene naturally: "This appears to be an action movie sequence based on the high-fidelity sound design and lack of natural reverberation."

## 19. Cognitive Pipeline
The strict sequence of steps enforced by the prompt and Pydantic models. By forcing the LLM to think in this specific order (Provenance before Interpretation), ALM avoids premature conclusions.

## 20. Data Flow
`Audio File -> Whisper/HTS-AT/CLAP -> Python Dict -> AudioEvidenceObject -> Qwen-4B LLM -> SemanticSceneObject -> HRE/SIR Engines -> Three-Tier JSON Report.`

## 21. Prompt Engineering Philosophy
The prompt acts as the "Cognitive Framework" for the LLM. It strictly instructs the model to act as a reasoning engine rather than a raw classifier. The prompt forbids the LLM from hallucinating specific identities or copyrighted material, enforcing a reliance on the `AudioEvidenceObject`.

## 22. Pydantic Schema Philosophy
ALM relies entirely on structured JSON outputs defined by Pydantic models. The schema is the source of truth. If a feature isn't in the schema, it doesn't exist in the pipeline.

## 23. Confidence Propagation
Confidence is tracked at multiple levels:
- Perceptual Confidence (from neural models).
- Provenance Reliability.
- Cross-Modal Agreement Level.
- Final Situation Confidence.

## 24. Recording Characterization
Conducted purely in the Neural Perception Layer. It identifies background music, compression artifacts, and acoustic environments without semantic bias.

## 25. Provenance Reasoning (Expanded)
Provenance is treated as a probabilistic hypothesis. ALM cannot definitively "know" an audio clip is a deepfake, but it can state that the pristine vocal quality lacking natural breathing suggests a synthetic origin (`RepresentationType.SYNTHETIC_GENERATION`).

## 26. Narrator vs Participant
A core distinction ALM makes during speech analysis. Is the speaker experiencing the event (Participant) or describing it over the event (Narrator)?

## 27. Evidence Dominates Assumptions
The golden rule of ALM. If there is no evidence for a visual detail or specific context, ALM must leave the field as `Unknown`.

## 28. Intellectual Property Policy
ALM is explicitly programmed never to identify specific actors, copyrighted movies, songs, or franchises. It describes the *situation* depicted within the media, not the media's commercial identity.

## 29. Audio Types Supported
ALM is designed to process:
- Live microphones
- Uploaded recordings
- Conversations
- Classrooms
- Disasters
- News Broadcasts
- Documentaries
- Podcasts
- Songs
- Movies
- Synthetic Audio
- Surveillance
- Emergency calls

## 30. Folder Structure
```text
alm-project/
├── main.py
├── COMMANDS
├── README.md
├── requirements.txt
├── configuration/
├── core_modules/
├── data/
├── documentation/
│   ├── ALM_MiniProject.md
│   ├── COMMANDS.md
│   ├── README.md
│   └── research/
│       └── archive/
├── evaluation/
├── models/
├── reasoning_engine/
│   ├── fusion/ (Neural Perception Layer)
│   ├── semantic/ (Qwen LLM Reasoning Layer)
│   ├── hre/ (Hypothesis Resolution Engine)
│   ├── spe/ (Situation Projection Engine)
│   ├── sir/ (Situation Intelligence Renderer)
│   ├── tre/ (Transparent Reasoning Engine)
│   └── wse/ (World State Engine)
├── samples/
└── scripts/
```

## 31. Current Codebase
The current codebase reflects ALM Version 12, featuring a clean separation between the Perception Layer (`fusion`) and the Semantic Layer (`semantic`), utilizing Qwen-4B for JSON generation.

## 32. Configuration
All system thresholds, file paths, and model parameters are managed centrally in `reasoning_engine/config.py`.

## 33. Models Used
- **ASR**: OpenAI Whisper Base
- **Zero-Shot Audio**: LAION CLAP (HTS-AT Fused)
- **Event Detection**: HTS-AT
- **Semantic Engine**: Qwen/Qwen3-4B-Instruct-2507 (4-bit quantized)

## 34. Deployment
ALM is designed as a local-first pipeline to guarantee privacy. It runs entirely on-device, leveraging local GPUs (e.g., MPS on Apple Silicon).

## 35. Limitations
- **Deepfake Detection**: Cannot definitively prove audio is a deepfake without a dedicated forensic model; relies on perceptual hints.
- **Latency**: End-to-end processing requires significant compute time depending on audio length.
- **Polyphony**: Highly dense audio scenes may mask subtle background events.

## 36. Evaluation Strategy
ALM is evaluated using diverse datasets comprising movies, news, emergency calls, and synthetic audio. Evaluation metrics focus on Provenance Accuracy, Cross-Modal Consistency, and Halucination Reduction.

## 37. Future Research
- Integration of dedicated Deepfake/Synthetic Audio forensic models into the Perception layer.
- Streaming architecture for real-time low-latency processing.
- Enhanced memory state (WSE) for long-form context tracking across multiple hours of audio.

## 38. References
- OpenAI Whisper
- LAION CLAP
- HTS-AT Audio Transformer
- Qwen LLM Series

## 39. Glossary
- **ALM**: Auditory Language Model
- **SIR**: Situation Intelligence Renderer
- **HRE**: Hypothesis Resolution Engine
- **Provenance**: The origin and nature of the audio representation.

## 40. Change Log
- **V12**: Final Architecture Implementation. Separated `RecordingCharacterization` from Semantic Engine to Perception Layer. Implemented `AudioProvenanceReasoning` and "Evidence Dominates Assumptions" rule.
