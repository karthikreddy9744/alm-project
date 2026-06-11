# 🎧 Audio Language Model (ALM)

## Project Overview
A deep learning system that **listens, thinks, and understands** both speech and environmental audio simultaneously.

## Features
- **Multi-modal Input**: Live microphone, file upload, and drag-and-drop audio
- **Speech Analysis**: OpenAI's Whisper for transcription and speech embeddings
- **Environmental Analysis**: CLAP (Contrastive Language-Audio Pretraining) for environmental sound classification
- **Fusion Architecture**: Custom Transformer Cross-Attention Fusion layer to dynamically combine speech + environmental embeddings
- **Scene Classification**: 20 categories including Traffic, Media, Emergency, Weather, Weather, Water, Wildlife, Indoor, Crowd, and more.
- **Context-Aware Response Engine (CASRE)**: Generates natural language scene understanding, media detection, and risk scoring without external LLMs
- **Premium UI/UX**: State-of-the-art Glassmorphic responsive interface with a dark theme and Google Inter typography.

## Architecture
1. **Audio Preprocessor**: Resamples to 16kHz, normalizes, applies VAD/RMS thresholding, and truncates/pads to max 60s
2. **Whisper Encoder**: Extracts 512-d speech embeddings and transcript (with repetition-based hallucination suppression)
3. **CLAP Encoder**: Extracts 512-d environmental audio embeddings
4. **Fusion Layer**: Combines embeddings into 256-d fused representation (regularized with `Dropout(0.3)` to prevent modality collapse)
5. **Scene Context Network**: Multi-label classification across 20 categories (trained with BCEWithLogitsLoss for class balancing)
6. **CASRE**: Generates natural language explanation of scene + recommended action (featuring cross-modal contradiction detection)

## 🚀 What's New in ALM v4.0 (Master Architecture Rewrite)
- **Cross-Attention Fusion**: Replaced basic MLP with a Transformer-based `nn.MultiheadAttention` fusion layer.
- **Sliding-Window Temporal Pipeline**: Analyzes audio in 5-second overlapping chunks to generate an exact timeline of acoustic events.
- **Silero VAD Frontend**: Completely eliminates Whisper hallucinations by slicing audio into strictly validated speech chunks before transcription.
- **Multi-Label Expansion**: Upgraded to 20 classes using independent Sigmoid outputs and `BCEWithLogitsLoss`, accurately detecting mixed environments.
- **Next-Gen CASRE**: The reasoning engine now computes Risk Scores, detects Media Playback (Movies/TV), and prevents semantic contradictions.


- **Premium Glassmorphic UI**: Redesigned Gradio dashboard with dark slate themes, CSS backdrop filters, and modern Inter typography.

## Setup & Installation
1. **Create virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   .\venv\Scripts\activate  # Windows
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download/Prepare model checkpoint** (dummy checkpoint provided in `models/`):
   ```bash
   mkdir -p models
   # To train your own model, see training/ directory
   ```

## Running the App
```bash
python app.py
```
Open the Gradio interface URL (e.g., `http://127.0.0.1:7860`) in your browser!

## Project Structure
```
alm-project/
├── app.py                     # Gradio app entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── core/                      # Core modules
│   ├── feature_extractor.py   # Whisper + CLAP feature extractors
│   ├── fusion_layer.py        # Fusion neural network
│   ├── scene_network.py       # Scene classification network
│   ├── casre_engine.py     # Context-Aware Smart Response Engine (CASRE)
│   └── inference_pipeline.py  # Full inference pipeline
├── models/                    # Model checkpoints
│   └── scene_model.pt         # Trained fusion+scene weights
├── training/                  # Training scripts
│   ├── dataset_builder.py     # Dataset preparation
│   ├── train.py               # Training loop
│   └── evaluate.py            # Evaluation
└── samples/                   # Sample audio files
```

## Training
1. Prepare ESC-50 dataset in `data/raw/`
2. Precompute embeddings using `training/dataset_builder.py`
3. Train model using `training/train.py`
4. Evaluate using `training/evaluate.py`

## Citation
Based on:
- OpenAI Whisper: Radford et al., 2022
- CLAP: Wu et al., 2022
- ESC-50: Piczak, 2015

## License
MIT
