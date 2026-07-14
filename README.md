# 🎧 ALM: Human-Oriented Auditory Situation Understanding

ALM is a multi-modal cognitive architecture designed to bridge the gap between raw auditory perception and human-plausible situation understanding. Traditional Audio Language Models either classify blindly without context, or use massive LLMs that suffer from high latency and hallucination. 

ALM completely replaces the stochastic guessing of pure LLMs by confining neural models inside a strict, deterministic cognitive graph, separating objective perception from subjective reasoning.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies (requires Python 3.10+):
```bash
git clone https://github.com/your-repo/alm-project.git
cd alm-project
pip install -r requirements.txt
```

### 2. Usage
To run a local validation of the pipeline:
```bash
python3 main.py
```

## 🧠 Core Features & Architecture
- **Offline & Local:** Runs entirely on your local machine. No API keys required.
- **Explainable AI:** Every situation deduction is tracked by a Transparent Reasoning Engine. No "black box" conclusions.
- **Provenance Reasoning:** Automatically deduces if audio is a live event, a movie, a song, or synthetic.
- **Evidence Dominates Assumptions:** Designed never to hallucinate visual details or specific identities without absolute acoustic proof.

## 💻 Models Used
- **Whisper Large-v3:** Linguistic perception and transcript generation.
- **CLAP & HTS-AT:** Acoustic and environmental feature extraction.
- **Qwen3-4B-Instruct-2507:** Highly constrained local language model used strictly for structured semantic reasoning.

## 📄 Comprehensive Documentation (Single Source of Truth)
For an in-depth breakdown of the architecture, data contracts, philosophy, and the complete 40-point technical specification, **you MUST read**:
[ALM_MiniProject.md](documentation/ALM_MiniProject.md)

---
*ALM is an active research project exploring evidence-grounded, explainable artificial intelligence.*
