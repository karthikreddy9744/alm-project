import os

def generate_diagrams():
    diagrams_path = os.path.join(os.path.dirname(__file__), "architecture_diagram.md")
    
    mermaid_code = """# ALM v12.0 Research Diagrams

## ALM v12.0 System Architecture

```mermaid
graph TD
    A[Raw Audio Input] --> B(Whisper Large-v3)
    A --> C(CLAP/HTS-AT)
    B -->|Transcripts| D(Auditory World Model)
    C -->|Scene Labels| D
    D --> E(Perceptual Segregation Engine)
    E -->|AudioEvidenceObject| F(Evidence Fusion Layer)
    F --> G(Semantic Interpretation Engine <br> Qwen2.5-3B-Instruct)
    G -->|SemanticSceneObject| H(Hypothesis Reasoning Engine)
    H -->|ManagedHypothesisState| I(World State Engine)
    I -->|CognitiveState| J(Situation Projection Engine)
    J --> K(Transparent Reasoning Engine)
    K --> L(Situation Intelligence Renderer)
    L --> M{Human-Oriented Output}
```

## ALM v12.0 Module Interaction

```mermaid
sequenceDiagram
    participant User
    participant AWM as Auditory World Model
    participant Fusion as Evidence Fusion
    participant Qwen as Semantic Engine
    participant Cog as Cognitive State Pipeline
    participant SIR as Intelligence Renderer
    
    User->>AWM: Audio Stream
    AWM->>Fusion: Acoustic & Linguistic Facts
    Fusion->>Qwen: Structured Evidence
    Qwen-->>Fusion: Semantic Breakdown JSON
    Fusion->>Cog: SemanticSceneObject
    Cog->>Cog: Project Future States
    Cog->>Cog: Generate Audit Trace
    Cog->>SIR: Comprehensive CognitiveState
    SIR-->>User: 3-Tier Human Readable Report
```
"""
    with open(diagrams_path, "w") as f:
        f.write(mermaid_code)
    print(f"Generated diagrams at {diagrams_path}")

if __name__ == "__main__":
    generate_diagrams()
