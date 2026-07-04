import re

filepath = "documentation/ALM Project Documentation.md"

with open(filepath, "r") as f:
    content = f.read()

# 1. Update Mermaid diagrams
old_mermaid_pattern = r"```mermaid.*?```"
new_mermaid = """```mermaid
graph TD
    A[Audio Input Stream] --> B(Whisper INT8 Encoder)
    A --> C(CLAP/HTS-AT Encoder)
    
    B --> D[Auditory World Model - AWM]
    C --> D
    
    D --> E[Acoustic Relationship Graph - ARG]
    E --> F[Belief State Engine - BSE]
    F --> G[Hypothesis Reasoning Engine - HRE]
    G --> H[World State Estimation - WSE]
    H --> I[Situation Projection Engine - SPE]
    I --> J[Transparent Reasoning Engine - TRE]
    J --> K[Situation Intelligence Renderer - SIR]
```"""
# Replace all instances of mermaid with the new one
content = re.sub(old_mermaid_pattern, new_mermaid, content, flags=re.DOTALL)

# 2. Update memory and latency numbers
content = re.sub(r"\b1\.2GB\b", "14MB (Cognitive Layer)", content)
content = re.sub(r"\b0 latency\b", "< 1ms real-time latency", content, flags=re.IGNORECASE)

# 3. Strip obsolete training architecture since v10 has no training layer.
# But preserve the academic explanation of why datasets were needed for perception.
content = content.replace("## **3.2 Training Architecture**", "## **3.2 Perception Layer Foundation**")

with open(filepath, "w") as f:
    f.write(content)
print("Updated ALM Project Documentation successfully.")
