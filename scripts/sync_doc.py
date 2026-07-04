import re

filepath = "documentation/ALM Project Documentation.md"

with open(filepath, "r") as f:
    content = f.read()

# Terminology Replacements
replacements = {
    "VERSION 7.0": "VERSION 10.4",
    "Version 7.0": "Version 10.4",
    "v7.0": "v10.4",
    "ALM v7.0": "ALM v10.4",
    "Context-Aware Smart Response Engine (CASRE)": "Cognitive Audio Scene Reasoning Engine (CASRE)",
    "CASRE Engine": "CASRE Pipeline",
    "Report Generator": "Situation Intelligence Renderer (SIR)",
    "Omni-Matrix": "Six-Stage Cognitive Pipeline",
    "Neural Network Fusion Layer": "Auditory World Model (AWM)",
    "Scene Context Network": "Hypothesis Reasoning Engine (HRE)",
    "Multilingual Speech Normalization Layer": "Auditory World Model",
    "MSNL": "AWM",
    "whisper-base": "Whisper Large-v3 Turbo (INT8)",
    "whisper-small": "Whisper Large-v3 Turbo (INT8)"
}

for old, new in replacements.items():
    content = content.replace(old, new)

# We need to add the new versions to the Revision History
history_addition = """| 8.0         | June 2026 | Re-architecture: Replaced trained fusion layers with AWM and cognitive rules. |
| 9.0         | July 2026 | Introduced Belief State Engine (BSE) and Hypothesis Reasoning Engine (HRE). |
| 10.0        | July 2026 | Added World State Estimation (WSE), Situation Projection Engine (SPE), and Transparent Reasoning Engine (TRE). |
| 10.4        | July 2026 | Release Candidate: Centralized configuration, full repository audit, and strict O(1) determinism. |
"""

if "| 10.4" not in content:
    content = content.replace("| 7.2", history_addition + "| 7.2")

# We should save it back
with open(filepath, "w") as f:
    f.write(content)

print("Basic terminology replacements done.")
