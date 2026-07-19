# Reproducibility Report
**Objective:** Identify gaps that could prevent another researcher from reproducing ALM's results.

## Reproducibility Checklist

| Item | Status | Action Required |
| :--- | :--- | :--- |
| **Environment Definition** | **WARNING** | `requirements.txt` lacks `==` version pinning. (e.g., `faster-whisper` instead of `faster-whisper==1.0.3`). This guarantees eventual breakage when dependencies update. |
| **Model Versions** | **PASS** | `feature_extractor.py` hardcodes `"large-v3"` for Whisper, ensuring the same weights are pulled. |
| **Prompt Versions** | **PASS** | Prompts are baked into the engine Python files and tracked by Git commits. |
| **Dataset Versions** | **PASS** | `hoasu_bench.json` is static and tracked in version control. |
| **Random Seeds** | **WARNING** | Qwen3-4B generation lacks explicit temperature=0.0 and fixed random seeds. This could cause slight variance in the qualitative JSON output across runs. |

## Recommendations
To achieve 100% academic reproducibility, the `requirements.txt` must be strictly pinned, and all LLM inference calls must enforce zero temperature and explicit seed seeding.
