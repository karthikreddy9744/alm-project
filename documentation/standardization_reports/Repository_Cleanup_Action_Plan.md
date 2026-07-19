# Repository Cleanup Action Plan
**Objective:** A phased, non-destructive roadmap for standardizing the repository.

## Guiding Rule
**NOTHING WILL BE DELETED AUTOMATICALLY.** All recommendations below must be executed by the researcher explicitly, following scientific justification.

## Phase 1: Immediate Actions (High Priority)
1. **Remove Duplicate Dependencies:** 
   - *Action:* Delete `configuration/requirements.txt`.
   - *Justification:* Prevents dependency drift between local and Colab environments.
2. **Consolidate Audio Assets:**
   - *Action:* Move all files from `samples/` (e.g., `Loki.mp3`, `test.wav`) into `datasets/`. Remove the `samples/` directory.
   - *Justification:* Enforces a single source of truth for audio inference.
3. **Merge Execution Scripts:**
   - *Action:* Move `scripts/benchmark.py` and `scripts/validation_suite.py` into `research/`. Delete `scripts/`.
   - *Justification:* Prevents fragmentation of evaluation logic.

## Phase 2: Short-Term Actions
1. **Archive Legacy Data Folders:**
   - *Action:* Move the empty `data/` and `models/` folders to `archive/` or delete them entirely via `.gitignore`.
   - *Justification:* Cleans up the root directory for new researchers.
2. **Standardize the README:**
   - *Action:* Replace the contents of `README.md` with a direct link or summary of `documentation/ALM_MiniProject.md`.
   - *Justification:* Prevents contradictions where the README describes v9 while the spec describes v12.

## Phase 3: Long-Term Maintenance Actions
1. **Formalize the Application GUI:**
   - *Action:* The `application/app.py` Gradio interface currently sits outside the core pipeline flow. It should be refactored to explicitly import `UnifiedPipelineValidator` from `core_modules/`.
2. **Establish `COMMANDS` File:**
   - *Action:* Move the raw text from `COMMANDS` into a formalized `documentation/11_Execution.md` or a `Makefile`.

## Future Archive Actions
Any file ending in `.pt`, `.ckpt`, or related to "training loops" discovered in the future must be immediately sequestered to `archive/` to prevent contamination of the zero-shot pipeline.
