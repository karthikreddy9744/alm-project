# ALM v12.0 Ablation Studies

**Objective:** To empirically demonstrate that the ALM Structured Cognitive Reasoning Architecture provides mathematically measurable improvements over end-to-end opaque LLMs in the domains of hallucination reduction, transparency, and provenance verification.

The ablation studies systematically dismantle the ALM architecture component by component, isolating the exact cognitive failures that occur when structured reasoning is removed.

---

## 1. Full ALM (The Anchor Baseline)

**Hypothesis:** The complete v12.0 ALM architecture provides the highest Traceability Index (TI) and lowest Hallucination Rate (HR) of any tested configuration due to its strict cognitive pipeline constraints.
- **Variables:** 
  - *Independent:* All cognitive engines (Semantic, HRE, WSE, SPE, TRE, SIR) active.
  - *Dependent:* Benchmark performance across all cognitive metrics.
- **Dataset:** Complete HOASU-Bench (250 E2E samples).
- **Metrics:** Traceability Index (TI), Hallucination Rate (HR), Situation Accuracy (SA), Cross-Modal Agreement Rate (CMAR), Provenance Accuracy (PA).
- **Expected Observations:** Near 100% TI and sub-5% HR, with peak SA across complex adversarial environments.
- **Failure Modes:** Rare Perceptual Masking failures (due to sensory layer limitations).
- **Statistical Validation:** This configuration serves as the statistical anchor ($p$-values and Confidence Intervals for all other ablations are computed against this baseline).

---

## 2. ALM without Schema Constraints

**Hypothesis:** Removing strict Pydantic JSON schema constraints allows the LLM to output unstructured natural language, leading to rapid drift into hallucination and ungrounded assumptions.
- **Variables:**
  - *Independent:* Removal of structured JSON enforcement in the Semantic Interpretation Engine.
  - *Dependent:* Hallucination Rate (HR), Situation Accuracy (SA).
- **Dataset:** Daily Life and Public Events subsets of HOASU-Bench (100 samples).
- **Metrics:** HR, SA.
- **Expected Observations:** A significant spike (+40%) in HR as the model defaults to standard conversational priors rather than acoustic evidence.
- **Failure Modes:** The model will invent visual details (e.g., describing a "red car") because standard NLP datasets prioritize descriptive storytelling over evidence-based deduction.
- **Statistical Validation:** Wilcoxon signed-rank test predicting $p < 0.01$ degradation in HR compared to Full ALM.

---

## 3. ALM without Provenance Reasoning

**Hypothesis:** Skipping the explicit Provenance Classification step (`source_type` estimation) renders the model gullible to synthetic, fictional, and manipulated audio.
- **Variables:**
  - *Independent:* Bypassing the Provenance Reasoning prompt constraint.
  - *Dependent:* Provenance Accuracy (PA), Situation Accuracy (SA).
- **Dataset:** Synthetic, Movie, and Documentary subsets of HOASU-Bench (110 samples).
- **Metrics:** PA, SA.
- **Expected Observations:** The model will misclassify Movie scenes as Real-world emergencies (violating the Intellectual Property Rule) and accept Synthetic AI speech as legitimate human interaction.
- **Failure Modes:** Panic/false alarms triggered by entertainment audio (e.g., reporting a terrorist attack when listening to an action movie trailer).
- **Statistical Validation:** Bootstrapped 95% CI indicating a massive drop in PA on non-Real-world subsets.

---

## 4. ALM without Cross-Modal Verification

**Hypothesis:** Removing the "Evidence Dominates Assumptions" conflict-resolution step causes the model to forcibly reconcile conflicting audio tracks (e.g., Speech vs. Environment) through hallucinated context.
- **Variables:**
  - *Independent:* Disabling the `cross_modal_assessment` phase in the Semantic Engine.
  - *Dependent:* Cross-Modal Agreement Rate (CMAR), Hallucination Rate (HR).
- **Dataset:** Mixed Provenance / Contradictory subset of HOASU-Bench (30 samples).
- **Metrics:** CMAR, HR.
- **Expected Observations:** Instead of flagging an audio file as "Contradictory / Fake", the model will invent bizarre scenarios (e.g., "The person is on a train but someone is mowing the lawn inside the train").
- **Failure Modes:** Absolute failure to detect Deepfakes and audio splicing.
- **Statistical Validation:** Wilcoxon signed-rank test predicting a near 100% failure rate in identifying Contradictory data vs. Full ALM.

---

## 5. ALM without Hypothesis Reasoning (HRE)

**Hypothesis:** Removing the Multi-Agent Hypothesis Competition (forcing the Semantic Engine to commit to its very first interpretation) significantly reduces accuracy in high-ambiguity environments.
- **Variables:**
  - *Independent:* Disabling the HRE, forcing a single-pass `primary_situation` output.
  - *Dependent:* Situation Accuracy (SA), Traceability Index (TI).
- **Dataset:** 'Hard' difficulty subset of HOASU-Bench (approx. 80 samples).
- **Metrics:** SA, TI.
- **Expected Observations:** A noticeable drop in SA on highly ambiguous audio, as the model cannot generate and refute alternative explanations.
- **Failure Modes:** Premature cognitive commitment to an incorrect conclusion.
- **Statistical Validation:** Wilcoxon signed-rank test ($p < 0.05$) proving SA degradation on the 'Hard' subset.

---

## 6. Opaque Baseline: Direct Audio → LLM

**Hypothesis:** A standard multimodal Audio-LLM lacks explicit tracing, rendering its outputs completely unexplainable and structurally opaque.
- **Variables:**
  - *Independent:* End-to-end unstructured inference using a SOTA multimodal model (e.g., Qwen-Audio).
  - *Dependent:* Traceability Index (TI), Hallucination Rate (HR).
- **Dataset:** Full HOASU-Bench.
- **Metrics:** TI, HR.
- **Expected Observations:** TI drops to exactly 0%, as the neural architecture lacks any discrete intermediate sensory graph (`AudioEvidenceObject`). HR remains moderately high due to the lack of evidence constraints.
- **Failure Modes:** "Black-box" phenomenon—when the model makes a mistake, researchers cannot debug whether the failure occurred in audio perception or semantic reasoning.
- **Statistical Validation:** Baseline comparison anchor.

---

## 7. Opaque Baseline: Chain-of-Thought (CoT) Prompting

**Hypothesis:** While prompting an Opaque LLM to "think step-by-step" improves reasoning, it cannot artificially generate mathematical Traceability without a deterministic perceptual graph architecture.
- **Variables:**
  - *Independent:* Zero-shot Chain-of-Thought prompting applied to the Direct Audio → LLM baseline.
  - *Dependent:* Situation Accuracy (SA), Traceability Index (TI).
- **Dataset:** Full HOASU-Bench.
- **Metrics:** SA, TI.
- **Expected Observations:** SA will improve relative to Baseline #6, but TI will remain 0% because the "thoughts" are still latent token generations rather than physical mappings to objective sensory extractors.
- **Failure Modes:** The model will confidently generate a logical step-by-step reasoning path that is entirely disconnected from the actual sensory reality (CoT Hallucination).
- **Statistical Validation:** Wilcoxon signed-rank test proving that while SA matches Full ALM, TI is statistically non-existent.
