import os
import gradio as gr
import json
import librosa
from main import UnifiedPipelineValidator

print("Bootstrapping ALM v10.7 Unified Pipeline (Neural Perception + Deterministic Graph)...")
validator = UnifiedPipelineValidator()

def run_alm_pipeline(audio_filepath: str):
    """
    Executes the ALM v10.7 deterministic pipeline:
    Audio -> Whisper/CLAP/HTS-AT -> Fusion -> SceneNet -> AWM -> ... -> SIR
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        return "Error: No audio provided.", {}, "N/A", "N/A", "N/A", "N/A"
        
    try:
        audio, sr = librosa.load(audio_filepath, sr=16000)
        report = validator.run_pipeline(audio, sr)
        
        human_summary = report if report else "Processing completed but no critical events triggered the SIR."
        
        if validator.wse.current_state:
            world_state_status = validator.wse.current_state.dominant_state
            world_state_dict = {
                "id": validator.wse.current_state.id,
                "dominant_state": validator.wse.current_state.dominant_state,
                "confidence": validator.wse.current_state.confidence.__dict__ if validator.wse.current_state.confidence else {},
                "ambiguity": validator.wse.current_state.ambiguity_score,
                "consistency": validator.wse.current_state.consistency_score,
                "environmental_context": validator.wse.current_state.environmental_context,
                "speech_context": validator.wse.current_state.speech_context
            }
        else:
            world_state_status = "UNKNOWN"
            world_state_dict = {}
        
        # Format the specific requested metrics
        
        # Transcript & Confidence
        speech_transcript = "No speech detected."
        speech_entities = [v for v in validator.awm.entities.values() if v.entity_type == "Speaker"]
        if speech_entities:
            spk = speech_entities[0]
            # using getattr because transcript is a dynamic attribute we added in pipeline
            speech_transcript = getattr(spk, 'transcript', 'Speech detected, no transcript.')
            
        # Active Sound Events & Confidences
        active_events = []
        for v in validator.awm.events.values():
            conf_str = f"Detection: {v.confidence.sound_detection:.2f}"
            active_events.append(f"- {v.class_map} (Salience: {v.acoustic_salience:.2f} | {conf_str})")
            
        active_events_str = "\n".join(active_events) if active_events else "No significant background events."
        
        # Pipeline processing status / latency
        latencies = validator.latencies
        process_status = (
            f"Extraction: {latencies.get('extraction_ms', 0):.1f}ms | "
            f"Fusion/Scene: {latencies.get('neural_fusion_ms', 0):.1f}ms | "
            f"AWM Graph: {latencies.get('awm_ms', 0):.1f}ms | "
            f"Reasoning: {latencies.get('reasoning_engine_ms', 0):.1f}ms | "
            f"Renderer: {latencies.get('renderer_ms', 0):.1f}ms\n"
            f"Total: {sum(latencies.values()):.1f}ms"
        )
        
        trace_json = {
            "AWM_Events": [v.class_map for v in validator.awm.events.values()],
            "AWM_Entities": [v.entity_type for v in validator.awm.entities.values()],
            "Active_Hypotheses": [h.statement for h in validator.hre.active_hypotheses],
            "Latencies_ms": validator.latencies
        }
        
        return json.dumps(trace_json, indent=4), world_state_dict, human_summary, world_state_status, active_events_str, speech_transcript, process_status

    except Exception as e:
        return f"Pipeline Error: {str(e)}", {}, "Error", "ERROR", "ERROR", "ERROR", "ERROR"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🎧 ALM v10.7: Acoustic Language Model")
    gr.Markdown("### Unified Architecture: Neural Perception + Deterministic Cognitive Graph")
    gr.Markdown("**Pipeline Phase Flow:** Audio → Frozen Foundation Models (Whisper/CLAP/HTS-AT) → Trainable Fusion Layer & Scene Context Network → Deterministic Reasoning Engine (AWM → SIR)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Acoustic Input")
            audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
            submit_btn = gr.Button("🚀 Execute ALM Cognitive Pipeline", variant="primary")
            
            gr.Markdown("### 2. Extracted Neural Perception")
            speech_transcript_display = gr.Textbox(label="🗣️ Speech Transcript (Whisper)", interactive=False)
            active_events_display = gr.Textbox(label="🔊 Active Sound Events & Confidences (HTS-AT / CLAP)", interactive=False, lines=5)
            
        with gr.Column(scale=2):
            gr.Markdown("### 3. Cognitive Reasoning Output")
            status_display = gr.Textbox(label="🚨 World State Engine (WSE) Dominant Status", text_align="center")
            human_report = gr.Textbox(label="📄 Situation Intelligence Renderer (Human-Readable Report)", lines=12)
            processing_status = gr.Textbox(label="⏱️ Pipeline Processing Latencies", interactive=False)
            
    gr.Markdown("---")
    gr.Markdown("### 4. Transparent Developer Trace (Under The Hood)")
    with gr.Accordion("🔍 View Internal Graph States (AWM / WSE / TRE)", open=False):
        with gr.Row():
            with gr.Column():
                world_state_json = gr.JSON(label="World State Object (WSE)")
            with gr.Column():
                raw_json = gr.Code(label="Transparent Reasoning Engine Trace (TRE JSON)", language="json")

    submit_btn.click(
        fn=run_alm_pipeline,
        inputs=audio_input,
        outputs=[
            raw_json, 
            world_state_json, 
            human_report, 
            status_display, 
            active_events_display, 
            speech_transcript_display, 
            processing_status
        ]
    )

if __name__ == "__main__":
    print("Launching ALM v10.7 Gradio Interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
