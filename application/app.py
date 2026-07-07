import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import json
import librosa
from main import UnifiedPipelineValidator

print("Bootstrapping ALM v12.0 Unified Pipeline (Human Situation Understanding)...")
validator = UnifiedPipelineValidator()

def safe_serialize(obj):
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o))))
    except Exception as e:
        return {"serialization_error": str(e), "repr": str(obj)}

def run_alm_pipeline(audio_filepath: str):
    """
    Executes the ALM v12.0 deterministic pipeline:
    Audio -> Whisper/CLAP/HTS-AT -> Fusion -> Semantic -> HRE -> WSE -> SPE -> TRE -> SIR
    """
    if not audio_filepath or not os.path.exists(audio_filepath):
        yield ("Error: No audio provided.", "", "", {}, {}, {}, {}, "Error")
        return
        
    try:
        duration = librosa.get_duration(path=audio_filepath)
        if duration > 180:
            yield (
                "⚠️ **Upload Limit Exceeded**\n\nThe maximum allowed upload length is 3 minutes.",
                "⚠️ **Upload Limit Exceeded**\n\nThe maximum allowed upload length is 3 minutes.",
                "⚠️ **Upload Limit Exceeded**\n\nThe maximum allowed upload length is 3 minutes.",
                {}, {}, {}, {}, "Error: Upload > 3 minutes."
            )
            return
        if duration > 90:
            yield (
                "⚠️ **Analysis Limit Exceeded**\n\nTo ensure system stability, ALM can only analyze clips under 90 seconds. Your file was uploaded successfully, but please crop it before analyzing.",
                "⚠️ **Analysis Limit Exceeded**\n\nAudio is too long for deep analysis.",
                "⚠️ **Analysis Limit Exceeded**\n\nPlease crop the audio.",
                {}, {}, {}, {}, "Error: Analysis > 90 seconds."
            )
            return
    except Exception as e:
        pass

    # Yield initial loading state for better UX
    yield (
        "⏳ **Processing Speech...**\n\nRunning Neural Perception (Whisper)...",
        "⏳ **Analyzing Environment...**\n\nExtracting acoustic features (CLAP/HTS-AT)...",
        "⏳ **Fusing Intelligence...**\n\nRunning Semantic Engine & Reasoning...",
        {"status": "processing..."},
        {"status": "processing..."},
        {"status": "processing..."},
        {"status": "processing..."},
        "Running pipeline... This may take up to a minute depending on hardware."
    )
        
    try:
        audio, sr = librosa.load(audio_filepath, sr=16000)
        report = validator.run_pipeline(audio, sr)
        
        if report:
            speech_out = report.get("speech", "No speech detected.")
            env_out = report.get("environment", "No environment sounds detected.")
            sit_out = report.get("situation", "Situation unknown.")
            
            # Developer Mode objects
            latencies = report.get("latencies", {})
            process_status = (
                f"Neural: {latencies.get('NeuralPerception', 0):.1f}ms | "
                f"PSE: {latencies.get('PSE', 0):.1f}ms | "
                f"Fusion: {latencies.get('EvidenceFusion', 0):.1f}ms | "
                f"Semantic: {latencies.get('SemanticEngine', 0):.1f}ms | "
                f"HRE: {latencies.get('HRE_rank', 0):.1f}ms | "
                f"WSE: {latencies.get('WSE', 0):.1f}ms | "
                f"SPE: {latencies.get('SPE', 0):.1f}ms | "
                f"TRE: {latencies.get('TRE', 0):.1f}ms | "
                f"SIR: {latencies.get('SIR', 0):.1f}ms\n"
                f"Total: {sum(latencies.values()):.1f}ms"
            )
            
            if report.get("audio_evidence"):
                ae = report["audio_evidence"]
                audio_evidence_dict = ae.model_dump() if hasattr(ae, "model_dump") else ae.dict() if hasattr(ae, "dict") else safe_serialize(ae)
            else:
                audio_evidence_dict = {}
            
            if report.get("semantic_json"):
                sj = report["semantic_json"]
                semantic_json_dict = sj.model_dump() if hasattr(sj, "model_dump") else sj.dict() if hasattr(sj, "dict") else safe_serialize(sj)
            else:
                semantic_json_dict = {}
            
            active_hyps = []
            for h in report.get("active_hyps", []):
                active_hyps.append(h.model_dump() if hasattr(h, "model_dump") else h.dict() if hasattr(h, "dict") else safe_serialize(h))
            
            if report.get("world_state"):
                ws = report["world_state"]
                world_state_dict = ws.model_dump() if hasattr(ws, "model_dump") else ws.dict() if hasattr(ws, "dict") else safe_serialize(ws)
            else:
                world_state_dict = {}
            
            dev_evidence = safe_serialize(audio_evidence_dict)
            dev_semantic = safe_serialize(semantic_json_dict)
            dev_hyps = safe_serialize(active_hyps)
            dev_state = safe_serialize(world_state_dict)
            
            yield (
                speech_out,
                env_out,
                sit_out,
                dev_evidence,
                dev_semantic,
                dev_hyps,
                dev_state,
                process_status
            )
        else:
             yield ("Pipeline generated no output.", "", "", {}, {}, {}, {}, "N/A")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Pipeline Error: {str(e)}", "", "", {}, {}, {}, {}, "Error")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🎧 ALM v12.0: Human Situation Understanding")
    gr.Markdown("### Advanced Acoustic Language Model")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Audio Input")
            audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
            submit_btn = gr.Button("🚀 Analyze Audio Scene", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 🧠 Situation Intelligence Renderer")
            with gr.Tabs():
                with gr.Tab("🗣️ Speech Understanding"):
                    speech_ui = gr.Markdown("Waiting for audio...")
                with gr.Tab("🌍 Environmental Understanding"):
                    env_ui = gr.Markdown("Waiting for audio...")
                with gr.Tab("🧠 Situation Understanding"):
                    sit_ui = gr.Markdown("Waiting for audio...")
            
    gr.Markdown("---")
    
    with gr.Accordion("⚙️ Developer Mode (Under The Hood)", open=False):
        gr.Markdown("### Technical Debugging Information")
        processing_status = gr.Textbox(label="⏱️ Pipeline Processing Latencies", interactive=False)
        with gr.Tabs():
            with gr.Tab("Audio Evidence"):
                dev_ev = gr.JSON()
            with gr.Tab("Semantic Interpretation"):
                dev_sem = gr.JSON()
            with gr.Tab("Active Hypotheses"):
                dev_hyp = gr.JSON()
            with gr.Tab("Cognitive State"):
                dev_st = gr.JSON()

    submit_btn.click(
        fn=run_alm_pipeline,
        inputs=audio_input,
        outputs=[
            speech_ui,
            env_ui,
            sit_ui,
            dev_ev,
            dev_sem,
            dev_hyp,
            dev_st,
            processing_status
        ]
    )

if __name__ == "__main__":
    print("Launching ALM v12.0 Gradio Interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
