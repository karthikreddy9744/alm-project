import gradio as gr
import numpy as np
from collections import deque
from core.inference_pipeline import ALMInferencePipeline

pipeline = ALMInferencePipeline()
scene_buffer = deque(maxlen=3)

def analyze_audio(audio_input):
    if audio_input is None:
        return 'No audio provided.', '', ''
    
    sr, audio = audio_input
    
    # VALIDATION: Catch corrupted/empty audio
    if audio is None or len(audio) == 0:
        return 'Invalid or empty audio file.', 'Error', 'Cannot process an empty audio signal.'
        
    audio = audio.astype(np.float32) / 32768.0
    
    # Convert stereo [N, channels] to mono [N]
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    try:
        transcript, scene, conf, ai_text = pipeline.run(audio, sr)
        
        # Live Smoothing Logic
        scene_buffer.append(scene)
        smoothed_scene = max(set(scene_buffer), key=scene_buffer.count)
        
    except Exception as e:
        return (f"Error during transcription: {str(e)}", 
                "Error", 
                "The pipeline encountered a critical failure. Please try a different audio file.")
    
    return (
        f'{transcript}' if transcript else '[No speech detected]',
        f'{smoothed_scene} ({conf:.0%} Confidence)',
        ai_text
    )

# Premium Custom Theme setup
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill_dark="#0f172a",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color_dark="#334155",
    block_radius="16px",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
    button_primary_border_color_dark="transparent"
)

with gr.Blocks(title='ALM — Audio Language Model') as demo:
    # Modern Glassmorphic Hero Banner
    gr.HTML('''
    <div style="text-align: center; padding: 2rem; margin-bottom: 2rem; background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(79,70,229,0.1) 100%); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(10px);">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem; background: -webkit-linear-gradient(45deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎧 Audio Language Model (ALM)</h1>
        <h3 style="color: #94a3b8; font-weight: 500; font-size: 1.25rem;">Listen • Think • Understand</h3>
        <p style="color: #64748b; margin-top: 1rem; max-width: 600px; margin-left: auto; margin-right: auto;">A context-aware AI capable of understanding the acoustic environment alongside human speech.</p>
    </div>
    ''')
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=['microphone', 'upload'],
                type='numpy',
                label='Input Audio (Streaming & Upload)'
            )
            analyze_btn = gr.Button('🔍 Analyze Audio Environment', variant='primary')
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Live Analysis"):
                    with gr.Row():
                        scene_out = gr.Textbox(label='Predicted Environment', lines=1)
                        transcript_out = gr.Textbox(label='Audio Transcript', lines=1)
                    ai_out = gr.Textbox(label='CASRE Scene Understanding & Action Engine', lines=6)
                
                with gr.TabItem("System Information"):
                    gr.Markdown("### ALM Architecture\n1. **Whisper Encoder:** Extracts 512-D speech features.\n2. **CLAP Encoder:** Extracts 512-D environmental features.\n3. **Fusion Layer:** Combines features (1024-D -> 256-D) using LayerNorm and Dropout.\n4. **Scene Network:** Classifies 15 robust environmental scenes.\n5. **CASRE:** Cross-modal context reasoning engine.")
    
    analyze_btn.click(
        analyze_audio,
        inputs=audio_input,
        outputs=[transcript_out, scene_out, ai_out]
    )

demo.queue(default_concurrency_limit=1)
demo.launch(share=True, theme=custom_theme)
