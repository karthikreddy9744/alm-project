import gradio as gr
import numpy as np
import traceback
import torch
import re
from collections import deque
from core.inference_pipeline import ALMInferencePipeline

pipeline = ALMInferencePipeline()
scene_buffer = deque(maxlen=3)

def format_casre_html(ai_text):
    if not ai_text or "CASRE SITUATIONAL ASSESSMENT" not in ai_text:
        return f"<div style='padding:1rem; color:#e2e8f0;'>{ai_text}</div>"
        
    sections_raw = ai_text.split("⸻\n")
    
    # 0 = Header
    # 1 = Evidence Strength Overview
    # 2 = Observations
    # 3 = Evidence Analysis Header
    # 4 = Speech Evidence
    # 5 = Environmental Evidence
    # 6 = Temporal Evidence
    # 7 = Media Evidence
    # 8 = Cross-Modal Relationship
    # 9 = Situation Assessment
    # 10 = Environmental Interpretation
    # 11 = Alternative Explanations
    # 12 = Uncertainty Analysis
    # 13 = Confidence Drivers
    # 14 = Confidence Limitations
    # 15 = Risk Assessment
    # 16 = Recommended Response
    # 17 = Analyst Conclusion

    def get_section(idx):
        return sections_raw[idx] if len(sections_raw) > idx else ""

    exec_summary = get_section(0)
    ev_strength = get_section(1)
    observations = get_section(2)
    
    sp_ev = get_section(4).replace("Speech Evidence\n", "").strip()
    en_ev = get_section(5).replace("Environmental Evidence\n", "").strip()
    tm_ev = get_section(6).replace("Temporal Evidence\n", "").strip()
    
    cross_modal = get_section(8).replace("Cross-Modal Relationship\n", "").strip()
    sit_assessment = get_section(9).replace("Situation Assessment\n", "").strip()
    env_interp = get_section(10).replace("Environmental Interpretation\n", "").strip()
    alt_exp = get_section(11).replace("Alternative Explanations\n", "").strip()
    uncertainty = get_section(12).replace("Uncertainty Analysis\n", "").strip()
    conf_drivers = get_section(13).replace("Confidence Drivers\n", "").strip()
    conf_limits = get_section(14).replace("Confidence Limitations\n", "").strip()
    risk_assessment = get_section(15).replace("Risk Assessment\n", "").strip()
    recommended = get_section(16).replace("Recommended Response\n", "").strip()
    analyst_conc = get_section(17).replace("Analyst Conclusion\n", "").strip()
    
    # Parse Situation
    scenario = "Unknown"
    for line in sit_assessment.split('\n'):
        if line.startswith("Most Likely Interpretation:"):
            scenario = line.replace("Most Likely Interpretation:", "").strip()
            
    # Determine Colors
    risk_color = "#10b981"
    if "Very High" in risk_assessment: risk_color = "#991b1b"
    elif "High" in risk_assessment: risk_color = "#ef4444"
    elif "Moderate" in risk_assessment: risk_color = "#f59e0b"
    
    # Evidence Strength extraction
    overall_strength = "Unknown"
    for line in ev_strength.split('\n'):
        if line.startswith("Overall Evidence Strength:"):
            overall_strength = line.replace("Overall Evidence Strength:", "").strip()
            
    def get_strength_color(strength_str):
        if "Strong" in strength_str: return "#10b981"
        if "Moderate" in strength_str: return "#f59e0b"
        if "Weak" in strength_str: return "#ef4444"
        return "#64748b"

    def format_list(text, icon=""):
        items = [line.strip().lstrip('*').strip() for line in text.split('\n') if line.strip().startswith('*')]
        if not items: return "<p style='color: #64748b; font-size:0.9rem;'>None</p>"
        html = "<ul style='list-style-type: none; padding-left: 0; margin-top:0.5rem;'>"
        for item in items:
            html += f"<li style='margin-bottom:0.4rem; font-size:0.95rem;'><span style='margin-right:8px;'>{icon}</span>{item}</li>"
        html += "</ul>"
        return html

    def build_list(text, icon):
        items = [line.replace("✓", "").replace("✗", "").strip() for line in text.split('\n') if line.strip()]
        if not items or items == ["None"]: return "<p style='color: #64748b; font-size:0.9rem;'>None</p>"
        html = "<ul style='list-style-type: none; padding-left: 0; margin-top:0.5rem;'>"
        for item in items:
            html += f"<li style='margin-bottom:0.4rem; font-size:0.95rem;'><span style='margin-right:8px;'>{icon}</span>{item}</li>"
        html += "</ul>"
        return html
        
    supp_html = build_list(conf_drivers, "✅")
    contra_html = build_list(conf_limits, "❌")
    obs_html = format_list(observations, "•")
    unc_html = format_list(uncertainty, "⚠️")

    transcript = "None"
    for line in sp_ev.split('\n'):
        if line.startswith("Transcript:"): transcript = line.replace("Transcript:", "").strip()
        
    cross_modal_text = cross_modal.replace("\n", "<br/>")

    def format_raw_text(text):
        styled_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                styled_lines.append("<div style='height: 0.5rem;'></div>")
            elif line == "⸻":
                styled_lines.append("<hr style='border: none; border-top: 1px dashed #475569; margin: 0.75rem 0;'/>")
            elif line == "CASRE SITUATIONAL ASSESSMENT":
                styled_lines.append(f"<div style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin-top: 1rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;'>{line}</div>")
            elif line in ["Executive Summary", "Evidence Strength Overview", "Observations", "Evidence Analysis", "Speech Evidence", "Environmental Evidence", "Temporal Evidence", "Media Evidence", "Cross-Modal Relationship", "Situation Assessment", "Environmental Interpretation", "Alternative Explanations", "Uncertainty Analysis", "Confidence Drivers", "Confidence Limitations", "Risk Assessment", "Recommended Response", "Analyst Conclusion", "Temporal Event Timeline:"]:
                styled_lines.append(f"<div style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-top: 0.75rem; margin-bottom: 0.25rem;'>{line}</div>")
            elif line.startswith("* "):
                styled_lines.append(f"<div style='margin-left: 1rem; color: #94a3b8; font-size: 0.9rem; line-height: 1.4;'><span style='color:#60a5fa; margin-right:6px;'>•</span>{line[2:]}</div>")
            elif line.startswith("✓ "):
                styled_lines.append(f"<div style='margin-left: 1rem; color: #10b981; font-size: 0.9rem; line-height: 1.4;'><span style='margin-right:6px;'>✓</span>{line[2:]}</div>")
            elif line.startswith("✗ "):
                styled_lines.append(f"<div style='margin-left: 1rem; color: #ef4444; font-size: 0.9rem; line-height: 1.4;'><span style='margin-right:6px;'>✗</span>{line[2:]}</div>")
            elif ":" in line:
                key, val = line.split(":", 1)
                styled_lines.append(f"<div style='font-size: 0.9rem; line-height: 1.4;'><strong style='color: #94a3b8; font-weight: 600;'>{key}:</strong> <span style='color: #e2e8f0;'>{val}</span></div>")
            else:
                styled_lines.append(f"<div style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.4;'>{line}</div>")
        return "".join(styled_lines)
        
    styled_raw_text = format_raw_text(ai_text)


    html = f'''
    <div style="font-family: 'Inter', sans-serif; color: #e2e8f0;">
        <div style="background: #0f172a; border: 1px solid #334155; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <h4 style="text-transform: uppercase; font-size: 0.75rem; font-weight: 700; color: #94a3b8; margin-bottom: 0.5rem; letter-spacing: 0.05em;">Situation Assessment</h4>
            <h2 style="font-size: 1.5rem; font-weight: 600; color: #f8fafc; margin-bottom: 1rem;">{scenario}</h2>
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 0.5rem;">
                <span style="background: {get_strength_color(overall_strength)}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">Evidence: {overall_strength}</span>
                <span style="background: {risk_color}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">Risk: {risk_assessment}</span>
                <span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">Action: {recommended}</span>
            </div>
            
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #334155;">
                <h4 style="font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.5rem;">Evidence Strength Breakdown</h4>
                <div style="font-size: 0.85rem; color: #94a3b8; white-space: pre-wrap; font-family: monospace;">{ev_strength.replace("Evidence Strength Overview\n", "").strip()}</div>
            </div>
        </div>
        
        <div style="background: rgba(59, 130, 246, 0.05); border-left: 4px solid #3b82f6; padding: 1rem 1.5rem; margin-bottom: 1.5rem; border-radius: 0 12px 12px 0;">
            <h4 style="font-size: 0.9rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Analyst Conclusion</h4>
            <p style="color: #e2e8f0; font-size: 1rem; line-height: 1.6;">{analyst_conc}</p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.5rem;">Cross-Modal Relationship</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">{cross_modal_text}</p>
            </div>
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.5rem;">Environmental Interpretation</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">{env_interp}</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 12px; padding: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 600; color: #10b981; margin-bottom: 0.5rem;">Confidence Drivers</h4>
                {supp_html}
            </div>
            <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 600; color: #ef4444; margin-bottom: 0.5rem;">Confidence Limitations</h4>
                {contra_html}
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 12px; padding: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 600; color: #f59e0b; margin-bottom: 0.5rem;">Uncertainty Analysis</h4>
                {unc_html}
            </div>
        </div>

        <details style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; cursor: pointer;">
            <summary style="font-weight: 600; font-size: 0.9rem; color: #cbd5e1; outline: none;">🎙️ View Transcript & Observations</summary>
            <div style="padding-top: 0.75rem; font-size: 0.95rem; color: #f8fafc; font-style: italic; margin-bottom: 1rem;">"{transcript}"</div>
            <h4 style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.5rem;">Observations</h4>
            {obs_html}
        </details>
        
        <details style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 0.75rem 1rem; cursor: pointer;">
            <summary style="font-weight: 600; font-size: 0.9rem; color: #cbd5e1; outline: none;">⚙️ Technical Details (Raw output)</summary>
            <div style="padding-top: 0.75rem; font-family: 'Inter', sans-serif;">{styled_raw_text}</div>
        </details>
    </div>
    '''
    return html
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
        msnl_out, scene, conf, ai_text = pipeline.run(audio, sr)
        transcript = msnl_out["original_transcript"]
        
        # Live Smoothing Logic (Removed for full timeline analysis in v4.0)
        smoothed_scene = scene
        
    except Exception as e:
        print("Pipeline Error:")
        traceback.print_exc()
        
        # Free up GPU/MPS memory to prevent the app from freezing on subsequent runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        return (f"Error during analysis: {str(e)}\n\n(System memory was automatically cleared to prevent crashing.)", 
                "Error", 
                "The pipeline encountered a critical failure. The system memory has been reset safely. Please try a different audio file.")
    
    return (
        f'{transcript}' if transcript else '[No speech detected]',
        f'{smoothed_scene} ({conf*100:.1f}% Max Confidence)',
        format_casre_html(ai_text),
        msnl_out.get("detected_language", "en"),
        msnl_out.get("semantic_transcript", ""),
        f'{msnl_out.get("translation_confidence", 1.0):.2f}'
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
                        transcript_out = gr.Textbox(label='Audio Transcript (Original)', lines=1)
                    with gr.Row():
                        msnl_lang_out = gr.Textbox(label='Detected Language', lines=1)
                        msnl_semantic_out = gr.Textbox(label='Semantic Transcript (En)', lines=1)
                        msnl_conf_out = gr.Textbox(label='Translation Confidence', lines=1)
                    ai_out = gr.HTML(label='Situational Assessment')
                
                with gr.TabItem("System Information"):
                    gr.Markdown("### ALM v7.0 Architecture\\n1. **Audio Frontend:** Silero VAD + LUFS Normalization.\\n2. **Whisper Encoder:** VAD-guided transcript extraction.\\n3. **CLAP Encoder:** 512-D environmental feature extraction.\\n4. **Fusion Layer:** V7.0 MLP Layer.\\n5. **Scene Network:** Multi-label environment prediction (40 categories).\\n6. **MSNL:** Multilingual Speech Normalization Layer for translation to English.\\n7. **CASRE Engine:** Media playback detection, risk scoring, and timeline reasoning.")
    
    analyze_btn.click(
        analyze_audio,
        inputs=audio_input,
        outputs=[transcript_out, scene_out, ai_out, msnl_lang_out, msnl_semantic_out, msnl_conf_out]
    )

demo.queue(default_concurrency_limit=1)
demo.launch(share=True, theme=custom_theme)
