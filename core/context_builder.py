# core/context_builder.py — ALM v2.0
# Context-Aware Smart Response Engine (CASRE)

SCENE_LABELS = [
    'Emergency', 'Traffic', 'Weather', 'Water', 'Wildlife & Animals',
    'Indoor/Domestic', 'Home Appliances', 'Office/Work', 'Human Crowd',
    'Human Speech & Non-speech', 'Tools & Construction', 'Explosions & Weaponry',
    'Music & Bells', 'Footsteps', 'Silence/Unknown'
]

# ── Keyword banks for transcript semantic analysis ──
EMERGENCY_KEYWORDS = [
    'help', 'fire', 'danger', 'emergency', 'hurt', 'injured',
    'attack', 'save', 'blood', 'ambulance', 'police', 'stop',
    'no', 'please', 'run', 'escape', 'call', 'crash'
]

CALM_KEYWORDS = [
    'okay', 'fine', 'good', 'great', 'hello', 'hi', 'thanks',
    'yes', 'alright', 'sure', 'ready', 'done', 'nice'
]

QUESTION_KEYWORDS = [
    'what', 'where', 'when', 'who', 'why', 'how', 'is', 'are',
    'can', 'could', 'should', 'would', 'do', 'does'
]

# ── Scene base templates ──
SCENE_TEMPLATES = {
    'Emergency': {
        'description': 'An emergency situation has been detected in the audio environment.',
        'context': 'Background acoustic signatures indicate active emergency response conditions.',
        'action': 'Contact emergency services immediately. Clear the surrounding area if safe to do so.'
    },
    'Traffic': {
        'description': 'An active traffic environment has been detected.',
        'context': 'Urban or roadway acoustic signatures are present in the background.',
        'action': 'Exercise caution if near roadways. Ensure audio communication is clear.'
    },
    'Weather': {
        'description': 'A weather-related event is detected.',
        'context': 'Acoustic signatures such as rain, wind, or a thunderstorm are present.',
        'action': 'No immediate action required unless weather becomes severe. Seek shelter if necessary.'
    },
    'Water': {
        'description': 'Water-related sounds are detected.',
        'context': 'Sounds such as sea waves, flowing streams, or water drops are present.',
        'action': 'No immediate action required.'
    },
    'Wildlife & Animals': {
        'description': 'Wildlife or animal sounds are detected.',
        'context': 'Acoustic elements from animals such as dogs, birds, or insects are present.',
        'action': 'No immediate action required. Outdoor ambient conditions are normal.'
    },
    'Indoor/Domestic': {
        'description': 'An indoor or domestic environment has been detected.',
        'context': 'Enclosed space acoustic characteristics (e.g., doors, clocks) are present.',
        'action': 'Standard indoor environment. No specific action required.'
    },
    'Home Appliances': {
        'description': 'Home appliance activity detected.',
        'context': 'Background noises correspond to appliances like vacuum cleaners or washing machines.',
        'action': 'Standard indoor environment. No specific action required.'
    },
    'Office/Work': {
        'description': 'An office or workspace environment detected.',
        'context': 'Sounds like keyboard typing, mouse clicks, or printers are present.',
        'action': 'Standard work environment. No specific action required.'
    },
    'Human Crowd': {
        'description': 'A crowded environment with multiple people has been detected.',
        'context': 'Multiple human voices and crowd activity are present in the background.',
        'action': 'Be aware of surroundings. Communication may be affected by crowd noise.'
    },
    'Human Speech & Non-speech': {
        'description': 'Human vocal activity detected.',
        'context': 'Sounds such as breathing, coughing, crying, or other non-verbal vocalizations are present.',
        'action': 'Check on the individual if distress sounds (like crying or severe coughing) are heard.'
    },
    'Tools & Construction': {
        'description': 'Construction or mechanical tool activity detected.',
        'context': 'Acoustic signatures match tools like chainsaws, drills, or hand saws.',
        'action': 'Exercise caution around active machinery. Ear protection may be advised.'
    },
    'Explosions & Weaponry': {
        'description': 'Extremely hazardous sounds such as explosions or gunshots detected.',
        'context': 'High-impact explosive or ballistic acoustic signatures are present.',
        'action': 'IMMEDIATE DANGER: Seek cover immediately, evacuate if possible, and contact authorities.'
    },
    'Music & Bells': {
        'description': 'Musical instruments or bells detected.',
        'context': 'Acoustic environment contains musical elements or ringing bells.',
        'action': 'No immediate action required. Enjoy the music.'
    },
    'Footsteps': {
        'description': 'Footstep sounds detected.',
        'context': 'Rhythmic walking or running sounds are present in the environment.',
        'action': 'No immediate action required unless unauthorized entry is suspected.'
    },
    'Silence/Unknown': {
        'description': 'No distinct environmental category detected or environment is relatively silent.',
        'context': 'The acoustic signatures do not strongly match any known category.',
        'action': 'No specific action required.'
    }
}

import re

def _analyze_transcript(transcript: str) -> dict:
    """Analyse transcript for semantic content."""
    if not transcript or len(transcript.strip()) < 2:
        return {'type': 'empty', 'urgency': 'none', 'keywords_found': []}
    
    t_lower = transcript.lower()
    
    def find_matches(words, text):
        return [w for w in words if re.search(rf'\b{w}\b', text)]
        
    found_emergency = find_matches(EMERGENCY_KEYWORDS, t_lower)
    found_calm = find_matches(CALM_KEYWORDS, t_lower)
    
    # Better question detection
    is_question = transcript.strip().endswith('?') or any(t_lower.startswith(q) for q in QUESTION_KEYWORDS)
    found_question = ['question_format'] if is_question else []
    
    if found_emergency:
        return {'type': 'distress', 'urgency': 'high', 'keywords_found': found_emergency}
    elif found_question:
        return {'type': 'question', 'urgency': 'low', 'keywords_found': found_question}
    elif found_calm:
        return {'type': 'calm', 'urgency': 'none', 'keywords_found': found_calm}
    else:
        return {'type': 'neutral', 'urgency': 'medium', 'keywords_found': []}

def _get_confidence_tone(confidence: float) -> str:
    """Return tone descriptor based on confidence level."""
    if confidence >= 0.80: return 'high'
    elif confidence >= 0.50: return 'medium'
    else: return 'low'

def _build_speech_context(transcript: str, t_analysis: dict) -> str:
    """Build the speech portion of the response."""
    if t_analysis['type'] == 'empty':
        return 'No speech was detected in this audio clip.'
    
    q = '"' + transcript.strip() + '"'
    if t_analysis['type'] == 'distress':
        return f'Speech detected: {q}. The spoken content indicates distress or urgency.'
    elif t_analysis['type'] == 'question':
        return f'Speech detected: {q}. The spoken content contains a question or inquiry.'
    elif t_analysis['type'] == 'calm':
        return f'Speech detected: {q}. The spoken content appears calm and conversational.'
    else:
        return f'Speech detected: {q}.'

def _build_cross_modal_insight(scene: str, t_analysis: dict, confidence: float, t_conf: float) -> str:
    """Core cross-modal fusion logic — combines speech + scene."""
    tone = _get_confidence_tone(confidence)
    
    high_alert_scenes = ('Emergency', 'Explosions & Weaponry')
    non_human_scenes = ('Wildlife & Animals', 'Water', 'Weather', 'Home Appliances')
    
    # Contradiction Detection
    if scene in non_human_scenes and t_analysis['urgency'] == 'high':
        return ('> CONTRADICTION OVERRIDE: Environment classified as non-human, '
                'but strict human distress detected in speech. '
                'Prioritizing Speech context -> EMERGENCY ASSUMED.')
    
    # Emergency scene + distress speech → highest alert
    if scene in high_alert_scenes and t_analysis['urgency'] == 'high':
        return ('Both speech content and environmental audio confirm a critical emergency. '
                'Speech indicates direct distress. Acoustic environment matches active dangerous conditions.')
    
    # Emergency scene + calm/no speech → environmental only
    if scene in high_alert_scenes and t_analysis['urgency'] in ('none', 'low'):
        return ('Environmental audio indicates dangerous conditions despite calm or absent speech. '
                'Critical acoustic signatures are detected.')
    
    # Non-emergency scene + distress speech → speech overrides
    if scene not in high_alert_scenes and t_analysis['urgency'] == 'high':
        return (f'Speech content indicates distress within a {scene.lower()} environment. '
                'This combination warrants attention — the environmental context may be misleading.')
    
    # Low confidence — flag uncertainty
    if tone == 'low':
        return (f'Scene classification confidence is low ({confidence:.0%}). '
                'Environmental audio characteristics are ambiguous. Manual review recommended.')
    
    # Default — scene + speech coherent
    return (f'Audio analysis indicates a {scene.lower()} environment with '
            f'{confidence:.0%} confidence. Speech and environmental features are consistent.')

def generate_response(transcript: str, scene_class: str,
                      confidence: float, scene_probs: list, t_conf: float = 1.0) -> str:
    """
    Main CASRE entry point.
    Returns full natural language scene understanding.
    """
    t_analysis = _analyze_transcript(transcript)
    template = SCENE_TEMPLATES[scene_class]
    speech_ctx = _build_speech_context(transcript, t_analysis)
    insight = _build_cross_modal_insight(scene_class, t_analysis, confidence, t_conf)
    
    # Top-3 scene probabilities
    top3 = sorted(enumerate(scene_probs), key=lambda x: -x[1])[:3]
    top3_str = ', '.join(f'{SCENE_LABELS[i]} ({p:.0%})' for i,p in top3)
    
    response = (
        f'Scene Summary:\n'
        f'{template["description"]}\n\n'
        f'Speech Analysis:\n'
        f'{speech_ctx}\n\n'
        f'Environmental Context:\n'
        f'{template["context"]}\n\n'
        f'Cross-Modal Understanding:\n'
        f'{insight}\n\n'
        f'Scene Distribution: {top3_str}\n\n'
        f'Recommended Action:\n'
        f'{template["action"]}'
    )
    return response