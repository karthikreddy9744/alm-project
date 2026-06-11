import numpy as np

# Multi-label taxonomy mapping
SCENE_LABELS = [
    "Traffic & Vehicles",
    "Siren & Alarm",
    "Crowd & Hubbub",
    "Weather & Nature",
    "Water",
    "Indoor / Domestic",
    "Office",
    "Construction",
    "Wildlife / Animals",
    "Music",
    "Television / Media",
    "Movie Scene",
    "Public Transport",
    "Airport",
    "Sports Event",
    "Restaurant / Cafe",
    "Mall",
    "Home",
    "Footsteps",
    "Silence / Unknown"
]

class CASREEngine:
    def __init__(self, threshold=0.30):
        self.threshold = threshold

    def analyze(self, transcript: str, scene_probs: list, t_conf: float):
        """
        Next-Generation CASRE (Phase 5, 6, 9, 10)
        Performs multi-factor reasoning on the timeline events.
        """
        # 1. Multi-Label Extraction
        active_scenes = []
        for i, prob in enumerate(scene_probs):
            if prob > self.threshold:
                active_scenes.append((SCENE_LABELS[i], prob))
        
        # Sort by confidence
        active_scenes.sort(key=lambda x: x[1], reverse=True)
        
        if not active_scenes:
            active_scenes.append(("Silence / Unknown", 1.0))

        scene_names = [s[0] for s in active_scenes]
        
        # 2. Risk & Urgency Scoring
        risk_score = 1
        urgency = "Low"
        
        if "Siren & Alarm" in scene_names or "Construction" in scene_names:
            risk_score = 7
            urgency = "High"
            
        # 3. Media Detection (Phase 6)
        is_media = False
        media_labels = {"Music", "Television / Media", "Movie Scene"}
        if any(label in scene_names for label in media_labels):
            is_media = True
            # Media usually downgrades real-world risk unless it's a real broadcast
            risk_score = max(1, risk_score - 5)
            urgency = "Low (Media Playback)"
            
        # 4. Speech-Environment Contradiction Detection (Phase 5)
        has_speech = len(transcript.strip()) > 0
        contradiction = False
        
        if has_speech and urgency == "High":
            # If someone is talking casually while sirens are blaring
            # We look for distress keywords in the transcript
            distress_words = {"help", "emergency", "fire", "police", "hurt", "stop"}
            transcript_words = set(transcript.lower().split())
            if not transcript_words.intersection(distress_words):
                contradiction = True
                urgency = "Moderate (Contradiction: Calm speech near high-risk event)"
        
        # 5. Build Comprehensive Output
        response = "CASRE Advanced Analysis:\n"
        response += "-----------------------\n"
        # Only show top 3 scenes for clarity to the common user
        top_3_scenes = active_scenes[:3]
        response += f"Multi-Label Scene: {', '.join([f'{s[0]} ({s[1]*100:.0f}%)' for s in top_3_scenes])}\n"
        response += f"Media Detection:   {'Positive (Recorded Media)' if is_media else 'Negative (Live Environment)'}\n"
        response += f"Risk Score:        {risk_score}/10\n"
        response += f"Urgency Level:     {urgency}\n"
        
        if contradiction:
            response += "\nContradiction Detected: Speech sentiment does not match environmental risk. Reviewing context...\n"
            
        if has_speech:
            response += f"Speech Content:    '{transcript}' (Confidence: {t_conf:.2f})\n"
        else:
            response += "Speech Content:    None detected.\n"
            
        return response, active_scenes, risk_score, is_media
