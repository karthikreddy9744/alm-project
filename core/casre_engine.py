import numpy as np
import re

SCENE_LABELS = [
    "Traffic & Vehicles", "Siren & Alarm", "Crowd & Hubbub", "Weather & Nature",
    "Water", "Indoor / Domestic", "Office", "Construction", "Wildlife / Animals",
    "Music", "Television / Media", "Movie Scene", "Public Transport", "Airport",
    "Sports Event", "Restaurant / Cafe", "Mall", "Home", "Footsteps", "Silence / Unknown"
]

class CASREEngine:
    def __init__(self, threshold=0.30):
        self.threshold = threshold
        
        self.env_map = {
            "Domestic": {"Home", "Indoor / Domestic"},
            "Public/Social": {"Restaurant / Cafe", "Mall", "Sports Event", "Crowd & Hubbub"},
            "Professional": {"Office", "Construction"},
            "Nature/Outdoors": {"Weather & Nature", "Wildlife / Animals", "Water"},
            "Transit": {"Public Transport", "Airport", "Traffic & Vehicles"}
        }

        self.keyword_categories = {
            "War/Crisis": {"target", "targeted", "tower", "bomb", "missile", "shelter", "explosion", "attack", "strike", "military", "troops", "blast", "go down", "rocket", "gunfire", "shooting", "evacuate"},
            "Emergency": {"help", "emergency", "fire", "police", "hurt", "stop", "danger", "911", "crash", "blood", "safe", "ambulance"},
            "Weather/Disaster": {"wind", "winds", "rain", "storm", "cyclone", "hurricane", "flood", "flooding", "water", "waters", "inundated", "damage", "destroyed", "earthquake", "tsunami", "weather", "embankment", "nature"},
            "Action/Combat": {"fight", "kill", "die", "punch", "sword", "shield", "smash"},
            "Sci-Fi/Fantasy/Drama": {"loki", "asgard", "magic", "alien", "monster", "spaceship", "galaxy", "superhero", "avenger", "thanos", "glorious purpose", "steve rogers", "purpose"},
            "Professional": {"meeting", "deadline", "work", "boss", "project", "presentation", "schedule", "report"},
            "Casual": {"yeah", "cool", "okay", "hey", "like", "literally", "stuff", "awesome", "dude", "uh", "um", "bro"},
            "Frustration": {"damn", "shit", "fuck", "annoying", "hate", "ugh", "stupid", "sorry"},
            "Broadcast": {"today", "news", "reporting", "live", "tonight", "update", "welcome", "breaking", "host", "studio", "broadcast", "correspondent", "camera", "debate"},
            "Music/Song": {"chorus", "verse", "rhythm", "beat", "melody", "singing", "lyrics", "song", "track"}
        }
        
        self.phrase_categories = {
            "Broadcast": ["reporting live", "as you can see", "back to you", "in the studio", "breaking news"],
            "War/Crisis": ["take cover", "get down", "direct hit", "under attack"]
        }
        
        self._build_omni_matrix()

    def _build_omni_matrix(self):
        # PRIORITY ORDERING: Critical -> Media -> Specific Environments -> Generic Environments
        self.omni_matrix = [
            
            # --- PRIORITY 1: UNIVERSAL EXTREMES (Tone-Driven Overrides regardless of scene) ---
            {"tones": ["Weather/Disaster", "Broadcast"], "scenario": "Live Weather Emergency Broadcast", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Transcript explicitly discusses severe weather alongside formal broadcast terminology, indicating a live reporting scenario of a disaster.", "risk": 9, "urgency": "CRITICAL", "media": True},
            {"tones": ["War/Crisis", "Broadcast"], "scenario": "War Zone Live Reporting", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Transcript contains war/crisis terminology delivered in a formal broadcast style, pointing to an active warzone correspondent report.", "risk": 10, "urgency": "EXTREME", "media": True},
            {"tones": ["Weather/Disaster"], "scenario": "Natural Disaster / Severe Weather Event", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Transcript explicitly discusses severe weather, flooding, winds, or structural damage. This completely changes the context to a disaster event.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"tones": ["War/Crisis"], "scenario": "Crisis / Conflict Zone (War/Attack)", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Transcript explicitly contains extreme crisis/war terminology. This strongly overrides baseline heuristics.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"tones": ["Emergency"], "scenario": "Emergency / Distress Situation", "reasoning": "SEMANTIC OVERRIDE: Transcript contains explicit calls for help, emergency services, or distress markers, overriding standard environmental noise.", "risk": 8, "urgency": "High", "media": False},
            {"scenes": ["Siren & Alarm"], "tones": ["War/Crisis"], "scenario": "Air Raid Siren / Nuclear Alarm", "reasoning": "Siren acoustics combined with war terminology explicitly points to an air raid or nuclear alarm.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Siren & Alarm"], "tones": ["Silence / Unknown", "Indoor / Domestic"], "scenario": "Building Security Alarm", "reasoning": "Siren acoustics in a quiet/indoor environment imply a localized security or fire alarm.", "risk": 5, "urgency": "Moderate", "media": False},

            # --- PRIORITY 2: MEDIA, FICTION & ENTERTAINMENT ---
            {"tones": ["Sci-Fi/Fantasy/Drama"], "scenario": "Sci-Fi / Fantasy Movie Scene", "reasoning": "Transcript contains explicit fictional, superhero, or fantasy terminology, confirming an entertainment/movie context.", "risk": 1, "urgency": "Low", "media": True},
            {"tones": ["Action/Combat", "Scream/Yell", "Casual"], "scenario": "Action Sequence / Movie Scene", "reasoning": "Contradictory mix of casual dialogue, action keywords, and extreme screaming is a hallmark of acted media or action movies.", "risk": 2, "urgency": "Low", "media": True},
            {"scenes": ["Movie Scene"], "tones": ["War/Crisis", "Scream/Yell", "Action/Combat"], "scenario": "Action / War Film", "reasoning": "Screams and war terminology detected inside a confirmed movie acoustic environment.", "risk": 1, "urgency": "Low", "media": True},
            {"scenes": ["Television / Media", "Broadcast"], "tones": ["War/Crisis"], "scenario": "Live News Report (Warzone)", "reasoning": "War terminology mapped to broadcast/media acoustics implies a televised warzone report.", "risk": 1, "urgency": "Low", "media": True},
            {"scenes": ["Television / Media", "Broadcast"], "tones": ["Weather/Disaster"], "scenario": "Live News Report (Weather)", "reasoning": "Disaster terminology mapped to broadcast/media acoustics implies a televised weather emergency report.", "risk": 1, "urgency": "Low", "media": True},
            {"scenes": ["Sports Event"], "tones": ["Television / Media", "Broadcast"], "scenario": "Televised Sports Game", "reasoning": "Sports acoustics combined with broadcast/media signals indicate a televised match.", "risk": 1, "urgency": "Low", "media": True},
            {"scenes": ["Sports Event", "Crowd & Hubbub"], "tones": ["Frustration", "Casual"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Live Stadium / Fan Reaction", "reasoning": "Crowd noise mixed with passionate casual/frustration tones without formal broadcast markers points to an in-person stadium crowd.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Television / Media"], "tones": ["Music", "Casual"], "scenario": "Music Video / MTV", "reasoning": "Music and casual tones detected on media/TV imply a music video or pop culture broadcast.", "risk": 1, "urgency": "Low", "media": True},
            {"scenes": ["Television / Media"], "tones": ["Wildlife / Animals", "Broadcast"], "scenario": "Nature Documentary (Media)", "reasoning": "Wildlife acoustics combined with media/broadcast markers explicitly points to a nature documentary.", "risk": 1, "urgency": "Low", "media": True},

            # --- PRIORITY 3: SPECIFIC REAL-WORLD ENVIRONMENTS ---
            # 1. WATER & MARINE
            {"scenes": ["Water"], "tones": ["Emergency", "Scream/Yell"], "scenario": "Marine Rescue / Drowning", "reasoning": "Distress markers or screaming in a marine environment indicate a severe drowning or water rescue crisis.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Water"], "tones": ["Weather/Disaster", "Siren & Alarm"], "scenario": "Marine Storm / Tsunami", "reasoning": "Severe weather keywords combined with aquatic acoustics indicate a dangerous marine storm or tsunami.", "risk": 10, "urgency": "CRITICAL", "media": False},
            {"scenes": ["Water"], "tones": ["Music", "Casual"], "exclude_tones": ["Scream/Yell", "Emergency"], "scenario": "Pool Party / Beach Hangout", "reasoning": "Music and casual tones near water suggest a recreational pool or beach gathering.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Water"], "tones": ["Silence / Unknown", "Casual"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Quiet Fishing / Sailing", "reasoning": "Calm water acoustics with no extreme tones imply peaceful fishing or sailing.", "risk": 1, "urgency": "Low", "media": False},
            
            # 2. TRANSIT, VEHICLES & ROADWAYS
            {"scenes": ["Traffic & Vehicles"], "tones": ["Emergency", "Siren & Alarm"], "scenario": "Car Crash / Major Collision", "reasoning": "Emergency calls in a heavy traffic environment strongly point to a vehicular accident.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"scenes": ["Traffic & Vehicles"], "tones": ["Frustration", "Scream/Yell"], "scenario": "Severe Road Rage", "reasoning": "Screaming and frustration in traffic acoustics are clear markers of a severe road rage incident.", "risk": 6, "urgency": "Moderate", "media": False},
            {"scenes": ["Public Transport", "Airport"], "tones": ["War/Crisis"], "scenario": "Transit Hijacking / Terror Crisis", "reasoning": "War and crisis terminology in a transit hub implies a severe hijacking or terror event.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Public Transport"], "tones": ["Frustration", "Scream/Yell"], "scenario": "Subway / Transit Brawl", "reasoning": "Frustration and screaming inside public transit strongly indicates a physical altercation or brawl.", "risk": 6, "urgency": "Moderate", "media": False},
            {"scenes": ["Airport", "Public Transport"], "tones": ["Broadcast"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Transit PA Announcement", "reasoning": "Formal broadcast speech in a transit environment is highly characteristic of overhead PA announcements.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Public Transport", "Airport"], "tones": ["Silence / Unknown"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Quiet Train / Flight Ride", "reasoning": "Public transit acoustics with silence indicates a peaceful commute.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Traffic & Vehicles"], "tones": ["Silence / Unknown", "Frustration"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Traffic Jam / Idling", "reasoning": "Vehicular acoustics with no crisis or mild frustration implies idling or a traffic jam.", "risk": 2, "urgency": "Low", "media": False},
            {"scenes": ["Traffic & Vehicles"], "tones": ["Action/Combat", "Siren & Alarm"], "scenario": "Police Chase / Pursuit", "reasoning": "Action/Combat sounds mixed with traffic and sirens indicate a high-speed police pursuit.", "risk": 8, "urgency": "High", "media": False},

            # 3. PUBLIC SPACES & CROWDS
            {"scenes": ["Mall", "Crowd & Hubbub"], "tones": ["War/Crisis", "Scream/Yell"], "scenario": "Mass Shooting / Terror Attack", "reasoning": "Screaming and war/crisis terms in a crowded public space are explicit markers of an active shooter or terror attack.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Music", "Crowd & Hubbub"], "tones": ["Emergency", "Scream/Yell"], "scenario": "Concert Stampede / Panic", "reasoning": "Distress and screaming overlaying music indicates a severe crowd crush or stampede at a live event.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Crowd & Hubbub", "Traffic & Vehicles"], "tones": ["Scream/Yell", "Frustration"], "scenario": "Violent Riot / Civil Unrest", "reasoning": "Crowds, traffic, frustration, and screaming characterize a violent street riot or civil unrest.", "risk": 8, "urgency": "High", "media": False},
            {"scenes": ["Crowd & Hubbub", "Traffic & Vehicles"], "tones": ["Broadcast"], "exclude_tones": ["War/Crisis"], "scenario": "Political Rally / Organized Protest", "reasoning": "Formal/Broadcast speech directed at a crowd in a public space indicates a political rally or organized protest.", "risk": 3, "urgency": "Low", "media": False},
            {"scenes": ["Crowd & Hubbub", "Music"], "tones": ["Casual"], "exclude_tones": ["Scream/Yell", "War/Crisis", "Emergency"], "scenario": "Flash Mob / Public Performance", "reasoning": "Music and casual crowd interaction without distress markers suggests a flash mob or public performance.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Restaurant / Cafe"], "tones": ["Frustration", "Scream/Yell"], "scenario": "Bar Brawl / Public Altercation", "reasoning": "Screaming and frustration in a dining or cafe environment indicates a physical altercation or bar fight.", "risk": 7, "urgency": "High", "media": False},
            {"scenes": ["Restaurant / Cafe", "Crowd & Hubbub"], "tones": ["Broadcast"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Stand-up Comedy / Live Show", "reasoning": "Broadcast-style speech directed at a relaxed crowd in a cafe/restaurant setting suggests a comedy or live show.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Restaurant / Cafe"], "tones": ["Music", "Casual"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Dining / Background Music", "reasoning": "Casual tones and background music in a cafe implies a standard dining environment.", "risk": 1, "urgency": "Low", "media": False},

            # 4. OFFICE & PROFESSIONAL
            {"scenes": ["Office"], "tones": ["War/Crisis", "Scream/Yell"], "scenario": "Office Active Shooter", "reasoning": "Screaming and crisis keywords in an office environment point to an active shooter or workplace attack.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Office"], "tones": ["Emergency", "Siren & Alarm"], "scenario": "Workplace Fire / Evacuation", "reasoning": "Sirens and emergency terminology in an office dictate a fire alarm or emergency evacuation.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"scenes": ["Office"], "tones": ["Professional", "Broadcast"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Corporate Presentation", "reasoning": "Formal/Broadcast speech mixed with professional keywords in an office indicates a structured corporate presentation.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Office"], "tones": ["Professional", "Frustration"], "scenario": "Workplace Dispute / Firing", "reasoning": "Professional terminology mixed with frustration indicates a severe workplace dispute or termination.", "risk": 4, "urgency": "Moderate", "media": False},
            {"scenes": ["Office"], "tones": ["Footsteps", "Silence / Unknown"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Solo Office Working (After Hours)", "reasoning": "Silence and footsteps in an office indicates solo working after hours.", "risk": 1, "urgency": "Low", "media": False},

            # 5. DOMESTIC & HOME
            {"scenes": ["Home", "Indoor / Domestic"], "tones": ["War/Crisis", "Scream/Yell"], "scenario": "Home Invasion / Armed Burglary", "reasoning": "Screaming and crisis keywords inside a home environment strongly indicate an armed home invasion.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Home", "Indoor / Domestic"], "tones": ["Frustration", "Scream/Yell"], "exclude_tones": ["War/Crisis", "Action/Combat", "Sci-Fi/Fantasy/Drama"], "scenario": "Domestic Abuse / Violent Dispute", "reasoning": "Screaming and frustration in a domestic environment are markers of severe domestic violence or a heated dispute.", "risk": 8, "urgency": "High", "media": False},
            {"scenes": ["Home", "Indoor / Domestic"], "tones": ["Emergency", "Siren & Alarm"], "scenario": "Domestic Medical Emergency", "reasoning": "Emergency keywords and sirens in a home environment point to an acute medical crisis or fire at a residence.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"scenes": ["Home", "Indoor / Domestic"], "tones": ["Casual", "Frustration"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Online Gaming / eSports LAN", "reasoning": "Casual tones mixed with sporadic frustration in a home setting strongly characterize online gaming or an eSports session.", "risk": 1, "urgency": "Low", "media": False},
            {"scenes": ["Home", "Indoor / Domestic"], "tones": ["Crowd & Hubbub", "Casual"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Dinner Party / Family Gathering", "reasoning": "A crowd of casual speakers inside a home indicates a dinner party or family gathering.", "risk": 1, "urgency": "Low", "media": False},

            # 6. NATURE & WILDLIFE
            {"scenes": ["Wildlife / Animals"], "tones": ["Emergency", "Scream/Yell"], "scenario": "Animal Attack / Predator Threat", "reasoning": "Screaming and emergency tones alongside prominent wildlife acoustics indicate a severe animal attack.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Wildlife / Animals"], "tones": ["War/Crisis", "Action/Combat"], "scenario": "Hunting / Gunshots Outdoors", "reasoning": "War/Combat keywords (like 'shoot') alongside wildlife acoustics imply a hunting environment.", "risk": 6, "urgency": "Moderate", "media": False},
            {"scenes": ["Weather & Nature"], "tones": ["Emergency"], "scenario": "Lost / Hiking Distress", "reasoning": "Emergency calls isolated in a natural environment imply a lost hiker or outdoor distress situation.", "risk": 8, "urgency": "High", "media": False},
            {"scenes": ["Weather & Nature"], "tones": ["Silence / Unknown", "Casual"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Nature Relaxation / Camping", "reasoning": "Peaceful nature acoustics with casual or no speech indicate camping or outdoor relaxation.", "risk": 1, "urgency": "Low", "media": False},

            # 7. INDUSTRIAL & CONSTRUCTION
            {"scenes": ["Construction"], "tones": ["Emergency", "Siren & Alarm"], "scenario": "Industrial Accident / Machinery Failure", "reasoning": "Sirens and emergency calls in a construction zone strongly point to a severe industrial or machinery accident.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"scenes": ["Construction"], "tones": ["War/Crisis"], "scenario": "Active Demolition", "reasoning": "Crisis/Blast keywords in a construction environment likely indicate a planned active demolition.", "risk": 5, "urgency": "Moderate", "media": False},
            {"scenes": ["Construction"], "tones": ["Professional"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Standard Construction / Tool Use", "reasoning": "Professional terminology mixed with construction acoustics indicates normal tool operation.", "risk": 2, "urgency": "Low", "media": False},
        ]

    def _infer_environment(self, scene_names):
        env_scores = {k: 0 for k in self.env_map}
        for scene in scene_names:
            for env, labels in self.env_map.items():
                if scene in labels:
                    env_scores[env] += 1
        
        best_env = max(env_scores, key=env_scores.get)
        if env_scores[best_env] > 0:
            return best_env
        return "Unknown Context"

    def _linguistic_analysis(self, transcript):
        if not transcript.strip():
            return "No Speech", "None", []
            
        words = re.findall(r'\b\w+\b', transcript.lower())
        if not words:
            return "No Speech", "None", []
            
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(w) for w in words) / word_count
        
        formal_score = 0
        casual_score = 0
        
        if avg_word_length > 4.5: formal_score += 1
        if unique_words / word_count > 0.8 and word_count > 10: formal_score += 1
        if unique_words / word_count < 0.5 and word_count > 15: casual_score += 2
        
        word_set = set(words)
        casual_score += len(self.keyword_categories["Casual"].intersection(word_set))
        formal_score += len(self.keyword_categories["Broadcast"].intersection(word_set))
        
        transcript_lower = transcript.lower()
        for phrase in self.phrase_categories["Broadcast"]:
            if phrase in transcript_lower:
                formal_score += 5
                
        if formal_score > casual_score:
            style = "Formal/Structured"
        elif casual_score > formal_score:
            style = "Casual/Conversational"
        else:
            style = "Neutral"
            
        if unique_words / word_count < 0.4 and word_count > 15:
            style = "Highly Repetitive (Potential Lyrics/Chant)"
            
        detected_tones = []
        for tone, keywords in self.keyword_categories.items():
            if word_set.intersection(keywords):
                detected_tones.append(tone)
                
        if re.search(r'\b(ah{2,}|a{3,}|no{3,}|oh{3,}|argh)\b', transcript_lower):
            detected_tones.append("Scream/Yell")
                
        for tone, phrases in self.phrase_categories.items():
            if tone not in detected_tones:
                for phrase in phrases:
                    if phrase in transcript_lower:
                        detected_tones.append(tone)
                        break
                
        tone_str = ", ".join(detected_tones) if detected_tones else "Neutral"
        return style, tone_str, detected_tones

    def analyze(self, transcript: str, scene_probs: list, t_conf: float, timeline: list = None):
        transcript_style, tone_str, detected_tones = self._linguistic_analysis(transcript)
        has_speech = len(transcript.strip()) > 0
        
        active_scenes = []
        music_prob = scene_probs[SCENE_LABELS.index("Music")]
        media_prob = scene_probs[SCENE_LABELS.index("Television / Media")]
        
        for i, prob in enumerate(scene_probs):
            if prob > self.threshold:
                active_scenes.append((SCENE_LABELS[i], prob))
        
        if has_speech and ("Highly Repetitive" in transcript_style or "Music/Song" in detected_tones):
            if music_prob > 0.10 and "Music" not in [s[0] for s in active_scenes]:
                active_scenes.append(("Music", music_prob + 0.30))
                
        if has_speech and "Broadcast" in detected_tones:
            if media_prob > 0.10 and "Television / Media" not in [s[0] for s in active_scenes]:
                active_scenes.append(("Television / Media", media_prob + 0.30))
        
        active_scenes.sort(key=lambda x: x[1], reverse=True)
        if not active_scenes:
            active_scenes.append(("Silence / Unknown", 1.0))

        scene_names = [s[0] for s in active_scenes]
        env = self._infer_environment(scene_names)
        
        media_labels = {"Music", "Television / Media", "Movie Scene"}
        detected_media = [label for label in scene_names if label in media_labels]
        is_media = len(detected_media) > 0
        
        # OMNI-MATRIX EVALUATION (WITH EXCLUDE_TONES LOGIC)
        matched_scenario = None
        for rule in self.omni_matrix:
            scene_match = True
            if "scenes" in rule:
                if not any(s in scene_names for s in rule["scenes"]):
                    scene_match = False
                    
            tone_match = True
            if "tones" in rule:
                if not all(t in detected_tones for t in rule["tones"]):
                    tone_match = False
                    
            exclusion_match = True
            if "exclude_tones" in rule:
                if any(t in detected_tones for t in rule["exclude_tones"]):
                    exclusion_match = False
                    
            if scene_match and tone_match and exclusion_match:
                matched_scenario = rule
                break
                
        if matched_scenario:
            scenario = matched_scenario["scenario"]
            reasoning = [matched_scenario["reasoning"]]
            risk_score = matched_scenario["risk"]
            urgency = matched_scenario["urgency"]
            if matched_scenario["media"]:
                is_media = True
        else:
            scenario = "Ambiguous Environment"
            reasoning = []
            risk_score = 1
            urgency = "Low"
            
            is_speech_reliable = has_speech and (t_conf >= 0.50 or len(detected_tones) > 0)
            top_scene_name = active_scenes[0][0] if active_scenes else None
            top_scene_prob = active_scenes[0][1] if active_scenes else 0.0
            
            if is_speech_reliable:
                if "Music" in scene_names:
                    scenario = "Media Playback (Music & Voice)"
                    reasoning.append("Music detected alongside speech, indicating a podcast or media clip.")
                elif is_media:
                    scenario = "Television Show / Movie Clip"
                    reasoning.append("Media sounds detected alongside speech, pointing to a standard TV show.")
                elif "Crowd & Hubbub" in scene_names:
                    scenario = "Public Social Interaction"
                    reasoning.append("Crowd present with speech, indicating a public space.")
                else:
                    scenario = f"Human Activity ({env})"
                    reasoning.append("Speech is the primary activity detected in the environment.")
            else:
                if top_scene_prob > 0.80 and top_scene_name not in ["Silence / Unknown", "Indoor / Domestic"]:
                    scenario = f"Acoustic Event: {top_scene_name}"
                    reasoning.append(f"A strong {top_scene_name.lower()} signal is the primary acoustic event.")
                else:
                    scenario = f"Pure Environmental Noise ({env})"
                    reasoning.append(f"Entirely driven by ambient environmental sounds from a {env.lower()} setting.")

        # NATE (Physics) Overrides
        nate_insights = []
        if timeline and len(timeline) > 1:
            rms_values = [t.get("rms", 0) for t in timeline]
            pitch_values = [t.get("pitch", 0) for t in timeline]
            
            avg_rms = np.mean(rms_values)
            max_rms = np.max(rms_values)
            avg_pitch = np.mean(pitch_values)
            max_pitch = np.max(pitch_values)
            
            if len(rms_values) >= 3:
                first, mid, last = rms_values[0], max(rms_values), rms_values[-1]
                if mid > first * 1.5 and mid > last * 1.5:
                    nate_insights.append("Dynamic Proximity: The primary sound source is passing by (Doppler-like fade in/out).")
                elif min(rms_values) > 0.1:
                    nate_insights.append("Static Proximity: The sound source is stationary and in immediate proximity.")
                    
            if max_pitch > avg_pitch * 2 and max_pitch > 3000:
                if "Siren & Alarm" in scene_names or "Emergency" in detected_tones:
                    nate_insights.append("Predictive Surprise: Sudden high-pitch acoustic spike indicates acute alarm or distress.")
                    risk_score = min(10, risk_score + 2)
                    urgency = "Elevated (High-Pitch Startle/Fear Detected)"
                    
            if scenario == "Crisis / Conflict Zone (War/Attack)" and avg_rms < 0.05:
                scenario = "Covert Military / Stealth Operation"
                nate_insights.append("PHYSICS OVERRIDE: Despite war terminology, the environment is exceptionally quiet (Low RMS), indicating stealth or covert activity rather than an active firefight.")
                
            if scenario in ["Emergency / Distress Situation", "Home Invasion / Armed Burglary"] and avg_rms < 0.05:
                scenario = "Covert Emergency / Hiding"
                nate_insights.append("PHYSICS OVERRIDE: Emergency terminology delivered at very low volume indicates the speaker is hiding or in a covert distress situation.")

            if nate_insights:
                reasoning.append(" | ".join(nate_insights))
        
        contextual_reasoning = " ".join(reasoning)
        
        response = "CASRE Omni-Matrix Analysis:\n"
        response += "=======================\n"
        response += f"Predicted Scenario: {scenario}\n"
        response += f"Contextual Reasoning: {contextual_reasoning}\n\n"
        
        response += "Technical Breakdown:\n"
        response += "-----------------------\n"
        top_3_scenes = active_scenes[:3]
        response += f"Multi-Label Scene: {', '.join([f'{s[0]} ({min(s[1], 1.0)*100:.0f}%)' for s in top_3_scenes])}\n"
        
        media_str = f"Positive ({', '.join(detected_media)})" if is_media else "Negative (Live Environment)"
        response += f"Media Detection:   {media_str}\n"
        response += f"Risk / Urgency:    {risk_score}/10 | {urgency}\n"
        
        if has_speech:
            response += f"Extracted Topics:  {tone_str}\n"
            response += f"Speech Style:      {transcript_style}\n"
            response += f"Transcript:        '{transcript}' (Conf: {t_conf:.2f})\n"
        else:
            response += "Speech Content:    None detected.\n"
            
        return response, active_scenes, risk_score, is_media
