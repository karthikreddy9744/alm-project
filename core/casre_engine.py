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

    def _evaluate_evidence_strength(self, has_speech, detected_tones, active_scenes, t_conf, timeline, is_media):
        speech_strength = "Absent"
        if has_speech:
            if t_conf > 0.7 and detected_tones and detected_tones != ["Neutral"]:
                speech_strength = "Strong"
            elif t_conf > 0.4 or detected_tones:
                speech_strength = "Moderate"
            else:
                speech_strength = "Weak"
                
        env_strength = "Absent"
        if active_scenes and active_scenes[0][0] != "Silence / Unknown":
            if active_scenes[0][1] > 0.75:
                env_strength = "Strong"
            elif active_scenes[0][1] > 0.4:
                env_strength = "Moderate"
            else:
                env_strength = "Weak"
                
        temporal_strength = "Absent"
        if timeline and len(timeline) > 1:
            rms_values = [t.get("rms", 0) for t in timeline]
            if max(rms_values) > min(rms_values) * 2:
                temporal_strength = "Strong"
            elif max(rms_values) > min(rms_values) * 1.2:
                temporal_strength = "Moderate"
            else:
                temporal_strength = "Weak"
                
        media_strength = "Absent"
        if is_media:
            media_strength = "Strong"
            
        strength_scores = {"Strong": 3, "Moderate": 2, "Weak": 1, "Absent": 0}
        total = strength_scores[speech_strength] + strength_scores[env_strength] + strength_scores[temporal_strength]
        if is_media: total += 3
        
        overall = "Weak"
        if total >= 6: overall = "Strong"
        elif total >= 3: overall = "Moderate"
        
        return {
            "Speech Evidence": speech_strength,
            "Environmental Evidence": env_strength,
            "Temporal Evidence": temporal_strength,
            "Media Evidence": media_strength,
            "Overall Evidence Strength": overall
        }

    def _generate_specific_observations(self, transcript, active_scenes, timeline_active, is_media, detected_tones):
        obs = []
        if transcript.strip():
            phrase_snippet = " ".join(transcript.split()[:5])
            if "Help" in transcript or "help" in transcript:
                obs.append("Transcript contains repeated requests for assistance.")
                obs.append("The phrase 'help' is present.")
            else:
                obs.append(f"Speech detected, beginning with '{phrase_snippet}...'")
        else:
            obs.append("No speech markers detected in the transcript.")
            
        if active_scenes and active_scenes[0][0] != "Silence / Unknown":
            obs.append(f"Acoustic classification indicates {active_scenes[0][0]} at {active_scenes[0][1]*100:.0f}% confidence.")
        else:
            obs.append("Environmental acoustics fall below detection thresholds.")
            
        if timeline_active:
            obs.append("Acoustic energy exhibits distinct transient variations over the observation window.")
            
        if is_media:
            obs.append("Features highly characteristic of broadcast, television, or recorded music are present.")
            
        return obs

    def _evaluate_relationship(self, detected_tones, scene_names, is_media, transcript_style):
        if is_media:
            return "Independent", "Media playback is dominating. Speech and acoustic features are artifacts of the broadcast rather than live physical events."
        
        if not detected_tones or detected_tones == ["Neutral"]:
            if "Siren & Alarm" in scene_names or "Scream/Yell" in scene_names:
                return "Independent", "The acoustic event appears unrelated to the speaker's activity. Speech remains casual or neutral while the environment is active."
            return "Ambiguous", "No significant relationship between speech and environment is evident."
            
        crisis_tones = {"War/Crisis", "Emergency", "Weather/Disaster", "Action/Combat", "Scream/Yell"}
        crisis_scenes = {"Siren & Alarm"}
        
        has_crisis_tone = bool(set(detected_tones).intersection(crisis_tones))
        has_crisis_scene = bool(set(scene_names).intersection(crisis_scenes))
        
        if has_crisis_tone and has_crisis_scene:
            if "Siren & Alarm" in scene_names and scene_names.index("Siren & Alarm") == 0:
                return "Strong Reinforcement", "Both modalities strongly support the interpretation of an acute event, with environmental cues dominating."
            return "Moderate Reinforcement", "Speech and environmental modalities support an overlapping semantic interpretation."
            
        if has_crisis_tone and not has_crisis_scene and "Silence / Unknown" in scene_names:
             return "Contradictory", "Speech indicates a crisis or extreme event, but the acoustic environment is completely silent or calm."
             
        if not has_crisis_tone and has_crisis_scene:
            return "Contradictory", "The environment contains crisis markers, but the speech directly conflicts, remaining casual or calm."
            
        if transcript_style == "Highly Repetitive (Potential Lyrics/Chant)" and "Music" not in scene_names:
            return "Weak Support", "Speech patterns suggest a performance, but no acoustic music is detected to confirm."
            
        if has_crisis_tone:
            return "Weak Support", "Speech indicates concern, but the environment lacks strong acoustic confirmation."
            
        return "Moderate Reinforcement", "Speech and environment generally align without major contradictions."

    def _interpret_environment(self, scene_names, active_scenes, env):
        if not scene_names or scene_names[0] == "Silence / Unknown":
            return "Acoustic characteristics are indistinct, providing no clear environmental context."
            
        primary = active_scenes[0]
        interpretation = f"The environment is dominated by {primary[0]} acoustic characteristics."
        
        if len(active_scenes) > 1:
            secondary = active_scenes[1]
            if primary[1] - secondary[1] < 0.2:
                interpretation += f" However, {secondary[0]} signals are also strongly present, suggesting a mixed or complex acoustic space."
            else:
                interpretation += f" Secondary signatures of {secondary[0]} are present but non-dominant."
                
        if "Siren & Alarm" in scene_names:
            interpretation += " Emergency-related signals are highly relevant and cut through the baseline environmental noise."
            
        return interpretation

    def _generate_alternative(self, scenario, is_media, env):
        if is_media:
            return "Movie scene, podcast recording, news broadcast, or recorded media playback."
        if "Emergency" in scenario or "Crisis" in scenario:
            return "Roleplay, acting rehearsal, training exercise, emergency simulation, or an intense gaming session."
        if "Music" in scenario or "Performance" in scenario:
            return "Karaoke, live public performance, or loud song playback."
        if env == "Domestic" or env == "Home":
            return "Practicing lines, dramatic media playing, or loud domestic disagreement."
        if env == "Professional" or env == "Office":
            return "Team-building exercise, simulation, or standard industrial operations."
        if env == "Public/Social":
            return "Flash mob, prank, street performance, or loud public gathering."
        return "Testing scenario, practice drill, or misinterpreted acoustic artifact."

    def _generate_uncertainty(self, t_conf, is_media, active_scenes, has_speech):
        unc = []
        unc.append("Visual evidence unavailable.")
        unc.append("Speaker identity and intent cannot be verified.")
        if is_media:
            unc.append("Audio source is likely indirect (recorded media).")
        else:
            unc.append("Media playback cannot be entirely excluded without secondary sensors.")
            
        if t_conf < 0.5 and has_speech:
            unc.append("Transcript confidence is limited, which constrains semantic analysis.")
            
        if not active_scenes or active_scenes[0][1] < 0.5:
            unc.append("Environmental acoustic signals are ambiguous.")
            
        return unc

    def _calculate_confidence(self, strength, rel_type):
        drivers = []
        limitations = []
        score = "Moderate"
        
        # Drivers based strictly on rules
        if strength["Speech Evidence"] == "Strong": drivers.append("Strong semantic clarity in speech")
        if strength["Environmental Evidence"] == "Strong": drivers.append("Stable primary scene classification")
        if strength["Temporal Evidence"] == "Strong": drivers.append("Consistent temporal variance")
        
        if "Reinforcement" in rel_type:
            drivers.append("Speech and environment cross-modally agree")
            
        # Limitations
        if strength["Speech Evidence"] == "Weak": limitations.append("Poor acoustic speech clarity")
        if strength["Environmental Evidence"] == "Weak": limitations.append("Weak acoustic scene probabilities")
        if strength["Overall Evidence Strength"] == "Weak": limitations.append("Overall lack of distinct evidence markers")
        
        if rel_type == "Contradictory":
            limitations.append("Speech and environmental modalities conflict")
            score = "Low"
            
        # Confidence calculation
        if strength["Overall Evidence Strength"] == "Strong" and rel_type not in ["Contradictory", "Independent"]:
            score = "High"
        elif strength["Overall Evidence Strength"] == "Weak":
            score = "Low"
            
        if not limitations:
            limitations.append("Source verification remains unconfirmed")
            
        if not drivers:
            drivers.append("None")
            
        return score, drivers, limitations

    def _calculate_risk(self, scenario, has_crisis_tone, has_crisis_scene, rel_type, is_media):
        if is_media:
            return "Low"
            
        if "Emergency" in scenario or "Crisis" in scenario or "Attack" in scenario:
            if has_crisis_tone and has_crisis_scene and rel_type in ["Strong Reinforcement", "Moderate Reinforcement"]:
                return "Very High"
            return "High"
            
        if has_crisis_tone or has_crisis_scene:
            return "Moderate"
            
        return "Low"
        
    def _generate_analyst_conclusion(self, rel_type, strength, scenario, is_media):
        conclusion = ""
        if is_media:
            conclusion = "The evidence strongly points to media playback rather than a live event. The acoustic profile aligns with broadcast or recorded content, mitigating potential physical risk."
            return conclusion
            
        if "Reinforcement" in rel_type and strength["Overall Evidence Strength"] in ["Strong", "Moderate"]:
            conclusion = f"Both speech semantics and environmental acoustics point toward a {scenario.lower()}. The consistent pattern across multiple modalities strengthens the assessment."
        elif rel_type == "Contradictory":
            conclusion = f"There is a stark contradiction between the detected environment and the speech. While one modality suggests {scenario.lower()}, the other does not align. This limits interpretability."
        elif rel_type == "Independent":
            conclusion = f"Acoustic events are occurring independently of the detected speech. The {scenario.lower()} appears to be a backdrop to unrelated human activity."
        else:
            conclusion = f"The overall pattern suggests a {scenario.lower()}, though the evidence is {strength['Overall Evidence Strength'].lower()}."
            
        conclusion += " Additional verification through secondary sensors or visual feeds is recommended to definitively confirm the situation."
        return conclusion

    def analyze(self, transcript: str, scene_probs: list, t_conf: float, timeline: list = None):
        transcript_style, tone_str, detected_tones = self._linguistic_analysis(transcript)
        has_speech = len(transcript.strip()) > 0
        
        active_scenes = []
        music_prob = scene_probs[SCENE_LABELS.index("Music")] if scene_probs is not None and len(scene_probs) > 0 else 0
        media_prob = scene_probs[SCENE_LABELS.index("Television / Media")] if scene_probs is not None and len(scene_probs) > 0 else 0
        
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
        
        # OMNI-MATRIX EVALUATION
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
            if matched_scenario["media"]:
                is_media = True
        else:
            scenario = "Ambiguous Environment"
            
            is_speech_reliable = has_speech and (t_conf >= 0.50 or len(detected_tones) > 0)
            top_scene_name = active_scenes[0][0] if active_scenes else None
            top_scene_prob = active_scenes[0][1] if active_scenes else 0.0
            
            if is_speech_reliable:
                if "Music" in scene_names:
                    scenario = "Media Playback (Music & Voice)"
                elif is_media:
                    scenario = "Television Show / Movie Clip"
                elif "Crowd & Hubbub" in scene_names:
                    scenario = "Public Social Interaction"
                else:
                    scenario = f"Human Activity ({env})"
            else:
                if top_scene_prob > 0.80 and top_scene_name not in ["Silence / Unknown", "Indoor / Domestic"]:
                    scenario = f"Acoustic Event: {top_scene_name}"
                else:
                    scenario = f"Pure Environmental Noise ({env})"

        # Temporal Awareness
        temporal_evidence = "No distinct temporal variance detected."
        timeline_active = False
        if timeline and len(timeline) > 1:
            rms_values = [t.get("rms", 0) for t in timeline]
            pitch_values = [t.get("pitch", 0) for t in timeline]
            
            avg_rms = np.mean(rms_values)
            max_pitch = np.max(pitch_values)
            avg_pitch = np.mean(pitch_values)
            
            if len(rms_values) >= 3:
                first, mid, last = rms_values[0], max(rms_values), rms_values[-1]
                if mid > first * 1.5 and mid > last * 1.5:
                    temporal_evidence = "Transient event passing by the sensor."
                    timeline_active = True
                elif min(rms_values) > 0.1:
                    temporal_evidence = "Continuous, stable proximity event."
                    
            if max_pitch > avg_pitch * 2 and max_pitch > 3000:
                if "Siren & Alarm" in scene_names or "Emergency" in detected_tones:
                    temporal_evidence += " Sudden high-pitch acoustic spike detected."
                    timeline_active = True
                    
            if scenario == "Crisis / Conflict Zone (War/Attack)" and avg_rms < 0.05:
                scenario = "Covert Military / Stealth Operation"
                temporal_evidence += " Exceptionally low energy indicating stealth."
                
            if scenario in ["Emergency / Distress Situation", "Home Invasion / Armed Burglary"] and avg_rms < 0.05:
                scenario = "Covert Emergency / Hiding"
                temporal_evidence += " Extremely quiet speech dynamics implying concealment."

        # Analytical Framework 18-Stage Execution
        strength = self._evaluate_evidence_strength(has_speech, detected_tones, active_scenes, t_conf, timeline, is_media)
        observations = self._generate_specific_observations(transcript, active_scenes, timeline_active, is_media, detected_tones)
        rel_type, rel_explanation = self._evaluate_relationship(detected_tones, scene_names, is_media, transcript_style)
        env_interpretation = self._interpret_environment(scene_names, active_scenes, env)
        alt_interpretation = self._generate_alternative(scenario, is_media, env)
        uncertainty = self._generate_uncertainty(t_conf, is_media, active_scenes, has_speech)
        conf_score, conf_drivers, conf_limitations = self._calculate_confidence(strength, rel_type)
        
        crisis_tones = {"War/Crisis", "Emergency", "Weather/Disaster", "Action/Combat"}
        has_crisis_tone = bool(set(detected_tones).intersection(crisis_tones))
        has_crisis_scene = bool(set(scene_names).intersection({"Siren & Alarm"}))
        calculated_risk = self._calculate_risk(scenario, has_crisis_tone, has_crisis_scene, rel_type, is_media)
        
        analyst_conclusion = self._generate_analyst_conclusion(rel_type, strength, scenario, is_media)

        rec_response = "Monitoring recommended."
        if calculated_risk == "Very High":
            rec_response = "Escalation recommended."
        elif calculated_risk == "High":
            rec_response = "Immediate attention recommended."
        elif calculated_risk == "Moderate":
            rec_response = "Further verification advised."

        # Assembly
        response = "CASRE SITUATIONAL ASSESSMENT\n\n"
        
        response += "Executive Summary\n"
        response += "⸻\n"
        
        response += "Evidence Strength Overview\n"
        response += f"Speech Evidence: {strength['Speech Evidence']}\n"
        response += f"Environmental Evidence: {strength['Environmental Evidence']}\n"
        response += f"Temporal Evidence: {strength['Temporal Evidence']}\n"
        response += f"Media Evidence: {strength['Media Evidence']}\n"
        response += f"Overall Evidence Strength: {strength['Overall Evidence Strength']}\n"
        response += "⸻\n"

        response += "Observations\n"
        for obs in observations: response += f"* {obs}\n"
        response += "⸻\n"
        
        response += "Evidence Analysis\n"
        response += "⸻\n"
        
        response += "Speech Evidence\n"
        response += f"Transcript: {transcript if transcript.strip() else 'None'}\n"
        response += f"Style: {transcript_style}\n"
        response += f"Topics: {tone_str}\n"
        response += "⸻\n"
        
        response += "Environmental Evidence\n"
        response += f"Primary Environment: {env}\n"
        response += f"Scenes: {', '.join([f'{s[0]} ({s[1]*100:.0f}%)' for s in active_scenes])}\n"
        response += "⸻\n"
        
        response += "Temporal Evidence\n"
        response += f"{temporal_evidence}\n"
        response += "⸻\n"
        
        response += "Media Evidence\n"
        response += f"{'Media signals detected.' if is_media else 'No media signals detected.'}\n"
        response += "⸻\n"
        
        response += "Cross-Modal Relationship\n"
        response += f"{rel_type}\n"
        response += f"{rel_explanation}\n"
        response += "⸻\n"
        
        response += "Situation Assessment\n"
        response += f"Most Likely Interpretation: {scenario}\n"
        response += "⸻\n"
        
        response += "Environmental Interpretation\n"
        response += f"{env_interpretation}\n"
        response += "⸻\n"
        
        response += "Alternative Explanations\n"
        response += f"{alt_interpretation}\n"
        response += "⸻\n"
        
        response += "Uncertainty Analysis\n"
        for unc in uncertainty: response += f"* {unc}\n"
        response += "⸻\n"
        
        response += "Confidence Drivers\n"
        for cd in conf_drivers: response += f"✓ {cd}\n"
        response += "⸻\n"
            
        response += "Confidence Limitations\n"
        for cl in conf_limitations: response += f"✗ {cl}\n"
        response += "⸻\n"
        
        response += "Risk Assessment\n"
        response += f"{calculated_risk}\n"
        response += "⸻\n"
        
        response += "Recommended Response\n"
        response += f"{rec_response}\n"
        response += "⸻\n"
        
        response += "Analyst Conclusion\n"
        response += f"{analyst_conclusion}\n"
        
        # Hack to inject the confidence so the UI parser can still find it (if we want, or UI parser can be fully updated)
        # We will add it strictly for the return tuple since the engine doesn't return conf_score directly.
        return response, active_scenes, 0, is_media
