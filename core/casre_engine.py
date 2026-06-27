import numpy as np
import re

SCENE_LABELS = [
    "Dog", "Poultry", "Pig", "Cow", "Frog",
    "Cat", "Insects", "Sheep", "Crow", "Rain & Thunder",
    "Sea & Water", "Crackling fire", "Crickets", "Chirping birds", "Water drops",
    "Wind", "Toilet flush", "Crying baby", "Coughing & Sneezing", "Clapping",
    "Breathing", "Footsteps", "Laughing", "Personal Care", "Snoring",
    "Door sounds", "Office", "Can opening", "Washing machine", "Vacuum cleaner",
    "Clock", "Glass breaking", "Aviation", "Saws", "Siren",
    "Cars/Traffic", "Train", "Church bells", "Fireworks", "Silence / Unknown"
]

class CASREEngine:
    def __init__(self, threshold=0.30):
        self.threshold = threshold
        
        self.env_map = {
            "Domestic": {"Toilet flush", "Crying baby", "Personal Care", "Snoring", "Door sounds", "Can opening", "Washing machine", "Vacuum cleaner", "Clock", "Glass breaking"},
            "Public/Social": {"Clapping", "Laughing", "Church bells"},
            "Professional": {"Office", "Saws"},
            "Nature/Outdoors": {"Dog", "Poultry", "Pig", "Cow", "Frog", "Cat", "Insects", "Sheep", "Crow", "Rain & Thunder", "Sea & Water", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind"},
            "Transit": {"Aviation", "Cars/Traffic", "Train"}
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
        # Replaced "scenes" with "any_scenes" and "tones" with "any_tones" / "all_tones" for realistic AND/OR logic
        self.omni_matrix = [
            # PRIORITY 1: UNIVERSAL EXTREMES (Tone-Driven Overrides regardless of scene)
            {"all_tones": ["Weather/Disaster", "Broadcast"], "scenario": "Live Weather Emergency Broadcast", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Explicit severe weather terminology delivered via formal broadcast.", "risk": 9, "urgency": "CRITICAL", "media": True},
            {"all_tones": ["War/Crisis", "Broadcast"], "scenario": "War Zone Live Reporting", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Active warzone correspondent reporting.", "risk": 10, "urgency": "EXTREME", "media": True},
            {"any_tones": ["Weather/Disaster"], "scenario": "Natural Disaster / Severe Weather Event", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Severe weather, flooding, winds, or structural damage discussed.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"any_tones": ["War/Crisis"], "scenario": "Crisis / Conflict Zone (War/Attack)", "reasoning": "CRITICAL SEMANTIC OVERRIDE: Extreme crisis/war terminology detected.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_tones": ["Emergency"], "scenario": "Emergency / Distress Situation", "reasoning": "SEMANTIC OVERRIDE: Explicit calls for help or emergency services.", "risk": 8, "urgency": "High", "media": False},
            {"any_scenes": ["Siren"], "any_tones": ["War/Crisis"], "scenario": "Air Raid Siren / Nuclear Alarm", "reasoning": "Siren acoustics combined with war terminology explicitly points to an air raid.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Siren"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Building Security Alarm / False Alarm", "reasoning": "Siren acoustics in an environment without verbal distress implies a generic or false security alarm.", "risk": 5, "urgency": "Moderate", "media": False},

            # PRIORITY 2: MEDIA, FICTION & ENTERTAINMENT
            {"any_tones": ["Sci-Fi/Fantasy/Drama"], "scenario": "Sci-Fi / Fantasy Movie Scene", "reasoning": "Fictional or superhero terminology confirming entertainment media.", "risk": 1, "urgency": "Low", "media": True},
            {"all_tones": ["Action/Combat", "Casual"], "scenario": "Action Sequence / Movie Scene", "reasoning": "Contradictory mix of casual dialogue, action keywords, and extreme screaming implies acted media.", "risk": 2, "urgency": "Low", "media": True},
            {"any_scenes": ["Television / Media"], "any_tones": ["War/Crisis", "Scream/Yell", "Action/Combat"], "scenario": "Action / War Film", "reasoning": "Screams and war terminology detected inside a confirmed media acoustic environment.", "risk": 1, "urgency": "Low", "media": True},
            {"any_scenes": ["Television / Media", "Broadcast"], "any_tones": ["War/Crisis"], "scenario": "Televised Live News Report (Warzone)", "reasoning": "War terminology mapped to broadcast/media acoustics implies a televised report.", "risk": 1, "urgency": "Low", "media": True},
            {"any_scenes": ["Television / Media", "Broadcast"], "any_tones": ["Weather/Disaster"], "scenario": "Televised Live News Report (Weather)", "reasoning": "Disaster terminology mapped to broadcast/media acoustics implies a televised report.", "risk": 1, "urgency": "Low", "media": True},
            {"any_scenes": ["Television / Media"], "any_tones": ["Music", "Casual"], "scenario": "Music Video / MTV", "reasoning": "Music and casual tones detected on media/TV imply a music video.", "risk": 1, "urgency": "Low", "media": True},
            {"any_scenes": ["Television / Media"], "any_tones": ["Broadcast"], "scenario": "Nature Documentary (Media)", "reasoning": "Wildlife acoustics combined with media/broadcast markers implies a documentary.", "risk": 1, "urgency": "Low", "media": True},
            
            # 1. WATER & MARINE
            {"any_scenes": ["Sea & Water", "Water drops"], "any_tones": ["Emergency", "Scream/Yell"], "scenario": "Marine Rescue / Drowning", "reasoning": "Distress markers or screaming in a marine environment indicate a severe drowning crisis.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Sea & Water", "Water drops"], "any_tones": ["Weather/Disaster", "Siren"], "scenario": "Marine Storm / Tsunami", "reasoning": "Severe weather keywords combined with aquatic acoustics indicate a dangerous marine storm.", "risk": 10, "urgency": "CRITICAL", "media": False},
            {"any_scenes": ["Sea & Water", "Water drops"], "any_tones": ["Music", "Casual"], "exclude_tones": ["Scream/Yell", "Emergency"], "scenario": "Pool Party / Beach Hangout", "reasoning": "Music and casual tones near water suggest a recreational pool or beach gathering.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Sea & Water", "Water drops"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Quiet Fishing / Sailing", "reasoning": "Calm water acoustics with no extreme tones imply peaceful fishing or sailing.", "risk": 1, "urgency": "Low", "media": False},
            
            # 2. TRANSIT, VEHICLES & ROADWAYS
            {"any_scenes": ["Cars/Traffic"], "any_tones": ["Emergency", "Siren"], "scenario": "Car Crash / Major Collision", "reasoning": "Emergency calls in a heavy traffic environment point to a vehicular accident.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"any_scenes": ["Cars/Traffic"], "all_tones": ["Frustration", "Scream/Yell"], "scenario": "Severe Road Rage", "reasoning": "Screaming and frustration in traffic acoustics characterize a road rage incident.", "risk": 6, "urgency": "Moderate", "media": False},
            {"any_scenes": ["Train", "Aviation"], "any_tones": ["War/Crisis"], "scenario": "Transit Hijacking / Terror Crisis", "reasoning": "War and crisis terminology in a transit hub implies a severe hijacking event.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Train"], "all_tones": ["Frustration", "Scream/Yell"], "scenario": "Subway / Transit Brawl", "reasoning": "Frustration and screaming inside public transit indicates a physical altercation.", "risk": 6, "urgency": "Moderate", "media": False},
            {"any_scenes": ["Aviation", "Train"], "any_tones": ["Broadcast"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Transit PA Announcement", "reasoning": "Formal broadcast speech in a transit environment is highly characteristic of PA announcements.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Train", "Aviation"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Quiet Train / Flight Ride", "reasoning": "Public transit acoustics with silence indicates a peaceful commute.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Cars/Traffic"], "any_tones": ["Frustration"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Traffic Jam / Idling", "reasoning": "Vehicular acoustics with mild frustration implies a traffic jam.", "risk": 2, "urgency": "Low", "media": False},
            {"any_scenes": ["Cars/Traffic"], "all_tones": ["Action/Combat", "Siren"], "scenario": "Police Chase / Pursuit", "reasoning": "Action/Combat sounds mixed with traffic and sirens indicate a police pursuit.", "risk": 8, "urgency": "High", "media": False},

            # 3. PUBLIC SPACES & CROWDS
            {"any_scenes": ["Clapping", "Laughing"], "all_tones": ["War/Crisis", "Scream/Yell"], "scenario": "Mass Shooting / Terror Attack", "reasoning": "Screaming and war/crisis terms in a crowded space are markers of an active attack.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Music", "Clapping"], "any_tones": ["Emergency", "Scream/Yell"], "scenario": "Concert Stampede / Panic", "reasoning": "Distress and screaming overlaying music indicates a severe crowd crush.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Clapping", "Cars/Traffic"], "all_tones": ["Scream/Yell", "Frustration"], "scenario": "Violent Riot / Civil Unrest", "reasoning": "Crowds, traffic, frustration, and screaming characterize a violent street riot.", "risk": 8, "urgency": "High", "media": False},
            {"any_scenes": ["Clapping", "Cars/Traffic"], "any_tones": ["Broadcast"], "exclude_tones": ["War/Crisis"], "scenario": "Political Rally / Organized Protest", "reasoning": "Broadcast-style speech directed at a crowd indicates an organized protest.", "risk": 3, "urgency": "Low", "media": False},
            {"any_scenes": ["Clapping", "Music"], "any_tones": ["Casual"], "exclude_tones": ["Scream/Yell", "War/Crisis", "Emergency"], "scenario": "Flash Mob / Public Performance", "reasoning": "Music and casual crowd interaction suggests a public performance.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Glass breaking", "Laughing"], "all_tones": ["Frustration", "Scream/Yell"], "scenario": "Bar Brawl / Public Altercation", "reasoning": "Screaming and frustration in a dining/cafe environment indicates a physical altercation.", "risk": 7, "urgency": "High", "media": False},
            {"any_scenes": ["Laughing"], "any_tones": ["Broadcast"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Stand-up Comedy / Live Show", "reasoning": "Broadcast-style speech directed at a relaxed crowd suggests a comedy show.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Clapping", "Laughing"], "any_tones": ["Frustration", "Casual"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Live Stadium / Fan Reaction", "reasoning": "Crowd noise mixed with passionate casual/frustration points to an in-person stadium crowd.", "risk": 1, "urgency": "Low", "media": False},
            
            # 4. OFFICE & PROFESSIONAL
            {"any_scenes": ["Office"], "all_tones": ["War/Crisis", "Scream/Yell"], "scenario": "Office Active Shooter", "reasoning": "Screaming and crisis keywords in an office environment point to an active shooter.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Office"], "any_tones": ["Emergency", "Siren"], "scenario": "Workplace Fire / Evacuation", "reasoning": "Sirens and emergency terminology in an office dictate a fire alarm.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"any_scenes": ["Office"], "all_tones": ["Professional", "Broadcast"], "exclude_tones": ["War/Crisis", "Emergency"], "scenario": "Corporate Presentation", "reasoning": "Formal speech mixed with professional keywords indicates a corporate presentation.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Office"], "all_tones": ["Professional", "Frustration"], "scenario": "Workplace Dispute / Firing", "reasoning": "Professional terminology mixed with frustration indicates a workplace dispute.", "risk": 4, "urgency": "Moderate", "media": False},
            {"any_scenes": ["Office"], "any_tones": ["Silence / Unknown"], "exclude_tones": ["War/Crisis", "Emergency", "Scream/Yell"], "scenario": "Solo Office Working (After Hours)", "reasoning": "Silence and footsteps in an office indicates solo working.", "risk": 1, "urgency": "Low", "media": False},

            # 5. DOMESTIC & HOME
            {"any_scenes": ["Door sounds", "Vacuum cleaner", "Washing machine"], "all_tones": ["War/Crisis", "Scream/Yell"], "scenario": "Home Invasion / Armed Burglary", "reasoning": "Screaming and crisis keywords inside a home indicate a home invasion.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Door sounds", "Glass breaking"], "any_tones": ["Scream/Yell"], "exclude_tones": ["War/Crisis", "Action/Combat", "Sci-Fi/Fantasy/Drama", "Casual"], "scenario": "Domestic Abuse / Violent Dispute", "reasoning": "Screaming in a domestic environment (without casual markers) strongly implies severe domestic violence.", "risk": 8, "urgency": "High", "media": False},
            {"any_scenes": ["Door sounds", "Vacuum cleaner"], "any_tones": ["Emergency", "Siren"], "scenario": "Domestic Medical Emergency", "reasoning": "Emergency keywords and sirens in a home environment point to an acute medical crisis.", "risk": 9, "urgency": "CRITICAL", "media": False},
            {"any_scenes": ["Door sounds", "Can opening"], "all_tones": ["Casual", "Frustration"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Online Gaming / eSports LAN", "reasoning": "Casual tones mixed with sporadic frustration characterize online gaming.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Door sounds", "Laughing"], "all_tones": ["Casual"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Dinner Party / Family Gathering", "reasoning": "A crowd of casual speakers inside a home indicates a dinner party.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Crying baby", "Snoring"], "exclude_tones": ["Scream/Yell"], "scenario": "Domestic Sleep / Nursery", "reasoning": "Biological sleep/infant sounds in a quiet house point to a nursery or sleep period.", "risk": 1, "urgency": "Low", "media": False},

            # 6. NATURE & WILDLIFE
            {"any_scenes": ["Dog", "Cow", "Pig", "Sheep", "Cat", "Insects", "Crow"], "any_tones": ["Emergency", "Scream/Yell"], "scenario": "Animal Attack / Predator Threat", "reasoning": "Screaming and emergency tones alongside wildlife acoustics indicate a severe animal attack.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Dog", "Crow", "Insects"], "any_tones": ["War/Crisis", "Action/Combat"], "scenario": "Hunting / Gunshots Outdoors", "reasoning": "War/Combat keywords alongside wildlife acoustics imply hunting.", "risk": 6, "urgency": "Moderate", "media": False},
            {"any_scenes": ["Rain & Thunder", "Wind"], "any_tones": ["Emergency"], "scenario": "Lost / Hiking Distress", "reasoning": "Emergency calls isolated in a natural environment imply a lost hiker.", "risk": 8, "urgency": "High", "media": False},
            {"any_scenes": ["Rain & Thunder", "Wind"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Nature Relaxation / Camping", "reasoning": "Peaceful nature acoustics with casual or no speech indicate camping.", "risk": 1, "urgency": "Low", "media": False},

            # 7. INDUSTRIAL & CONSTRUCTION
            {"any_scenes": ["Saws"], "any_tones": ["Emergency", "Siren"], "scenario": "Industrial Accident / Machinery Failure", "reasoning": "Sirens and emergency calls in a construction zone point to an industrial accident.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Saws"], "any_tones": ["War/Crisis"], "scenario": "Active Demolition", "reasoning": "Crisis/Blast keywords in a construction environment indicate an active demolition.", "risk": 5, "urgency": "Moderate", "media": False},
            {"any_scenes": ["Saws"], "any_tones": ["Professional"], "exclude_tones": ["Scream/Yell", "Emergency", "War/Crisis"], "scenario": "Standard Construction / Tool Use", "reasoning": "Professional terminology mixed with construction acoustics indicates normal tool operation.", "risk": 2, "urgency": "Low", "media": False},
            
            # 8. FIRE & PYROTECHNICS
            {"any_scenes": ["Crackling fire"], "any_tones": ["Emergency", "Scream/Yell"], "scenario": "Building/Wildfire Trap", "reasoning": "Fire acoustics mixed with distress indicates a trapped individual in a fire.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Crackling fire"], "exclude_tones": ["Emergency", "Scream/Yell"], "scenario": "Campfire / Bonfire", "reasoning": "Fire acoustics with calm tones indicate a recreational campfire.", "risk": 1, "urgency": "Low", "media": False},
            {"any_scenes": ["Fireworks"], "any_tones": ["War/Crisis"], "scenario": "Combat / Artillery Fire", "reasoning": "Explosive sounds matched with crisis terminology indicate actual combat rather than celebration.", "risk": 10, "urgency": "EXTREME", "media": False},
            {"any_scenes": ["Fireworks"], "exclude_tones": ["War/Crisis"], "scenario": "Public Fireworks Display", "reasoning": "Explosions coupled with clapping/crowds without distress indicate a celebration.", "risk": 1, "urgency": "Low", "media": False}
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
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        formal_score = 0
        casual_score = 0
        
        if avg_word_length > 4.5: formal_score += 1
        if unique_words / max(1, word_count) > 0.8 and word_count > 10: formal_score += 1
        if unique_words / max(1, word_count) < 0.5 and word_count > 15: casual_score += 2
        
        transcript_lower = transcript.lower()
        
        # New logic: Check substring to allow multi-word keywords (e.g. "steve rogers")
        detected_tones = []
        for tone, keywords in self.keyword_categories.items():
            for kw in keywords:
                # Use regex with word boundaries to avoid partial word matches
                if re.search(rf'\b{re.escape(kw)}\b', transcript_lower):
                    if tone not in detected_tones:
                        detected_tones.append(tone)
                    
                    if tone == "Casual": casual_score += 1
                    if tone == "Broadcast": formal_score += 1
        
        for phrase in self.phrase_categories["Broadcast"]:
            if phrase in transcript_lower:
                formal_score += 5
                
        if formal_score > casual_score:
            style = "Formal/Structured"
        elif casual_score > formal_score:
            style = "Casual/Conversational"
        else:
            style = "Neutral"
            
        if unique_words / max(1, word_count) < 0.4 and word_count > 15:
            style = "Highly Repetitive (Potential Lyrics/Chant)"
                
        if re.search(r'\b(ah{2,}|a{3,}|no{3,}|oh{3,}|argh)\b', transcript_lower):
            if "Scream/Yell" not in detected_tones:
                detected_tones.append("Scream/Yell")
                
        for tone, phrases in self.phrase_categories.items():
            if tone not in detected_tones:
                for phrase in phrases:
                    if phrase in transcript_lower:
                        detected_tones.append(tone)
                        break
                
        tone_str = ", ".join(detected_tones) if detected_tones else "Neutral"
        return style, tone_str, detected_tones

    def _evaluate_nate(self, scenario, active_scenes, timeline, is_media):
        """
        Neuro-Acoustic Temporal Expectation (NATE) logic based on Predictive Coding.
        Generates expectations and measures acoustic mismatches (MMN) and proximity tracking.
        """
        nate_report = []
        mismatch_score = "Low"
        
        # 1. Predictive Coding Mismatch (Semantic vs Acoustic Expectation)
        if "Media" in scenario or is_media:
            nate_report.append("Generative Expectation: Highly scripted and synthetically controlled acoustics.")
            if "Silence / Unknown" in [s[0] for s in active_scenes]:
                mismatch_score = "High"
                nate_report.append("Prediction Error: The environment is silent despite media semantic cues. Indicates potential false positive on speech translation or deepfake artifacts.")
            else:
                nate_report.append("Sensory Alignment: Low prediction error. The acoustics match the semantic media profile.")
        elif "Crisis" in scenario or "Emergency" in scenario or "Abuse" in scenario:
            nate_report.append("Generative Expectation: High-energy transient acoustics, sirens, or abrupt amplitude spikes.")
            if [s[0] for s in active_scenes][0] == "Silence / Unknown" or [s[0] for s in active_scenes][0] == "Domestic":
                mismatch_score = "Critical"
                nate_report.append("Prediction Error (Mismatch Negativity - MMN): Speech suggests a severe crisis, but bottom-up acoustics represent a calm environment. High cognitive dissonance.")
            else:
                nate_report.append("Sensory Alignment: High alignment. The expected acoustic chaos of a crisis is present.")
        else:
            nate_report.append("Generative Expectation: Stable, continuous background acoustics appropriate to the identified scenario.")
            mismatch_score = "Low"

        # 2. Proximity and Pitch Tracking (Temporal Variance)
        temporal_track = "Static/No Movement"
        if timeline and len(timeline) >= 2:
            rms_values = [t.get("rms", 0) for t in timeline]
            pitch_values = [t.get("pitch", 0) for t in timeline]
            
            if rms_values[-1] > rms_values[0] * 1.5:
                temporal_track = "Approaching (Proximity Alert)"
            elif rms_values[-1] < rms_values[0] * 0.6:
                temporal_track = "Receding"
                
            pitch_max = np.max(pitch_values)
            if pitch_max > 4000:
                temporal_track += " | High-Pitch Escalation (Potential alarm or distress vocalization)"
        
        nate_report.append(f"Proximity & Trajectory: {temporal_track}")
        return "\n".join([f"* {r}" for r in nate_report]), mismatch_score

    def _evaluate_hifs(self, scenario, mismatch_score, is_media):
        """
        Quality of Experience (QoE) / Human Influence Factors (HIF) 
        Estimates the cognitive and emotional impact on a human listener.
        """
        hif_report = []
        
        # Emotional Load (Low-level HIF)
        if "Crisis" in scenario or "Emergency" in scenario or "Attack" in scenario or "Abuse" in scenario:
            stress_level = "Severe Stress Induction"
            hif_report.append("Emotional Impact: High physiological arousal, inducing stress and fight-or-flight responses.")
        elif "Music" in scenario or "Party" in scenario or "Comedy" in scenario:
            stress_level = "Low (Positive Valence)"
            hif_report.append("Emotional Impact: Positive arousal, mood enhancement, recreational listening.")
        else:
            stress_level = "Neutral"
            hif_report.append("Emotional Impact: Baseline emotional state, non-intrusive environment.")
            
        # Cognitive Load & Competence (High-level HIF)
        if mismatch_score in ["High", "Critical"]:
            hif_report.append("Cognitive Load: Extremely High. Conflicting multi-modal information requires high cognitive effort to parse the true context.")
            hif_report.append("Required Competence: Domain-Specific Innovativeness (DSI) or professional training needed to resolve the ambiguity.")
        elif is_media:
            hif_report.append("Cognitive Load: Low/Passive. Media consumption requires minimal active problem-solving.")
            hif_report.append("Required Competence: Basic media literacy.")
        elif stress_level == "Severe Stress Induction":
            hif_report.append("Cognitive Load: High/Overwhelming. The scenario demands rapid decision-making under duress.")
            hif_report.append("Required Competence: Emergency response training or high socio-cultural resilience.")
        else:
            hif_report.append("Cognitive Load: Moderate. Typical daily environmental processing.")
            hif_report.append("Required Competence: General real-world experience.")
            
        return "\n".join([f"* {r}" for r in hif_report])

    def _calculate_semantic_acoustic_alignment(self, transcript, active_scenes, detected_tones, is_media):
        """
        Scores the alignment between the semantic transcript and the acoustic scene.
        """
        score = 50 # Base neutral score
        if is_media and ("Media" in [s[0] for s in active_scenes] or "Music" in [s[0] for s in active_scenes]):
            score += 40
        if not is_media:
            crisis_tones = {"War/Crisis", "Emergency", "Weather/Disaster", "Action/Combat", "Scream/Yell"}
            has_crisis = bool(set(detected_tones).intersection(crisis_tones))
            has_siren = "Siren" in [s[0] for s in active_scenes]
            if has_crisis and has_siren:
                score += 45
            elif has_crisis and "Silence / Unknown" in [s[0] for s in active_scenes]:
                score -= 30
            elif not has_crisis and "Silence / Unknown" in [s[0] for s in active_scenes]:
                score += 30
        
        score = max(0, min(100, score))
        if score > 80: alignment_text = "High Alignment (Congruent)"
        elif score > 40: alignment_text = "Moderate Alignment (Ambiguous)"
        else: alignment_text = "Low Alignment (Dissonant / Mismatch)"
        
        return f"{score}% - {alignment_text}"

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
            if "Siren" in scene_names or "Scream/Yell" in scene_names:
                return "Independent", "The acoustic event appears unrelated to the speaker's activity. Speech remains casual or neutral while the environment is active."
            return "Ambiguous", "No significant relationship between speech and environment is evident."
            
        crisis_tones = {"War/Crisis", "Emergency", "Weather/Disaster", "Action/Combat", "Scream/Yell"}
        crisis_scenes = {"Siren"}
        
        has_crisis_tone = bool(set(detected_tones).intersection(crisis_tones))
        has_crisis_scene = bool(set(scene_names).intersection(crisis_scenes))
        
        if has_crisis_tone and has_crisis_scene:
            if "Siren" in scene_names and scene_names.index("Siren") == 0:
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
                
        if "Siren" in scene_names:
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
        
        if strength["Speech Evidence"] == "Strong": drivers.append("Strong semantic clarity in speech")
        if strength["Environmental Evidence"] == "Strong": drivers.append("Stable primary scene classification")
        if strength["Temporal Evidence"] == "Strong": drivers.append("Consistent temporal variance")
        
        if "Reinforcement" in rel_type:
            drivers.append("Speech and environment cross-modally agree")
            
        if strength["Speech Evidence"] == "Weak": limitations.append("Poor acoustic speech clarity")
        if strength["Environmental Evidence"] == "Weak": limitations.append("Weak acoustic scene probabilities")
        if strength["Overall Evidence Strength"] == "Weak": limitations.append("Overall lack of distinct evidence markers")
        
        if rel_type == "Contradictory":
            limitations.append("Speech and environmental modalities conflict")
            score = "Low"
            
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
            
        if "Emergency" in scenario or "Crisis" in scenario or "Attack" in scenario or "Abuse" in scenario:
            if has_crisis_tone and has_crisis_scene and rel_type in ["Strong Reinforcement", "Moderate Reinforcement"]:
                return "Very High"
            return "High"
            
        if has_crisis_tone or has_crisis_scene:
            return "Moderate"
            
        return "Low"
        
    def _generate_analyst_conclusion(self, rel_type, strength, scenario, is_media):
        conclusion = ""
        if is_media:
            conclusion = "The evidence strongly points to media playback rather than a live physical event. The acoustic profile aligns with broadcast or recorded content, mitigating potential physical risk."
            return conclusion
            
        if "Reinforcement" in rel_type and strength["Overall Evidence Strength"] in ["Strong", "Moderate"]:
            conclusion = f"Both speech semantics and environmental acoustics point toward a {scenario.lower()}. The consistent pattern across modalities strengthens the assessment."
        elif rel_type == "Contradictory":
            conclusion = f"There is a stark contradiction between the detected environment and the speech. While one modality suggests {scenario.lower()}, the other does not align. This mismatch signals a potential false positive or complex hidden event."
        elif rel_type == "Independent":
            conclusion = f"Acoustic events are occurring independently of the detected speech. The {scenario.lower()} appears to be a backdrop to unrelated human activity."
        else:
            conclusion = f"The overall pattern suggests a {scenario.lower()}, though the evidence is {strength['Overall Evidence Strength'].lower()}."
            
        conclusion += " Verification through secondary visual or sensory feeds is recommended to confirm the situation."
        return conclusion

    def analyze(self, transcript: str, scene_probs: list, t_conf: float, timeline: list = None):
        transcript_style, tone_str, detected_tones = self._linguistic_analysis(transcript)
        has_speech = len(transcript.strip()) > 0
        
        active_scenes = []
        for i, prob in enumerate(scene_probs):
            if prob > self.threshold:
                active_scenes.append((SCENE_LABELS[i], prob))
        
        if has_speech and ("Highly Repetitive" in transcript_style or "Music/Song" in detected_tones):
            if "Music" not in [s[0] for s in active_scenes]:
                active_scenes.append(("Music", 0.50))
                
        if has_speech and "Broadcast" in detected_tones:
            if "Television / Media" not in [s[0] for s in active_scenes]:
                active_scenes.append(("Television / Media", 0.50))
        
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
            if "any_scenes" in rule:
                if not any(s in scene_names for s in rule["any_scenes"]):
                    scene_match = False
                    
            tone_match = True
            if "any_tones" in rule:
                if not any(t in detected_tones for t in rule["any_tones"]):
                    tone_match = False
            if "all_tones" in rule:
                if not all(t in detected_tones for t in rule["all_tones"]):
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
                elif "Clapping" in scene_names or "Laughing" in scene_names:
                    scenario = "Public Social Interaction"
                else:
                    scenario = f"Human Activity ({env})"
            else:
                if top_scene_prob > 0.80 and top_scene_name not in ["Silence / Unknown", "Domestic"]:
                    scenario = f"Acoustic Event: {top_scene_name}"
                else:
                    scenario = f"Pure Environmental Noise ({env})"

        # Temporal Awareness
        timeline_active = False
        if timeline and len(timeline) > 1:
            timeline_active = True
            
        # Execute Sub-Modules
        strength = self._evaluate_evidence_strength(has_speech, detected_tones, active_scenes, t_conf, timeline, is_media)
        observations = self._generate_specific_observations(transcript, active_scenes, timeline_active, is_media, detected_tones)
        rel_type, rel_explanation = self._evaluate_relationship(detected_tones, scene_names, is_media, transcript_style)
        env_interpretation = self._interpret_environment(scene_names, active_scenes, env)
        alt_interpretation = self._generate_alternative(scenario, is_media, env)
        uncertainty = self._generate_uncertainty(t_conf, is_media, active_scenes, has_speech)
        conf_score, conf_drivers, conf_limitations = self._calculate_confidence(strength, rel_type)
        
        nate_text, mismatch_score = self._evaluate_nate(scenario, active_scenes, timeline, is_media)
        hif_text = self._evaluate_hifs(scenario, mismatch_score, is_media)
        semantic_acoustic_align = self._calculate_semantic_acoustic_alignment(transcript, active_scenes, detected_tones, is_media)
        
        crisis_tones = {"War/Crisis", "Emergency", "Weather/Disaster", "Action/Combat"}
        has_crisis_tone = bool(set(detected_tones).intersection(crisis_tones))
        has_crisis_scene = bool(set(scene_names).intersection({"Siren"}))
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
        
        response += "Neuro-Acoustic Temporal Expectation (NATE)\n"
        response += f"{nate_text}\n"
        response += "⸻\n"
        
        response += "Human Influence Factors (HIF)\n"
        response += f"{hif_text}\n"
        response += "⸻\n"

        response += "Semantic-Acoustic Alignment\n"
        response += f"{semantic_acoustic_align}\n"
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
        
        return response, active_scenes, 0, is_media
