import torch
import numpy as np
import librosa
from core_modules.feature_extractor import WhisperFeatureExtractor, CLAPFeatureExtractor
from reasoning_engine.awm.world_model import AuditoryWorldModel
from reasoning_engine.awm.models import EntityNode, EventNode, NodeState, Trajectory, HierarchicalConfidence
from reasoning_engine.fusion.models import RecordingCharacterization


class ALMInferencePipeline:
    """
    ALM v12.7 Unified Inference Pipeline.
    Bridges the Neural Perception Layer (Whisper/CLAP/HTS-AT + Fusion) 
    with the Deterministic Cognitive Graph (AWM).
    """
    def __init__(self, model_path=None):
        print("Initializing Neural Perception Extractors...")
        self.whisper_fe = WhisperFeatureExtractor('base')
        self.clap_fe = CLAPFeatureExtractor()
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def process(self, audio: np.ndarray, sr: int, awm: AuditoryWorldModel) -> AuditoryWorldModel:
        """
        Runs the neural perception pipeline and populates the given AuditoryWorldModel.
        """
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # 1. Feature Extraction
        w_emb, transcript, dominant_lang, timestamps = self.whisper_fe.extract(audio, sr, extract_text=True)
        c_emb = self.clap_fe.extract(audio, sr)
        
        # 2.5 Recording Characterization
        awm.recording_characterization = self._characterize_recording(audio, sr)
            
        # CLAP semantic concepts (Mega-Scale Universal Acoustic Dictionary: 160+ Unambiguous Scenarios)
        concept_list = [
            # 1. Extreme Weather & Natural Disasters
            "destructive hurricane or cyclone winds", "torrential monsoon rain and flooding", 
            "a violent thunderstorm with lightning strikes", "a raging forest fire crackling", 
            "a massive earthquake rumbling", "a harsh blizzard with howling snow winds",
            "a massive tornado destroying property", "a dangerous avalanche of snow falling",
            "a volcano erupting with flowing lava", "a dry dusty sandstorm in a desert",
            "hailstones crashing against windows and roofs", "a powerful tsunami wave crashing",
            "a severe drought with dry cracking dirt", "a heavy mudslide crashing down a hill",
            "a dense eerie fog with low visibility",
            
            # 2. Nature & Wildlife Ecosystems
            "a peaceful dense jungle with exotic birds", "crashing ocean waves on a rocky beach", 
            "a gentle flowing river in a forest", "crickets and frogs in a swamp at night", 
            "an underwater marine environment", "a barren desert with dry howling wind",
            "a quiet mountain peak with light breeze", "a damp underground cave with water dripping",
            "a noisy colony of seagulls by the sea", "a serene meadow with buzzing bees and insects",
            "a large waterfall cascading down rocks", "a dark spooky forest at midnight",
            "a tropical rainforest with monkeys howling", "a freezing glacier with ice cracking",
            "wolves howling in a snowy tundra",
            
            # 3. Urban Infrastructure & Street Life
            "heavy gridlock traffic on a city highway", "a busy pedestrian crosswalk in a metropolis", 
            "a quiet suburban street at night", "a crowded city square with people walking",
            "a loud garbage truck picking up trash bins", "a noisy street sweeper cleaning roads",
            "a bustling outdoor farmer's market", "a street musician playing an acoustic guitar",
            "a lively city park with people chatting", "a dark alleyway with distant city hum",
            "a busy bus stop with people boarding", "water splashing from a passing car in a puddle",
            "a high-rise rooftop with strong wind", "a parade marching down a main street",
            "a protest march with megaphones and chanting",
            
            # 4. Transport & Vehicles
            "an underground subway train arriving at a station", "a large commercial airplane taking off from an airport", 
            "a helicopter hovering at low altitude", "a loud diesel train passing over tracks",
            "a motorcycle revving its engine loudly", "a bustling harbor with ferry boats blowing horns",
            "a high-speed bullet train zooming past", "a large cargo ship blowing its foghorn",
            "a bicycle bell ringing on a quiet trail", "a small propeller plane flying overhead",
            "a car driving on a bumpy gravel road", "a screeching car braking suddenly",
            "a loud sports car accelerating fast", "a noisy tractor trailer truck on a highway",
            "a crowded bus interior with engine rumbling",
            
            # 5. Industrial, Construction & Tools
            "a heavy construction site with jackhammers", "a mechanized automotive factory assembly line", 
            "a noisy carpentry workshop with electric saws", "a busy shipping port with cargo cranes", 
            "a large server room with continuous fan humming", "a metalworking shop with sparks and welding",
            "a busy logging site with chainsaws cutting trees", "an underground coal mine with drilling machinery",
            "a giant printing press printing newspapers", "a bustling warehouse with forklifts beeping",
            "a loud pneumatic drill breaking concrete", "a blacksmith hammering hot metal on an anvil",
            "a large oil rig pumping machinery", "a busy shipyard constructing a massive vessel",
            "a textile mill with loud weaving looms",
            
            # 6. Medical & Emergency Services
            "a frantic hospital emergency room", "an ambulance siren blaring through traffic", 
            "a police car with sirens in pursuit", "a fire truck siren and air horn responding to a fire", 
            "a medical heart monitor beeping in an ICU", "a dentist office with a drill operating",
            "a busy 911 emergency dispatch call center", "a rescue helicopter searching for survivors",
            "paramedics treating a patient outdoors", "a hospital waiting room with coughing patients",
            "an MRI machine making loud rhythmic knocking sounds", "a busy pharmacy with cash registers",
            "a chaotic disaster triage tent", "a quiet hospice or care home",
            "a physical therapy room with exercise equipment",
            
            # 7. Commercial, Retail & Dining
            "a crowded indoor shopping mall", "a busy supermarket with shopping carts and cash registers", 
            "a noisy local street market with vendors", "a quiet corporate office with keyboard typing", 
            "a lively restaurant dining room with clinking silverware", "a loud crowded nightclub or bar",
            "a busy banking hall with tellers talking", "a quiet bookstore or library with pages turning",
            "a local barber shop with electric clippers buzzing", "a high-end casino floor with slot machines ringing",
            "a fast food drive-thru intercom", "a busy coffee shop with espresso machines hissing",
            "a large department store with background music", "a noisy fish market with ice crushing",
            "a quiet luxury boutique with soft ambient music",
            
            # 8. Entertainment, Sports & Recreation
            "a massive stadium crowd roaring during a sports game", "a live rock music concert with heavy bass", 
            "a classical symphonic orchestra performance", "an arcade with loud retro gaming electronic sounds", 
            "a bowling alley with balls striking pins", "a public swimming pool with water splashing",
            "a competitive basketball game with squeaking shoes", "a theatrical play with a live audience clapping",
            "a busy theme park with rollercoasters screaming", "a quiet art museum with echoing footsteps",
            "a tennis match with rackets hitting balls", "a golf course with a club striking a ball",
            "a loud boxing match with a bell ringing", "a lively ice skating rink",
            "a busy gym with weights clanking",
            
            # 9. Educational, Civic & Religious
            "a noisy elementary school playground", "a quiet university library or study hall", 
            "a solemn church or cathedral choir singing", "a high school classroom during a lecture",
            "a busy college cafeteria during lunch hour", "a loud school bell ringing in a hallway",
            "a quiet graduation ceremony outdoors", "a Buddhist temple with chanting monks",
            "an Islamic call to prayer from a mosque", "a busy science laboratory with bubbling beakers",
            "a courtroom trial with a judge's gavel", "a crowded town hall political meeting",
            "a bustling museum exhibit hall", "a quiet meditation retreat",
            "a rowdy college dormitory hallway",
            
            # 10. Domestic & Residential
            "a home kitchen with sizzling pans and cooking", "a bathroom shower running with water splashing", 
            "a noisy laundry room with a washing machine spinning", "a quiet living room with television playing in the background", 
            "a crying baby in a quiet nursery", "dogs barking aggressively inside a house",
            "a vacuum cleaner operating on a carpet", "glass shattering or breaking indoors",
            "wooden doors opening and slamming closed", "a ticking clock in a silent bedroom",
            "a toilet flushing in a bathroom", "a blender crushing ice in a kitchen",
            "a loud snoring person sleeping in bed", "water boiling in a tea kettle",
            "a cat meowing and purring on a couch",
            
            # 11. Broadcast, Narrative & Abstract
            "a professional television news studio broadcast", "a high-energy sports commentary broadcast", 
            "a static-heavy emergency radio transmission", "a documentary voiceover narration",
            "a military battlefield or combat zone with explosions", "rapid machine gun fire in a warzone",
            "a massive military airstrike or bombing bombardment", "heavy artillery explosions destroying a building",
            "a high-speed car chase with screeching tires", "a chaotic riot or protest with crowds chanting",
            "a police interrogation room", "a public address system announcing a train delay",
            "a sci-fi spaceship engine humming", "a creepy haunted house with eerie winds",
            "a retro vinyl record crackling", "an old film projector clicking",
            "a dramatic cinematic movie trailer soundscape",
            
            # 12. Softmax Sinks (To absorb probability mass when no environment is present)
            "a person speaking clearly", "complete absolute silence", "background room noise"
        ]
        nearest_concepts_dict = self.clap_fe.get_nearest_concepts(audio, sr, concept_list)
        
        # Filter out the sinks and low-similarity hallucinations
        valid_concepts = []
        for k, prob in nearest_concepts_dict.items():
            if k not in ["a person speaking clearly", "complete absolute silence", "background room noise"] and prob > 0.10:
                valid_concepts.append((k, round(prob, 2)))
                
        # Keep top 3 concepts
        sorted_concepts = sorted(valid_concepts, key=lambda x: x[1], reverse=True)[:3]
        nearest_concepts = [k for k, v in sorted_concepts]
        
        # 3. Deterministic AWM Population 
        # 4. Insert into World Model
        # Insert Speech Entity if present
        if transcript:
            speaker_conf = HierarchicalConfidence(speech_recognition=0.9, sound_detection=1.0)
            speaker = EntityNode(
                id="spk_1",
                entity_type="Speaker",
                state=NodeState.UNKNOWN,
                confidence=speaker_conf
            )
            # Add attributes manually if we need to track transcript inside AWM
            # Though strictly not part of the base EntityNode fields, Python objects allow it
            speaker.transcript = transcript 
            speaker.language = dominant_lang
            speaker.timestamps = [(t['start']/sr, t['end']/sr) for t in timestamps]
            awm.add_entity(speaker)
            
            speech_event_conf = HierarchicalConfidence(sound_detection=1.0)
            speech_event = EventNode(
                id="evt_speech_1",
                class_map="Speech Activity",
                trajectory=Trajectory.UNKNOWN,
                acoustic_salience=0.8,
                confidence=speech_event_conf
            )
            awm.add_event(speech_event)
            
        # 5. Dynamic Acoustic Masking Penalty
        # (Disabled: Cross-modal conflict resolution is now handled purely at the Semantic LLM layer)
        speech_penalty = 1.0

        # Insert Zero-Shot Environmental Events from CLAP
        for i, (concept, prob) in enumerate(sorted_concepts):
            # Base confidence from softmax
            base_conf = min(1.0, prob * 3.0) 
            clap_event_conf = HierarchicalConfidence(sound_detection=base_conf)
            
            # Apply dynamic masking penalty
            salience = min(1.0, base_conf + 0.3) * speech_penalty
            
            clap_event = EventNode(
                id=f"obs_clap_{i}",
                class_map=concept.title(),
                trajectory=Trajectory.UNKNOWN,
                acoustic_salience=salience,
                confidence=clap_event_conf,
                start_time=0.0,
                end_time=round(len(audio)/sr, 2),
                detector="CLAP"
            )
            awm.add_event(clap_event)
            
        awm.clap_concepts = nearest_concepts
            
        return awm

    def _characterize_recording(self, audio: np.ndarray, sr: int) -> RecordingCharacterization:
        # Clipping: Check if max amplitude is near 1.0
        max_amp = np.max(np.abs(audio))
        clipping = bool(max_amp >= 0.99)
        
        # Dynamic Range
        rms = librosa.feature.rms(y=audio)[0]
        dynamic_range = "Wide" if np.std(rms) > 0.05 else "Compressed"
        
        # Reverb & Noise Profile (Heuristic)
        # Check if 'music' or 'studio' related scenes scored high in SceneNet
        # For simplicity in this pipeline, we use basic statistical heuristics
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        noise_profile = "High Frequency Noise" if np.mean(zcr) > 0.2 else "Clean"
        
        return RecordingCharacterization(
            reverberation_profile="Unknown",
            compression_level="High" if dynamic_range == "Compressed" else "Low",
            post_processing_artifacts=clipping,
            dynamic_range=dynamic_range,
            noise_profile=noise_profile,
            recording_quality="Poor" if clipping or noise_profile != "Clean" else "Good"
        )
