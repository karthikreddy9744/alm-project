import os
import argparse
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

# Supported FLEURS languages (BCP-47 codes)
FLEURS_LANGS = {
    "hindi": "hi_in",
    "telugu": "te_in",
    "tamil": "ta_in",
    "kannada": "kn_in",
    "malayalam": "ml_in",
    "marathi": "mr_in",
    "bengali": "bn_in",
    "gujarati": "gu_in",
    "punjabi": "pa_in",
}

# ESC-50 classes to map to Ambient Beds
AMBIENT_CLASSES = {
    "rain": "Rain & Thunder", "wind": "Wind", "sea_waves": "Sea & Water", 
    "crackling_fire": "Crackling fire", "crickets": "Crickets", "chirping_birds": "Chirping birds",
    "water_drops": "Water drops", "helicopter": "Aviation", "chainsaw": "Saws",
    "siren": "Siren", "engine": "Cars/Traffic", "train": "Train"
}

# ESC-50 classes to map to exact SCENE_LABELS
EVENT_CLASSES = {
    "dog": "Dog", "rooster": "Poultry", "pig": "Pig", "cow": "Cow", "frog": "Frog", 
    "cat": "Cat", "hen": "Poultry", "insects": "Insects", "sheep": "Sheep", "crow": "Crow", 
    "crying_baby": "Crying baby", "sneezing": "Coughing & Sneezing", "clapping": "Clapping", 
    "breathing": "Breathing", "coughing": "Coughing & Sneezing", "footsteps": "Footsteps", 
    "laughing": "Laughing", "brushing_teeth": "Personal Care", "snoring": "Snoring",
    "drinking_sipping": "Personal Care", "door_wood_knock": "Door sounds", "mouse_click": "Office", 
    "keyboard_typing": "Office", "door_wood_creaks": "Door sounds", "can_opening": "Can opening", 
    "washing_machine": "Washing machine", "vacuum_cleaner": "Vacuum cleaner",
    "clock_alarm": "Clock", "clock_tick": "Clock", "glass_breaking": "Glass breaking", "car_horn": "Cars/Traffic"
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_audio(audio_array, sr, path, target_sr=16000):
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
    sf.write(path, audio_array, target_sr)

def download_speech(output_dir, max_samples=500):
    speech_dir = os.path.join(output_dir, "speech")
    ensure_dir(speech_dir)
    
    print("Downloading LibriSpeech (English)...")
    en_dir = os.path.join(speech_dir, "english")
    ensure_dir(en_dir)
    try:
        ls = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
        count = 0
        for item in tqdm(ls, total=max_samples, desc="LibriSpeech"):
            save_path = os.path.join(en_dir, f"en_librispeech_{count}.wav")
            if not os.path.exists(save_path):
                audio = item["audio"]
                save_audio(audio["array"], audio["sampling_rate"], save_path)
            count += 1
            if count >= max_samples:
                break
    except Exception as e:
        print(f"Failed to download LibriSpeech: {e}")

    print("Downloading Google FLEURS (Multi-lingual)...")
    for lang_name, lang_code in FLEURS_LANGS.items():
        lang_dir = os.path.join(speech_dir, lang_name)
        ensure_dir(lang_dir)
        try:
            fl = load_dataset("google/fleurs", lang_code, split="validation", streaming=True)
            count = 0
            for item in tqdm(fl, total=max_samples//5, desc=f"FLEURS {lang_name}"):
                save_path = os.path.join(lang_dir, f"{lang_code}_fleurs_{count}.wav")
                if not os.path.exists(save_path):
                    audio = item["audio"]
                    save_audio(audio["array"], audio["sampling_rate"], save_path)
                count += 1
                if count >= max_samples // 5:
                    break
        except Exception as e:
            print(f"Failed to download FLEURS {lang_name}: {e}")

def download_environment(output_dir, max_samples=50):
    ambient_dir = os.path.join(output_dir, "ambient")
    events_dir = os.path.join(output_dir, "events")
    ensure_dir(ambient_dir)
    ensure_dir(events_dir)
    
    print("Downloading ESC-50 (Environment & Events)...")
    try:
        esc50 = load_dataset("ashraq/esc50", split="train", streaming=True)
        counts = {c: 0 for c in list(AMBIENT_CLASSES.keys()) + list(EVENT_CLASSES.keys())}
        
        for item in tqdm(esc50, desc="ESC-50"):
            category = item["category"]
            
            if category in AMBIENT_CLASSES and counts[category] < max_samples:
                mapped_ambient_cat = AMBIENT_CLASSES[category]
                cat_dir = os.path.join(ambient_dir, mapped_ambient_cat)
                ensure_dir(cat_dir)
                save_path = os.path.join(cat_dir, f"{category}_{counts[category]}.wav")
                if not os.path.exists(save_path):
                    audio = item["audio"]
                    save_audio(audio["array"], audio["sampling_rate"], save_path)
                counts[category] += 1
                
            elif category in EVENT_CLASSES and counts.get(category, 0) < max_samples:
                mapped_event_cat = EVENT_CLASSES[category]
                cat_dir = os.path.join(events_dir, mapped_event_cat)
                ensure_dir(cat_dir)
                save_path = os.path.join(cat_dir, f"{category}_{counts[category]}.wav")
                if not os.path.exists(save_path):
                    audio = item["audio"]
                    save_audio(audio["array"], audio["sampling_rate"], save_path)
                counts[category] += 1
                
            if all(c >= max_samples for c in counts.values()):
                break
                
    except Exception as e:
        print(f"Failed to download ESC-50: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Real ALM Datasets")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--max_speech", type=int, default=500, help="Max speech samples per source")
    parser.add_argument("--max_env", type=int, default=10, help="Max environmental samples per class")
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "metadata"))
    
    download_speech(args.output_dir, max_samples=args.max_speech)
    download_environment(args.output_dir, max_samples=args.max_env)
    print(f"Dataset downloaded successfully to {args.output_dir}!")
