import warnings
warnings.filterwarnings('ignore')
import librosa
from core.inference_pipeline import ALMInferencePipeline
import sys

def analyze_file(pipeline, file_path):
    print(f"\n=============================================")
    print(f"Analyzing: {file_path}")
    print(f"=============================================")
    print("Loading audio...")
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    print("Running inference...")
    msnl_out, scene, conf, ai_text = pipeline.run(audio, sr)
    
    print("\n====== CASRE REPORT ======")
    print(ai_text)
    print("\n")

def main():
    print("Loading pipeline...")
    pipeline = ALMInferencePipeline()
    
    files = [
        'samples/YTDown_Shorts_Loki-s-Glorious-Purpose_Media_KPz1r8DCuH0_009_128k.mp3',
        'samples/I am Steve Rogers. - Marvel Entertainment (128k).mp3'
    ]
    
    for f in files:
        analyze_file(pipeline, f)

if __name__ == "__main__":
    main()
