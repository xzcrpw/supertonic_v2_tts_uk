#!/usr/bin/env python3
"""
Manifest Preparation - DATA DRIVEN FIX (Jan 2026)
Correct column names based on inspection:
- OpenTTS: 'transcription'
- EuroSpeech: 'human_transcript'
- Podcasts: 'audio_filepath'
"""
import os
import sys
import json
import argparse
import random
import io
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import soundfile as sf
import pandas as pd

try:
    import torchaudio
except ImportError:
    pass

def get_audio_duration(file_path: str) -> float:
    try:
        f = sf.SoundFile(file_path)
        return len(f) / f.samplerate
    except Exception:
        return 0.0

def process_opentts(data_dir: Path) -> List[Dict]:
    """Extracts OpenTTS using column 'transcription'."""
    samples = []
    base_dir = data_dir / "opentts"
    if not base_dir.exists(): return samples

    print("🔍 Processing OpenTTS...")
    
    # Рекурсивний пошук parquet
    parquet_files = list(base_dir.rglob("*.parquet"))
    
    for p_file in parquet_files:
        # Визначаємо ім'я голосу з папки (opentts/lada/...)
        parts = p_file.parts
        try:
            # Шукаємо де 'opentts' і беремо наступну папку
            idx = parts.index('opentts')
            voice_name = parts[idx+1]
        except:
            voice_name = "unknown"

        wav_out = p_file.parent.parent / "extracted_wavs" # opentts/lada/extracted_wavs
        wav_out.mkdir(parents=True, exist_ok=True)
        
        try:
            df = pd.read_parquet(p_file)
            print(f"   👤 Voice: {voice_name} | File: {p_file.name} | Rows: {len(df)}")

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"   Extr. {voice_name}", leave=False):
                # !!! FIX: Correct column name
                text = row.get('transcription', '')
                if not text: continue
                
                audio_data = row.get('audio', {})
                if not isinstance(audio_data, dict): continue
                audio_bytes = audio_data.get('bytes')
                if not audio_bytes: continue
                
                wav_path = wav_out / f"{voice_name}_{idx}.wav"
                
                # Save WAV
                if not wav_path.exists():
                    try:
                        data, sr = sf.read(io.BytesIO(audio_bytes))
                        sf.write(str(wav_path), data, sr)
                    except: continue
                
                if wav_path.exists():
                    dur = get_audio_duration(str(wav_path))
                    if 0.5 < dur < 30:
                        samples.append({
                            "audio_path": str(wav_path),
                            "text": text,
                            "language": "uk",
                            "speaker_id": voice_name,
                            "duration": dur
                        })
        except Exception as e:
            print(f"   ❌ Error {p_file.name}: {e}")
            
    print(f"   ✅ OpenTTS total: {len(samples)}")
    return samples

def process_eurospeech(data_dir: Path, max_hours: int = 300) -> List[Dict]:
    """Extracts EuroSpeech using column 'human_transcript'."""
    samples = []
    base_dir = data_dir / "eurospeech-uk"
    if not base_dir.exists(): return samples

    print("🔍 Processing EuroSpeech...")
    parquet_files = list(base_dir.rglob("*.parquet"))
    parquet_files.sort()
    
    wav_out = base_dir / "extracted_wavs"
    wav_out.mkdir(parents=True, exist_ok=True)
    
    total_dur = 0
    limit_sec = max_hours * 3600
    global_idx = 0
    
    for p_file in tqdm(parquet_files, desc="Extracting EuroSpeech"):
        try:
            df = pd.read_parquet(p_file)
            
            for _, row in df.iterrows():
                # !!! FIX: Correct column name
                text = row.get('human_transcript', '')
                if not text: continue
                
                audio_data = row.get('audio')
                if not audio_data or not isinstance(audio_data, dict): continue
                audio_bytes = audio_data.get('bytes')
                if not audio_bytes: continue
                
                # Use key or global index for filename
                key = row.get('key', f'es_{global_idx}')
                wav_path = wav_out / f"{key}.wav"
                global_idx += 1
                
                if not wav_path.exists():
                    try:
                        data, sr = sf.read(io.BytesIO(audio_bytes))
                        sf.write(str(wav_path), data, sr)
                    except: continue
                    
                if wav_path.exists():
                    dur = get_audio_duration(str(wav_path))
                    if 0.5 < dur < 30:
                        samples.append({
                            "audio_path": str(wav_path),
                            "text": text,
                            "language": "uk",
                            "speaker_id": "euro_mp",
                            "duration": dur
                        })
                        total_dur += dur
                
                if total_dur > limit_sec: break
            if total_dur > limit_sec: break
        except Exception as e:
            print(f"Error {p_file.name}: {e}")

    print(f"   ✅ EuroSpeech samples: {len(samples)} ({total_dur/3600:.1f} hours)")
    return samples

def process_podcasts(data_dir: Path) -> List[Dict]:
    """Process Podcasts using 'audio_filepath'."""
    samples = []
    base_dir = data_dir / "uk-pods"
    if not base_dir.exists(): return samples
    
    print("🔍 Processing Podcasts...")
    clips_dir = base_dir / "clips"
    json_files = list(base_dir.glob("*.json"))
    
    for jf in json_files:
        count = 0
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        item = json.loads(line)
                        # !!! FIX: Correct column name
                        rel_path = item.get('audio_filepath')
                        text = item.get('text') # text is correct here
                        
                        if not rel_path or not text: continue
                        
                        # rel_path is like "clips/filename.wav", but clips_dir is already "uk-pods/clips"
                        # We need to handle this carefully
                        if rel_path.startswith("clips/"):
                            fname = os.path.basename(rel_path)
                            audio_path = clips_dir / fname
                        else:
                            audio_path = clips_dir / rel_path
                            
                        if audio_path.exists():
                            dur = get_audio_duration(str(audio_path))
                            if 0.5 < dur < 30:
                                samples.append({
                                    "audio_path": str(audio_path),
                                    "text": text,
                                    "language": "uk",
                                    "speaker_id": "uk_pod",
                                    "duration": dur
                                })
                                count += 1
                    except: continue
        except: pass
        print(f"   Parsed {count} from {jf.name}")
        
    print(f"   ✅ Podcasts total: {len(samples)}")
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/manifests")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    all_samples.extend(process_opentts(data_dir))
    all_samples.extend(process_podcasts(data_dir))
    all_samples.extend(process_eurospeech(data_dir, max_hours=300))
    
    if not all_samples:
        print("❌ CRITICAL: No samples found! Check paths.")
        sys.exit(1)
        
    random.seed(42)
    random.shuffle(all_samples)
    
    val_size = max(1, int(len(all_samples) * 0.01))
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print(f"\n📊 FINAL DATASET:")
    print(f"   Total: {len(all_samples)}")
    print(f"   Train: {len(train_samples)}")
    print(f"   Val:   {len(val_samples)}")
    
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Manifests saved.")

if __name__ == "__main__":
    main()