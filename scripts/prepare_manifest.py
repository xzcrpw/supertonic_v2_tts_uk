#!/usr/bin/env python3
"""
Manifest Preparation Script для Supertonic v2 TTS

Створює JSON manifests з різних датасетів:
- M-AILABS Ukrainian
- OpenTTS-UK (Yehor/opentts-uk)
- Common Voice Ukrainian
- Voice of America
- Ukrainian Broadcast
- LJSpeech (English)

Output format:
[
    {
        "audio_path": "path/to/audio.wav",
        "text": "Транскрипція тексту",
        "language": "uk",
        "speaker_id": "speaker_name",
        "duration": 5.2
    },
    ...
]

Usage:
    python scripts/prepare_manifest.py --data-dir data/raw --output-dir data/manifests
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    import wave
    import contextlib

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        if TORCHAUDIO_AVAILABLE:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        else:
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return 0.0


def process_mailabs_ukrainian(data_dir: Path) -> List[Dict]:
    """
    Process M-AILABS Ukrainian dataset.
    
    Structure:
    uk_UK/
    ├── by_book/
    │   ├── female/
    │   │   └── sumska/
    │   │       └── svit_u_sto_rokiv/
    │   │           ├── wavs/
    │   │           └── metadata.csv
    │   └── male/
    │       └── shepel/
    │           └── kobzar/
    │               ├── wavs/
    │               └── metadata.csv
    """
    samples = []
    mailabs_dir = data_dir / "ukrainian" / "uk_UK"
    
    if not mailabs_dir.exists():
        print(f"M-AILABS not found at {mailabs_dir}")
        return samples
    
    print("Processing M-AILABS Ukrainian...")
    
    # Find all metadata files
    for metadata_path in mailabs_dir.rglob("metadata.csv"):
        wavs_dir = metadata_path.parent / "wavs"
        
        # Determine speaker from path
        parts = metadata_path.parts
        speaker_id = None
        for i, part in enumerate(parts):
            if part in ["female", "male"]:
                if i + 1 < len(parts):
                    speaker_id = f"mailabs_{parts[i+1]}"
                break
        
        if speaker_id is None:
            speaker_id = "mailabs_unknown"
        
        # Read metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 2:
                    file_id = parts[0]
                    text = parts[1] if len(parts) == 2 else parts[2]
                    
                    audio_path = wavs_dir / f"{file_id}.wav"
                    if audio_path.exists():
                        duration = get_audio_duration(str(audio_path))
                        
                        if 0.5 < duration < 30:  # Filter by duration
                            samples.append({
                                "audio_path": str(audio_path),
                                "text": text.strip(),
                                "language": "uk",
                                "speaker_id": speaker_id,
                                "duration": duration
                            })
    
    print(f"  Found {len(samples)} samples from M-AILABS")
    return samples


def process_opentts_uk(data_dir: Path) -> List[Dict]:
    """
    Process OpenTTS-UK dataset from HuggingFace.
    
    Voices: LADA, TETIANA, KATERYNA (female), MYKYTA, OLEKSA (male)
    """
    samples = []
    opentts_dir = data_dir / "ukrainian" / "opentts-uk"
    
    if not opentts_dir.exists():
        print(f"OpenTTS-UK not found at {opentts_dir}")
        return samples
    
    print("Processing OpenTTS-UK...")
    
    # Process each voice
    for voice_dir in opentts_dir.iterdir():
        if not voice_dir.is_dir():
            continue
        
        speaker_id = f"opentts_{voice_dir.name.lower()}"
        
        # Look for metadata/transcripts
        metadata_files = list(voice_dir.glob("*.txt")) + list(voice_dir.glob("*.csv"))
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 2:
                        file_id = parts[0].strip()
                        text = parts[-1].strip()
                        
                        # Find audio file
                        for ext in ['.wav', '.mp3', '.flac']:
                            audio_path = voice_dir / f"{file_id}{ext}"
                            if audio_path.exists():
                                duration = get_audio_duration(str(audio_path))
                                
                                if 0.5 < duration < 30:
                                    samples.append({
                                        "audio_path": str(audio_path),
                                        "text": text,
                                        "language": "uk",
                                        "speaker_id": speaker_id,
                                        "duration": duration
                                    })
                                break
    
    print(f"  Found {len(samples)} samples from OpenTTS-UK")
    return samples


def process_common_voice(data_dir: Path, language: str = "uk") -> List[Dict]:
    """
    Process Common Voice dataset.
    
    Expected structure:
    common_voice_uk/
    ├── clips/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
    """
    samples = []
    cv_dir = data_dir / "ukrainian" / "common_voice_uk"
    
    if not cv_dir.exists():
        print(f"Common Voice not found at {cv_dir}")
        return samples
    
    print("Processing Common Voice Ukrainian...")
    
    clips_dir = cv_dir / "clips"
    
    for tsv_file in ["train.tsv", "validated.tsv"]:
        tsv_path = cv_dir / tsv_file
        
        if not tsv_path.exists():
            continue
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            
            # Find column indices
            try:
                path_idx = header.index('path')
                sentence_idx = header.index('sentence')
                client_idx = header.index('client_id') if 'client_id' in header else None
            except ValueError:
                continue
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= max(path_idx, sentence_idx):
                    continue
                
                audio_filename = parts[path_idx]
                text = parts[sentence_idx]
                speaker_id = f"cv_{parts[client_idx][:8]}" if client_idx and len(parts) > client_idx else "cv_unknown"
                
                audio_path = clips_dir / audio_filename
                if not audio_path.exists():
                    # Try with .mp3 extension
                    audio_path = clips_dir / (audio_filename.replace('.mp3', '') + '.mp3')
                
                if audio_path.exists():
                    duration = get_audio_duration(str(audio_path))
                    
                    if 0.5 < duration < 30:
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text.strip(),
                            "language": "uk",
                            "speaker_id": speaker_id,
                            "duration": duration
                        })
    
    print(f"  Found {len(samples)} samples from Common Voice")
    return samples


def process_ljspeech(data_dir: Path) -> List[Dict]:
    """Process LJSpeech dataset (English)."""
    samples = []
    lj_dir = data_dir / "english" / "LJSpeech-1.1"
    
    if not lj_dir.exists():
        print(f"LJSpeech not found at {lj_dir}")
        return samples
    
    print("Processing LJSpeech...")
    
    metadata_path = lj_dir / "metadata.csv"
    wavs_dir = lj_dir / "wavs"
    
    if not metadata_path.exists():
        return samples
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                file_id = parts[0]
                text = parts[2]  # Normalized text
                
                audio_path = wavs_dir / f"{file_id}.wav"
                if audio_path.exists():
                    duration = get_audio_duration(str(audio_path))
                    
                    if 0.5 < duration < 30:
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text.strip(),
                            "language": "en",
                            "speaker_id": "ljspeech",
                            "duration": duration
                        })
    
    print(f"  Found {len(samples)} samples from LJSpeech")
    return samples


def process_hf_dataset(dataset_name: str, language: str) -> List[Dict]:
    """
    Process HuggingFace dataset directly.
    
    Supported:
    - speech-uk/voice-of-america
    - Yehor/broadcast-speech-uk
    """
    if not HF_DATASETS_AVAILABLE:
        print(f"huggingface datasets not available, skipping {dataset_name}")
        return []
    
    samples = []
    print(f"Processing HuggingFace dataset: {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        count = 0
        for item in tqdm(dataset, desc=f"Loading {dataset_name}"):
            audio = item.get("audio", {})
            text = item.get("text", item.get("sentence", item.get("transcription", "")))
            
            if not text:
                continue
            
            # For streaming, we need to save audio temporarily
            # In practice, you'd process this differently
            duration = len(audio.get("array", [])) / audio.get("sampling_rate", 16000)
            
            if 0.5 < duration < 30:
                samples.append({
                    "audio_path": f"{dataset_name}/{count}.wav",  # Placeholder
                    "text": text.strip(),
                    "language": language,
                    "speaker_id": f"{dataset_name.split('/')[-1]}_{count % 100}",
                    "duration": duration
                })
                count += 1
                
                if count >= 100000:  # Limit for memory
                    break
                    
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
    
    print(f"  Found {len(samples)} samples from {dataset_name}")
    return samples


def split_data(
    samples: List[Dict],
    val_ratio: float = 0.02,
    test_ratio: float = 0.01,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/val/test sets."""
    random.seed(seed)
    random.shuffle(samples)
    
    n = len(samples)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    
    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare TTS manifests")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", type=str, default="data/manifests", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.02, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.01, help="Test split ratio")
    parser.add_argument("--languages", type=str, nargs="+", default=["uk"], help="Languages to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # Process Ukrainian datasets
    if "uk" in args.languages:
        all_samples.extend(process_mailabs_ukrainian(data_dir))
        all_samples.extend(process_opentts_uk(data_dir))
        all_samples.extend(process_common_voice(data_dir))
    
    # Process English datasets
    if "en" in args.languages:
        all_samples.extend(process_ljspeech(data_dir))
    
    if not all_samples:
        print("No samples found! Please download datasets first.")
        print("\nDownload instructions:")
        print("1. M-AILABS: wget http://www.caito.de/data/Training/stt_tts/uk_UK.tgz")
        print("2. Common Voice: https://commonvoice.mozilla.org/uk/datasets")
        print("3. OpenTTS-UK: huggingface-cli download Yehor/opentts-uk")
        return
    
    print(f"\nTotal samples: {len(all_samples)}")
    
    # Calculate statistics
    total_duration = sum(s["duration"] for s in all_samples)
    print(f"Total duration: {total_duration/3600:.1f} hours")
    
    # Language breakdown
    lang_stats = {}
    for s in all_samples:
        lang = s["language"]
        lang_stats[lang] = lang_stats.get(lang, 0) + s["duration"]
    
    print("\nDuration by language:")
    for lang, dur in sorted(lang_stats.items()):
        print(f"  {lang}: {dur/3600:.1f} hours")
    
    # Speaker breakdown
    speaker_stats = {}
    for s in all_samples:
        speaker = s["speaker_id"]
        speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
    
    print(f"\nTotal speakers: {len(speaker_stats)}")
    
    # Split data
    train_samples, val_samples, test_samples = split_data(
        all_samples,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        seed=args.seed
    )
    
    print(f"\nSplit: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    # Save manifests
    with open(output_dir / "train_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "val_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "test_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    
    # Save full manifest
    with open(output_dir / "all_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nManifests saved to {output_dir}/")
    print("  - train_manifest.json")
    print("  - val_manifest.json")
    print("  - test_manifest.json")
    print("  - all_manifest.json")


if __name__ == "__main__":
    main()
