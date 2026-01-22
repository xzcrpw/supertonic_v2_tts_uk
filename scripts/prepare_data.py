#!/usr/bin/env python3
"""
SUPERTONIC V2 TTS - DATA PREPARATION SCRIPT
============================================
Завантажує та готує датасети для тренування TTS моделі.

ДАТАСЕТИ (перевірені, січень 2026):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset                    | Size    | Hours | Speakers | Sample Rate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OpenTTS-UK (HuggingFace)  | ~1 GB   | ~24h  | 5        | 48000 Hz
  ├─ speech-uk/opentts-lada      (201 MB, 7h, female)
  ├─ speech-uk/opentts-tetiana   (156 MB, 5h, female)
  ├─ speech-uk/opentts-mykyta    (195 MB, 6h, male)
  ├─ speech-uk/opentts-oleksa    (363 MB, 4h, male)
  └─ speech-uk/opentts-kateryna  (123 MB, 2h, female)

UK-Pods (HuggingFace)     | ~5 GB   | ~51h  | ~20      | 22050 Hz
  └─ taras-sereda/uk-pods

Common Voice 15 Ukrainian | ~3 GB   | ~40h  | ~1000+   | 48000 Hz
  └─ mozilla-foundation/common_voice_15_0 (subset: uk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

РЕКОМЕНДАЦІЯ для vast.ai:
  - Диск: 50 GB (датасети + checkpoints + samples)
  - Якщо тільки OpenTTS: 20 GB достатньо

Usage:
    python scripts/prepare_data.py --dataset opentts      # 1 GB, 24h, найкраща якість
    python scripts/prepare_data.py --dataset ukpods       # 5 GB, 51h, природне мовлення  
    python scripts/prepare_data.py --dataset all          # 9 GB, 115h, максимум даних
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

OPENTTS_VOICES = [
    # (repo_id, voice_name, size_mb, hours, gender)
    ("speech-uk/opentts-lada", "lada", 201, 7, "female"),
    ("speech-uk/opentts-tetiana", "tetiana", 156, 5, "female"),
    ("speech-uk/opentts-mykyta", "mykyta", 195, 6, "male"),
    ("speech-uk/opentts-oleksa", "oleksa", 363, 4, "male"),
    ("speech-uk/opentts-kateryna", "kateryna", 123, 2, "female"),
]

UKPODS_INFO = {
    "repo_id": "taras-sereda/uk-pods",
    "size_gb": 5,
    "hours": 51,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_dependencies():
    """Check required packages."""
    errors = []
    
    if not HF_AVAILABLE:
        errors.append("huggingface_hub and datasets not found. Install: pip install huggingface_hub datasets")
    
    if not TORCHAUDIO_AVAILABLE and not SOUNDFILE_AVAILABLE:
        errors.append("Audio library not found. Install: pip install torchaudio OR pip install soundfile")
    
    if errors:
        print("❌ Missing dependencies:")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)
    
    print("✅ All dependencies available")


def resample_audio(audio_path: Path, output_path: Path, target_sr: int = 22050) -> bool:
    """Resample audio file to target sample rate."""
    try:
        if TORCHAUDIO_AVAILABLE:
            import torch
            waveform, sr = torchaudio.load(audio_path)
            
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            torchaudio.save(output_path, waveform, target_sr)
            return True
            
        elif SOUNDFILE_AVAILABLE:
            import librosa
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            sf.write(output_path, audio, target_sr)
            return True
            
    except Exception as e:
        print(f"   ⚠️ Error processing {audio_path}: {e}")
        return False
    
    return False


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        if TORCHAUDIO_AVAILABLE:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        elif SOUNDFILE_AVAILABLE:
            info = sf.info(str(audio_path))
            return info.duration
    except:
        return 0.0
    return 0.0


# ============================================================================
# DATASET DOWNLOADERS
# ============================================================================

def download_opentts(
    output_dir: Path,
    voices: Optional[List[str]] = None,
    target_sr: int = 22050
) -> List[Dict]:
    """
    Download OpenTTS-UK from HuggingFace.
    
    Returns: List of manifest entries
    """
    print("\n" + "="*70)
    print("📥 Downloading OpenTTS-UK...")
    print(f"   Target: {output_dir}")
    print(f"   Sample rate: {target_sr} Hz")
    print("="*70)
    
    if voices is None:
        selected = OPENTTS_VOICES
    else:
        selected = [v for v in OPENTTS_VOICES if v[1] in voices]
    
    total_size = sum(v[2] for v in selected)
    total_hours = sum(v[3] for v in selected)
    print(f"\n   Voices: {[v[1] for v in selected]}")
    print(f"   Total size: ~{total_size} MB")
    print(f"   Total duration: ~{total_hours} hours")
    
    manifest_entries = []
    audio_dir = output_dir / "opentts" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for repo_id, voice_name, size_mb, hours, gender in selected:
        print(f"\n   Downloading {voice_name} ({size_mb} MB, {hours}h, {gender})...")
        
        try:
            # Load dataset from HuggingFace
            ds = load_dataset(repo_id, split="train")
            
            voice_dir = audio_dir / voice_name
            voice_dir.mkdir(exist_ok=True)
            
            for idx, item in enumerate(ds):
                # Get audio
                audio_array = item["audio"]["array"]
                sr = item["audio"]["sampling_rate"]
                text = item.get("text", item.get("sentence", ""))
                
                # Save audio
                audio_filename = f"{voice_name}_{idx:06d}.wav"
                audio_path = voice_dir / audio_filename
                
                # Resample if needed
                import torch
                import torchaudio
                
                waveform = torch.tensor(audio_array).unsqueeze(0).float()
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                
                torchaudio.save(audio_path, waveform, target_sr)
                
                duration = waveform.shape[1] / target_sr
                
                # Add to manifest
                manifest_entries.append({
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "text": text,
                    "speaker_id": voice_name,
                    "duration": round(duration, 3),
                    "sample_rate": target_sr,
                })
                
                if (idx + 1) % 1000 == 0:
                    print(f"      Processed {idx + 1} samples...")
            
            print(f"   ✅ {voice_name}: {len(ds)} samples")
            
        except Exception as e:
            print(f"   ❌ Error downloading {voice_name}: {e}")
            continue
    
    return manifest_entries


def download_ukpods(
    output_dir: Path,
    target_sr: int = 22050,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Download UK-Pods from HuggingFace.
    
    Returns: List of manifest entries
    """
    print("\n" + "="*70)
    print("📥 Downloading UK-Pods...")
    print(f"   Target: {output_dir}")
    print(f"   Sample rate: {target_sr} Hz")
    print("="*70)
    
    manifest_entries = []
    audio_dir = output_dir / "ukpods" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ds = load_dataset("taras-sereda/uk-pods", split="train")
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        print(f"   Total samples: {len(ds)}")
        
        for idx, item in enumerate(ds):
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item.get("text", item.get("sentence", ""))
            
            # Save audio
            audio_filename = f"ukpods_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            
            import torch
            import torchaudio
            
            waveform = torch.tensor(audio_array).unsqueeze(0).float()
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            torchaudio.save(audio_path, waveform, target_sr)
            
            duration = waveform.shape[1] / target_sr
            
            manifest_entries.append({
                "audio_path": str(audio_path.relative_to(output_dir)),
                "text": text,
                "speaker_id": "ukpods",
                "duration": round(duration, 3),
                "sample_rate": target_sr,
            })
            
            if (idx + 1) % 1000 == 0:
                print(f"   Processed {idx + 1} samples...")
        
        print(f"   ✅ UK-Pods: {len(manifest_entries)} samples")
        
    except Exception as e:
        print(f"   ❌ Error downloading UK-Pods: {e}")
    
    return manifest_entries


# ============================================================================
# MANIFEST CREATION
# ============================================================================

def create_manifests(
    entries: List[Dict],
    output_dir: Path,
    val_ratio: float = 0.05,
    min_duration: float = 0.5,
    max_duration: float = 15.0
) -> Tuple[int, int]:
    """
    Create train/val manifest files.
    
    Returns: (train_count, val_count)
    """
    print("\n" + "="*70)
    print("📝 Creating manifests...")
    print("="*70)
    
    # Filter by duration
    filtered = [
        e for e in entries 
        if min_duration <= e["duration"] <= max_duration
    ]
    
    print(f"   Total entries: {len(entries)}")
    print(f"   After filtering ({min_duration}s - {max_duration}s): {len(filtered)}")
    
    # Shuffle and split
    random.shuffle(filtered)
    
    val_count = int(len(filtered) * val_ratio)
    train_count = len(filtered) - val_count
    
    train_entries = filtered[:train_count]
    val_entries = filtered[train_count:]
    
    # Save manifests
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = manifest_dir / "train.json"
    val_path = manifest_dir / "val.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)
    
    # Calculate total duration
    train_duration = sum(e["duration"] for e in train_entries) / 3600
    val_duration = sum(e["duration"] for e in val_entries) / 3600
    
    print(f"\n   Train: {train_count} samples ({train_duration:.1f} hours)")
    print(f"   Val:   {val_count} samples ({val_duration:.1f} hours)")
    print(f"\n   Saved: {train_path}")
    print(f"   Saved: {val_path}")
    
    return train_count, val_count


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Supertonic v2 TTS")
    parser.add_argument(
        "--dataset",
        choices=["opentts", "ukpods", "all"],
        default="opentts",
        help="Which dataset(s) to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate"
    )
    parser.add_argument(
        "--voices",
        type=str,
        nargs="+",
        default=None,
        help="OpenTTS voices to download (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎵 SUPERTONIC V2 TTS - DATA PREPARATION")
    print("="*70)
    print(f"   Dataset: {args.dataset}")
    print(f"   Output: {args.output}")
    print(f"   Sample rate: {args.sample_rate} Hz")
    print("="*70)
    
    check_dependencies()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_entries = []
    
    # Download datasets
    if args.dataset in ["opentts", "all"]:
        entries = download_opentts(
            output_dir,
            voices=args.voices,
            target_sr=args.sample_rate
        )
        all_entries.extend(entries)
    
    if args.dataset in ["ukpods", "all"]:
        entries = download_ukpods(
            output_dir,
            target_sr=args.sample_rate,
            max_samples=args.max_samples
        )
        all_entries.extend(entries)
    
    # Create manifests
    if all_entries:
        train_count, val_count = create_manifests(all_entries, output_dir)
        
        print("\n" + "="*70)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*70)
        print(f"   Total samples: {train_count + val_count}")
        print(f"   Train: {train_count}")
        print(f"   Val: {val_count}")
        print(f"\n   Next step: Run training with:")
        print(f"   ./scripts/train_22khz_optimal.sh")
        print("="*70 + "\n")
    else:
        print("\n❌ No data downloaded!")
        sys.exit(1)


if __name__ == "__main__":
    main()
