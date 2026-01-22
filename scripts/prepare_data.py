#!/usr/bin/env python3
"""
SUPERTONIC V2 TTS - DATA PREPARATION SCRIPT (VERIFIED JANUARY 2026)
====================================================================
Завантажує та готує датасети для тренування TTS моделі.

✅ ПЕРЕВІРЕНІ ДАТАСЕТИ (HuggingFace, активні на січень 2026):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset                        | Size     | Hours  | Rows   | License
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OpenTTS-UK (5 voices):
  speech-uk/opentts-lada       | 201 MB   | ~7h    | 6,962  | Apache 2.0
  speech-uk/opentts-tetiana    | 156 MB   | ~5h    | 5,227  | Apache 2.0  
  speech-uk/opentts-mykyta     | 195 MB   | ~6h    | 6,436  | Apache 2.0
  speech-uk/opentts-oleksa     | 363 MB   | ~4h    | 3,555  | Apache 2.0
  speech-uk/opentts-kateryna   | 123 MB   | ~2h    | 1,803  | CC BY-NC 4.0
  ─────────────────────────────────────────────────────────────────────────
  SUBTOTAL OpenTTS             | ~1 GB    | ~24h   | 23,983 |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UK-Pods                        | ~5 GB    | ~51h   | 34,231 | CC BY-NC 4.0
  taras-sereda/uk-pods
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Broadcast Speech UK            | 33.9 GB  | ~300h  | 136,736| Apache 2.0
  Yehor/broadcast-speech-uk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Voice of America UK            | ~15 GB   | ~391h  | ~300k  | CC BY 4.0
  speech-uk/voice-of-america
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL (all datasets)           | ~55 GB   | ~766h  | ~500k  |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ M-AILABS (caito.de) - НЕДОСТУПНИЙ (сервер не відповідає)

РЕКОМЕНДАЦІЇ:
  --quality    : OpenTTS only (~1 GB, 24h) - найкраща якість, мало даних
  --medium     : OpenTTS + UK-Pods (~6 GB, 75h) - гарний баланс
  --large      : + Broadcast (~40 GB, 375h) - багато даних
  --full       : + VoA (~55 GB, 766h) - максимум даних

Для vast.ai:
  --quality  : 20 GB disk
  --medium   : 30 GB disk  
  --large    : 75 GB disk
  --full     : 100 GB disk

Usage:
    python scripts/prepare_data.py --quality
    python scripts/prepare_data.py --medium
    python scripts/prepare_data.py --large
    python scripts/prepare_data.py --full
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

try:
    import torch
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


# ============================================================================
# VERIFIED DATASET DEFINITIONS (January 2026)
# ============================================================================

OPENTTS_VOICES = [
    # (repo_id, voice_name, size_mb, rows, license)
    ("speech-uk/opentts-lada", "lada", 201, 6962, "Apache 2.0"),
    ("speech-uk/opentts-tetiana", "tetiana", 156, 5227, "Apache 2.0"),
    ("speech-uk/opentts-mykyta", "mykyta", 195, 6436, "Apache 2.0"),
    ("speech-uk/opentts-oleksa", "oleksa", 363, 3555, "Apache 2.0"),
    ("speech-uk/opentts-kateryna", "kateryna", 123, 1803, "CC BY-NC 4.0"),
]

DATASETS_INFO = {
    "ukpods": {
        "repo_id": "taras-sereda/uk-pods",
        "size_gb": 5,
        "hours": 51,
        "rows": 34231,
        "license": "CC BY-NC 4.0",
        "format": "tar.gz",
    },
    "broadcast": {
        "repo_id": "Yehor/broadcast-speech-uk",
        "size_gb": 33.9,
        "hours": 300,
        "rows": 136736,
        "license": "Apache 2.0",
        "format": "parquet",
    },
    "voa": {
        "repo_id": "speech-uk/voice-of-america",
        "size_gb": 15,
        "hours": 391,
        "rows": 300000,
        "license": "CC BY 4.0",
        "format": "files",
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_banner():
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║         SUPERTONIC V2 TTS - DATA PREPARATION                             ║
║         Verified Datasets (January 2026)                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def check_dependencies():
    """Check required packages."""
    errors = []
    
    if not HF_AVAILABLE:
        errors.append("huggingface_hub and datasets not found")
        errors.append("  Install: pip install huggingface_hub datasets")
    
    if not TORCHAUDIO_AVAILABLE:
        errors.append("torchaudio not found")
        errors.append("  Install: pip install torchaudio")
    
    if errors:
        print("❌ Missing dependencies:")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)
    
    print("✅ All dependencies available\n")


def estimate_disk_space(preset: str) -> str:
    """Estimate required disk space."""
    estimates = {
        "quality": "20 GB",
        "medium": "30 GB", 
        "large": "75 GB",
        "full": "100 GB",
    }
    return estimates.get(preset, "50 GB")


def estimate_hours(preset: str) -> str:
    """Estimate total hours."""
    estimates = {
        "quality": "~24 hours",
        "medium": "~75 hours",
        "large": "~375 hours", 
        "full": "~766 hours",
    }
    return estimates.get(preset, "unknown")


# ============================================================================
# DATASET DOWNLOADERS
# ============================================================================

def check_existing_data(audio_dir: Path, source_name: str, expected_count: int) -> Tuple[bool, int]:
    """
    Check if data already exists.
    Returns (is_complete, existing_count)
    """
    if not audio_dir.exists():
        return False, 0
    
    wav_files = list(audio_dir.glob("**/*.wav"))
    count = len(wav_files)
    
    # Consider complete if we have at least 95% of expected
    is_complete = count >= expected_count * 0.95
    
    return is_complete, count


def load_existing_manifest(output_dir: Path, source: str) -> List[Dict]:
    """Load entries from existing manifest for a specific source."""
    manifest_path = output_dir / "manifests" / "train.json"
    if not manifest_path.exists():
        return []
    
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
        return [e for e in all_entries if e.get("source") == source]
    except:
        return []


def download_opentts(
    output_dir: Path,
    target_sr: int = 22050,
    voices: Optional[List[str]] = None
) -> List[Dict]:
    """
    Download OpenTTS-UK voices from HuggingFace.
    All 5 voices: ~1 GB, ~24 hours, ~24k samples
    """
    print("\n" + "="*70)
    print("📥 OpenTTS-UK (5 studio-quality voices)")
    print("   Size: ~1 GB | Duration: ~24 hours | Samples: ~24,000")
    print("="*70)
    
    if voices:
        selected = [v for v in OPENTTS_VOICES if v[1] in voices]
    else:
        selected = OPENTTS_VOICES
    
    print(f"\n   Voices: {[v[1] for v in selected]}")
    
    manifest_entries = []
    
    # Support both old structure (opentts/audio/) and new (audio/opentts/)
    old_audio_dir = output_dir / "opentts" / "audio"
    new_audio_dir = output_dir / "audio" / "opentts"
    
    # Use old structure if it exists, otherwise use new
    if old_audio_dir.exists():
        audio_dir = old_audio_dir
        print(f"   📂 Using existing structure: opentts/audio/")
    else:
        audio_dir = new_audio_dir
        print(f"   📂 Using structure: audio/opentts/")
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for repo_id, voice_name, size_mb, rows, license_type in selected:
        voice_dir = audio_dir / voice_name
        
        # Check if already downloaded
        is_complete, existing_count = check_existing_data(voice_dir, voice_name, rows)
        
        if is_complete:
            print(f"\n   ✅ {voice_name}: Already downloaded ({existing_count} files)")
            print(f"      Loading texts from HuggingFace...")
            
            # Load dataset to get texts (streaming to avoid re-downloading audio)
            try:
                ds = load_dataset(repo_id, split="train")
                text_map = {}
                for idx, item in enumerate(ds):
                    filename = f"{voice_name}_{idx:06d}.wav"
                    text = item.get("text", item.get("sentence", ""))
                    text_map[filename] = text
                
                # Match files with texts
                loaded_count = 0
                for wav_file in sorted(voice_dir.glob("*.wav")):
                    try:
                        info = torchaudio.info(str(wav_file))
                        duration = info.num_frames / info.sample_rate
                        text = text_map.get(wav_file.name, "")
                        
                        # Calculate relative path properly
                        try:
                            rel_path = wav_file.relative_to(output_dir)
                        except ValueError:
                            # If relative_to fails, construct path manually
                            rel_path = Path("opentts") / "audio" / voice_name / wav_file.name
                        
                        manifest_entries.append({
                            "audio_path": str(rel_path),
                            "text": text,
                            "speaker_id": voice_name,
                            "duration": round(duration, 3),
                            "sample_rate": target_sr,
                            "source": "opentts",
                        })
                        loaded_count += 1
                    except Exception as e:
                        pass
                print(f"      ✅ Loaded {loaded_count} entries with texts")
            except Exception as e:
                print(f"      ⚠️ Could not load texts: {e}")
                # Fallback - add entries without text
                for wav_file in sorted(voice_dir.glob("*.wav")):
                    try:
                        info = torchaudio.info(str(wav_file))
                        duration = info.num_frames / info.sample_rate
                        try:
                            rel_path = wav_file.relative_to(output_dir)
                        except ValueError:
                            rel_path = Path("opentts") / "audio" / voice_name / wav_file.name
                        manifest_entries.append({
                            "audio_path": str(rel_path),
                            "text": "",
                            "speaker_id": voice_name,
                            "duration": round(duration, 3),
                            "sample_rate": target_sr,
                            "source": "opentts",
                        })
                    except:
                        pass
            continue
        
        print(f"\n   📥 {voice_name} ({size_mb} MB, {rows} samples, {license_type})")
        print(f"      URL: https://huggingface.co/datasets/{repo_id}")
        
        try:
            ds = load_dataset(repo_id, split="train")
            
            voice_dir.mkdir(exist_ok=True)
            
            processed = 0
            for idx, item in enumerate(ds):
                audio_array = item["audio"]["array"]
                sr = item["audio"]["sampling_rate"]
                text = item.get("text", item.get("sentence", ""))
                
                # Save audio
                audio_filename = f"{voice_name}_{idx:06d}.wav"
                audio_path = voice_dir / audio_filename
                
                waveform = torch.tensor(audio_array).unsqueeze(0).float()
                
                # Resample if needed
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                
                torchaudio.save(str(audio_path), waveform, target_sr)
                
                duration = waveform.shape[1] / target_sr
                
                manifest_entries.append({
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "text": text,
                    "speaker_id": voice_name,
                    "duration": round(duration, 3),
                    "sample_rate": target_sr,
                    "source": "opentts",
                })
                
                processed += 1
                if processed % 1000 == 0:
                    print(f"      Processed {processed}/{rows}...")
            
            print(f"   ✅ {voice_name}: {processed} samples saved")
            
        except Exception as e:
            print(f"   ❌ Error downloading {voice_name}: {e}")
            continue
    
    return manifest_entries


def download_ukpods(
    output_dir: Path,
    target_sr: int = 22050
) -> List[Dict]:
    """
    Download UK-Pods from HuggingFace.
    Size: ~5 GB | Duration: ~51 hours | Samples: 34,231
    """
    print("\n" + "="*70)
    print("📥 UK-Pods (Ukrainian podcasts)")
    print("   Size: ~5 GB | Duration: ~51 hours | Samples: 34,231")
    print("   URL: https://huggingface.co/datasets/taras-sereda/uk-pods")
    print("="*70)
    
    manifest_entries = []
    audio_dir = output_dir / "audio" / "ukpods"
    
    # Check if already downloaded
    is_complete, existing_count = check_existing_data(audio_dir, "ukpods", 34231)
    
    if is_complete:
        print(f"\n   ✅ UK-Pods: Already downloaded ({existing_count} files)")
        for wav_file in audio_dir.glob("*.wav"):
            try:
                info = torchaudio.info(str(wav_file))
                duration = info.num_frames / info.sample_rate
                manifest_entries.append({
                    "audio_path": str(wav_file.relative_to(output_dir)),
                    "text": "",
                    "speaker_id": "ukpods",
                    "duration": round(duration, 3),
                    "sample_rate": target_sr,
                    "source": "ukpods",
                })
            except:
                pass
        return manifest_entries
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ds = load_dataset("taras-sereda/uk-pods", split="train")
        
        print(f"\n   Total samples: {len(ds)}")
        
        processed = 0
        for idx, item in enumerate(ds):
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item.get("text", item.get("sentence", ""))
            
            audio_filename = f"ukpods_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            
            waveform = torch.tensor(audio_array).unsqueeze(0).float()
            
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            torchaudio.save(str(audio_path), waveform, target_sr)
            
            duration = waveform.shape[1] / target_sr
            
            manifest_entries.append({
                "audio_path": str(audio_path.relative_to(output_dir)),
                "text": text,
                "speaker_id": "ukpods",
                "duration": round(duration, 3),
                "sample_rate": target_sr,
                "source": "ukpods",
            })
            
            processed += 1
            if processed % 5000 == 0:
                print(f"   Processed {processed}/{len(ds)}...")
        
        print(f"   ✅ UK-Pods: {processed} samples saved")
        
    except Exception as e:
        print(f"   ❌ Error downloading UK-Pods: {e}")
    
    return manifest_entries


def download_broadcast(
    output_dir: Path,
    target_sr: int = 22050
) -> List[Dict]:
    """
    Download Broadcast Speech UK from HuggingFace.
    Size: ~34 GB | Duration: ~300 hours | Samples: 136,736
    """
    print("\n" + "="*70)
    print("📥 Downloading Broadcast Speech UK")
    print("   Size: ~34 GB | Duration: ~300 hours | Samples: 136,736")
    print("   URL: https://huggingface.co/datasets/Yehor/broadcast-speech-uk")
    print("   ⚠️  This will take a while...")
    print("="*70)
    
    manifest_entries = []
    audio_dir = output_dir / "audio" / "broadcast"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use streaming for large dataset
        ds = load_dataset("Yehor/broadcast-speech-uk", split="train", streaming=True)
        
        processed = 0
        for idx, item in enumerate(ds):
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item.get("text", item.get("sentence", ""))
            
            audio_filename = f"broadcast_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            
            waveform = torch.tensor(audio_array).unsqueeze(0).float()
            
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            torchaudio.save(str(audio_path), waveform, target_sr)
            
            duration = waveform.shape[1] / target_sr
            
            manifest_entries.append({
                "audio_path": str(audio_path.relative_to(output_dir)),
                "text": text,
                "speaker_id": "broadcast",
                "duration": round(duration, 3),
                "sample_rate": target_sr,
                "source": "broadcast",
            })
            
            processed += 1
            if processed % 10000 == 0:
                print(f"   Processed {processed}...")
        
        print(f"   ✅ Broadcast: {processed} samples saved")
        
    except Exception as e:
        print(f"   ❌ Error downloading Broadcast: {e}")
    
    return manifest_entries


def download_voa(
    output_dir: Path,
    target_sr: int = 22050
) -> List[Dict]:
    """
    Download Voice of America UK from HuggingFace.
    Size: ~15 GB | Duration: ~391 hours
    """
    print("\n" + "="*70)
    print("📥 Downloading Voice of America UK")
    print("   Size: ~15 GB | Duration: ~391 hours")
    print("   URL: https://huggingface.co/datasets/speech-uk/voice-of-america")
    print("   ⚠️  This dataset uses custom format - downloading files...")
    print("="*70)
    
    manifest_entries = []
    
    try:
        voa_dir = output_dir / "voa_raw"
        
        print(f"\n   Downloading to: {voa_dir}")
        snapshot_download(
            repo_id="speech-uk/voice-of-america",
            repo_type="dataset",
            local_dir=str(voa_dir),
        )
        
        print(f"   ✅ VoA downloaded to {voa_dir}")
        print(f"   ⚠️  Manual processing required - check voa_raw/ folder")
        
    except Exception as e:
        print(f"   ❌ Error downloading VoA: {e}")
    
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
    """Create train/val manifest files."""
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
    
    # Stats
    train_duration = sum(e["duration"] for e in train_entries) / 3600
    val_duration = sum(e["duration"] for e in val_entries) / 3600
    
    # Source breakdown
    sources = {}
    for e in train_entries:
        src = e.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\n   Train: {train_count:,} samples ({train_duration:.1f} hours)")
    print(f"   Val:   {val_count:,} samples ({val_duration:.1f} hours)")
    print(f"\n   By source:")
    for src, count in sorted(sources.items()):
        print(f"      {src}: {count:,}")
    
    print(f"\n   Saved: {train_path}")
    print(f"   Saved: {val_path}")
    
    return train_count, val_count


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Ukrainian TTS datasets (verified January 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  --quality   OpenTTS only (~1 GB, 24h) - best quality, limited data
  --medium    + UK-Pods (~6 GB, 75h) - good balance
  --large     + Broadcast (~40 GB, 375h) - lots of data
  --full      + VoA (~55 GB, 766h) - maximum data

Examples:
  python scripts/prepare_data.py --quality
  python scripts/prepare_data.py --medium --sample-rate 22050
        """
    )
    
    # Presets
    preset_group = parser.add_mutually_exclusive_group(required=True)
    preset_group.add_argument("--quality", action="store_true", help="OpenTTS only (~1GB, 24h)")
    preset_group.add_argument("--medium", action="store_true", help="OpenTTS + UK-Pods (~6GB, 75h)")
    preset_group.add_argument("--large", action="store_true", help="+ Broadcast (~40GB, 375h)")
    preset_group.add_argument("--full", action="store_true", help="+ VoA (~55GB, 766h)")
    
    # Options
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--voices", type=str, nargs="+", help="Specific OpenTTS voices")
    
    args = parser.parse_args()
    
    # Determine preset
    if args.quality:
        preset = "quality"
    elif args.medium:
        preset = "medium"
    elif args.large:
        preset = "large"
    else:
        preset = "full"
    
    print_banner()
    print(f"   Preset: {preset}")
    print(f"   Output: {args.output}")
    print(f"   Sample rate: {args.sample_rate} Hz")
    print(f"   Estimated disk: {estimate_disk_space(preset)}")
    print(f"   Estimated hours: {estimate_hours(preset)}")
    print()
    
    check_dependencies()
    
    output_dir = Path(args.output).resolve()  # Use absolute path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_entries = []
    
    # Always download OpenTTS
    entries = download_opentts(output_dir, args.sample_rate, args.voices)
    all_entries.extend(entries)
    
    # Medium+ includes UK-Pods
    if preset in ["medium", "large", "full"]:
        entries = download_ukpods(output_dir, args.sample_rate)
        all_entries.extend(entries)
    
    # Large+ includes Broadcast
    if preset in ["large", "full"]:
        entries = download_broadcast(output_dir, args.sample_rate)
        all_entries.extend(entries)
    
    # Full includes VoA
    if preset == "full":
        entries = download_voa(output_dir, args.sample_rate)
        all_entries.extend(entries)
    
    # Create manifests
    if all_entries:
        train_count, val_count = create_manifests(all_entries, output_dir)
        
        print("\n" + "="*70)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*70)
        print(f"   Total samples: {train_count + val_count:,}")
        print(f"   Train: {train_count:,}")
        print(f"   Val: {val_count:,}")
        print(f"\n   Next step:")
        print(f"   ./scripts/train_22khz_optimal.sh")
        print("="*70 + "\n")
    else:
        print("\n❌ No data downloaded!")
        sys.exit(1)


if __name__ == "__main__":
    main()
