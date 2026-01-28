#!/usr/bin/env python3
"""
SUPERTONIC V2 TTS - DATA PREPARATION FOR STAGE 2 (Text-to-Latent)
===================================================================
Завантажує multi-speaker датасети для тренування voice cloning.

МЕТА: Максимум різних спікерів/тембрів для zero-shot voice cloning.
      Мова НЕ важлива - важлива різноманітність голосів!

✅ ПЕРЕВІРЕНІ ДАТАСЕТИ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset                    | Speakers | Hours  | Size    | Language
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LibriTTS-R clean-360       | 904      | 360h   | ~70 GB  | English
VCTK                       | 110      | 44h    | ~12 GB  | English
OpenTTS-UK                 | 5        | 24h    | ~1 GB   | Ukrainian
Common Voice UK (subset)   | 1000+    | ~80h   | ~15 GB  | Ukrainian
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL (recommended)        | 2000+    | ~500h  | ~100 GB |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRESETS:
  --minimal   : OpenTTS only (5 speakers, 24h) - testing
  --medium    : OpenTTS + VCTK (115 speakers, 68h) - good start
  --large     : + LibriTTS-R clean-100 (351 speakers, 168h)
  --full      : + LibriTTS-R clean-360 (1255 speakers, 500h+)

ВАЖЛИВО для Voice Cloning:
  - Мінімум 100+ різних спікерів
  - Рекомендовано 500-1000+ спікерів
  - Різноманітність: чоловіки, жінки, різні віки, акценти

Usage:
    python scripts/prepare_data_stage2.py --medium
    python scripts/prepare_data_stage2.py --large --output data_stage2
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tarfile
import shutil
import time
import multiprocessing

# Use most of available CPUs for parallel processing
NUM_WORKERS = min(64, multiprocessing.cpu_count())

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"   {desc}: {i}/{total} ({100*i/total:.0f}%)")
            yield item

try:
    import torch
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

OPENTTS_VOICES = [
    ("speech-uk/opentts-lada", "lada", 6962),
    ("speech-uk/opentts-tetiana", "tetiana", 5227),
    ("speech-uk/opentts-mykyta", "mykyta", 6436),
    ("speech-uk/opentts-oleksa", "oleksa", 3555),
    ("speech-uk/opentts-kateryna", "kateryna", 1803),
]


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     SUPERTONIC V2 - STAGE 2 DATA PREPARATION                             ║
║     Multi-Speaker Data for Voice Cloning                                 ║
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
    if errors:
        print("❌ Missing dependencies:")
        for e in errors:
            print(f"   {e}")
        sys.exit(1)
    print("✅ All dependencies available\n")


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def get_audio_duration(audio_path: str, target_sr: int = 22050) -> Optional[float]:
    """Get audio duration in seconds. Works with different torchaudio versions."""
    audio_path = str(audio_path)
    
    # Method 1: Try torchaudio.info (older versions)
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except (AttributeError, Exception):
        pass
    
    # Method 2: Load audio and calculate duration
    try:
        waveform, sr = torchaudio.load(audio_path)
        return waveform.shape[1] / sr
    except Exception as e:
        print(f"      ⚠️ Could not read {audio_path}: {e}")
        return None


def process_audio_file(args) -> Optional[Dict]:
    """Process single audio file."""
    audio_path, output_path, text, speaker_id, target_sr, source, language = args
    
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Save
        torchaudio.save(str(output_path), waveform, target_sr)
        
        duration = waveform.shape[1] / target_sr
        
        return {
            "audio_path": str(output_path),
            "text": text,
            "speaker_id": speaker_id,
            "duration": round(duration, 3),
            "sample_rate": target_sr,
            "source": source,
            "language": language,
        }
    except Exception as e:
        return None


# ============================================================================
# DATASET DOWNLOADERS  
# ============================================================================

def download_opentts(output_dir: Path, target_sr: int = 22050) -> Tuple[List[Dict], int]:
    """Download OpenTTS-UK (5 speakers)."""
    print("\n" + "="*70)
    print("📥 OpenTTS-UK (5 Ukrainian studio voices)")
    print("="*70)
    
    entries = []
    speakers = set()
    audio_dir = output_dir / "audio" / "opentts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for repo_id, voice_name, expected_rows in OPENTTS_VOICES:
        voice_dir = audio_dir / voice_name
        
        # Check if exists
        existing_files = list(voice_dir.glob("*.wav")) if voice_dir.exists() else []
        
        if len(existing_files) >= expected_rows * 0.95:
            print(f"   ✅ {voice_name}: Already downloaded ({len(existing_files)} files)")
            
            # Load from HuggingFace to get texts
            try:
                ds = load_dataset(repo_id, split="train")
                for idx, item in enumerate(ds):
                    audio_path = voice_dir / f"{voice_name}_{idx:06d}.wav"
                    if audio_path.exists():
                        text = item.get("text", item.get("sentence", ""))
                        duration = get_audio_duration(audio_path, target_sr)
                        if duration is None:
                            continue
                        
                        entries.append({
                            "audio_path": str(audio_path.relative_to(output_dir)),
                            "text": text,
                            "speaker_id": f"opentts_{voice_name}",
                            "duration": round(duration, 3),
                            "sample_rate": target_sr,
                            "source": "opentts",
                            "language": "uk",
                        })
                speakers.add(f"opentts_{voice_name}")
            except Exception as e:
                print(f"      ⚠️ Could not load texts: {e}")
            continue
        
        print(f"\n   📥 {voice_name} ({expected_rows} samples)")
        sys.stdout.flush()
        
        try:
            ds = load_dataset(repo_id, split="train")
            voice_dir.mkdir(exist_ok=True)
            
            # Process in parallel using ThreadPoolExecutor
            def process_item(args):
                idx, item = args
                try:
                    audio_array = item["audio"]["array"]
                    sr = item["audio"]["sampling_rate"]
                    text = item.get("text", item.get("sentence", ""))
                    
                    waveform = torch.tensor(audio_array).unsqueeze(0).float()
                    if sr != target_sr:
                        resampler = torchaudio.transforms.Resample(sr, target_sr)
                        waveform = resampler(waveform)
                    
                    audio_path = voice_dir / f"{voice_name}_{idx:06d}.wav"
                    torchaudio.save(str(audio_path), waveform, target_sr)
                    
                    duration = waveform.shape[1] / target_sr
                    
                    return {
                        "audio_path": str(audio_path.relative_to(output_dir)),
                        "text": text,
                        "speaker_id": f"opentts_{voice_name}",
                        "duration": round(duration, 3),
                        "sample_rate": target_sr,
                        "source": "opentts",
                        "language": "uk",
                    }
                except:
                    return None
            
            # Use ThreadPoolExecutor for I/O bound tasks
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                results = list(tqdm(
                    executor.map(process_item, enumerate(ds)),
                    desc=voice_name,
                    total=len(ds)
                ))
            
            # Filter None results and add to entries
            for r in results:
                if r is not None:
                    entries.append(r)
            
            speakers.add(f"opentts_{voice_name}")
            print(f"   ✅ {voice_name}: {len([e for e in entries if e['speaker_id'] == f'opentts_{voice_name}'])} samples")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return entries, len(speakers)


def download_vctk(output_dir: Path, target_sr: int = 22050) -> Tuple[List[Dict], int]:
    """
    Download VCTK dataset (110 English speakers).
    Uses HuggingFace: speechcolab/vctk
    """
    print("\n" + "="*70)
    print("📥 VCTK (110 English speakers)")
    print("   Size: ~12 GB | Duration: ~44 hours")
    print("="*70)
    
    entries = []
    speakers = set()
    audio_dir = output_dir / "audio" / "vctk"
    
    # Check if already downloaded
    if audio_dir.exists():
        existing = list(audio_dir.glob("**/*.wav"))
        if len(existing) > 40000:
            print(f"   ✅ VCTK: Already downloaded ({len(existing)} files)")
            # Count speakers
            speaker_dirs = [d for d in audio_dir.iterdir() if d.is_dir()]
            print(f"      Loading metadata...")
            
            for spk_dir in speaker_dirs:
                speaker_id = f"vctk_{spk_dir.name}"
                speakers.add(speaker_id)
                
                for wav_file in spk_dir.glob("*.wav"):
                    try:
                        duration = get_audio_duration(wav_file, target_sr)
                        if duration is None:
                            continue
                        
                        # VCTK text files
                        txt_file = wav_file.with_suffix('.txt')
                        text = ""
                        if txt_file.exists():
                            text = txt_file.read_text().strip()
                        
                        entries.append({
                            "audio_path": str(wav_file.relative_to(output_dir)),
                            "text": text,
                            "speaker_id": speaker_id,
                            "duration": round(duration, 3),
                            "sample_rate": target_sr,
                            "source": "vctk",
                            "language": "en",
                        })
                    except:
                        pass
            
            print(f"      ✅ Loaded {len(entries)} entries, {len(speakers)} speakers")
            return entries, len(speakers)
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("   📦 Downloading VCTK from HuggingFace (this may take a while)...")
        print("   ⏳ First download caches the dataset, subsequent runs are faster")
        print("   📊 VCTK: 110 speakers, ~44 hours, 48kHz audio")
        sys.stdout.flush()
        
        start_time = time.time()
        # VCTK: https://huggingface.co/datasets/CSTR-Edinburgh/vctk
        # No predefined splits, load all data
        ds = load_dataset(
            "CSTR-Edinburgh/vctk", 
            trust_remote_code=True,
            num_proc=4
        )
        # VCTK returns a DatasetDict, get the only split
        ds = ds[list(ds.keys())[0]]
        print(f"   ✅ Dataset loaded in {time.time() - start_time:.1f}s ({len(ds)} samples)")
        sys.stdout.flush()
        
        print("   🔄 Processing audio files with {0} workers...".format(NUM_WORKERS))
        
        # Prepare items for parallel processing
        items_to_process = list(enumerate(ds))
        processed_count = [0]  # Use list for mutable counter in closure
        lock = __import__('threading').Lock()
        
        def process_vctk_item(args):
            idx, item = args
            try:
                audio = item["audio"]
                speaker = item["speaker_id"]
                text = item.get("text", "")
                
                speaker_dir = audio_dir / speaker
                with lock:
                    speaker_dir.mkdir(exist_ok=True)
                    speakers.add(f"vctk_{speaker}")
                
                audio_array = audio["array"]
                sr = audio["sampling_rate"]
                
                waveform = torch.tensor(audio_array).unsqueeze(0).float()
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                
                filename = f"{speaker}_{idx:06d}.wav"
                audio_path = speaker_dir / filename
                torchaudio.save(str(audio_path), waveform, target_sr)
                
                duration = waveform.shape[1] / target_sr
                
                return {
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "text": text,
                    "speaker_id": f"vctk_{speaker}",
                    "duration": round(duration, 3),
                    "sample_rate": target_sr,
                    "source": "vctk",
                    "language": "en",
                }
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_vctk_item, items_to_process),
                desc="VCTK",
                total=len(items_to_process)
            ))
        
        for r in results:
            if r is not None:
                entries.append(r)
        
        print(f"   ✅ VCTK: {len(entries)} samples, {len(speakers)} speakers")
        
    except Exception as e:
        print(f"   ❌ Error downloading VCTK: {e}")
        import traceback
        traceback.print_exc()
    
    return entries, len(speakers)


def download_libritts(
    output_dir: Path, 
    target_sr: int = 22050,
    subset: str = "clean-100"  # "clean-100", "clean-360", "other-500"
) -> Tuple[List[Dict], int]:
    """
    Download LibriTTS-R dataset.
    
    Subsets:
      clean-100: 247 speakers, 100 hours
      clean-360: 904 speakers, 360 hours  
      other-500: 1166 speakers, 500 hours (noisier)
    """
    print("\n" + "="*70)
    print(f"📥 LibriTTS-R ({subset})")
    
    subset_info = {
        "clean-100": {"speakers": 247, "hours": 100, "size": "18 GB"},
        "clean-360": {"speakers": 904, "hours": 360, "size": "70 GB"},
        "other-500": {"speakers": 1166, "hours": 500, "size": "85 GB"},
    }
    
    info = subset_info.get(subset, {})
    print(f"   Speakers: ~{info.get('speakers', '?')} | Hours: ~{info.get('hours', '?')} | Size: ~{info.get('size', '?')}")
    print("="*70)
    
    entries = []
    speakers = set()
    
    subset_name = subset.replace("-", "_")  # clean-100 -> clean_100
    audio_dir = output_dir / "audio" / f"libritts_{subset_name}"
    
    # Check if already downloaded
    if audio_dir.exists():
        existing = list(audio_dir.glob("**/*.wav"))
        if len(existing) > 10000:
            print(f"   ✅ LibriTTS-R {subset}: Already downloaded ({len(existing)} files)")
            print(f"      Loading metadata...")
            
            for wav_file in existing:
                try:
                    # Speaker ID from path: libritts_clean_100/1234/5678/1234_5678_000001.wav
                    parts = wav_file.stem.split("_")
                    speaker = parts[0] if parts else "unknown"
                    speaker_id = f"libritts_{speaker}"
                    speakers.add(speaker_id)
                    
                    duration = get_audio_duration(wav_file, target_sr)
                    if duration is None:
                        continue
                    
                    # Load normalized text
                    txt_file = wav_file.with_suffix('.normalized.txt')
                    text = ""
                    if txt_file.exists():
                        text = txt_file.read_text().strip()
                    
                    entries.append({
                        "audio_path": str(wav_file.relative_to(output_dir)),
                        "text": text,
                        "speaker_id": speaker_id,
                        "duration": round(duration, 3),
                        "sample_rate": target_sr,
                        "source": "libritts",
                        "language": "en",
                    })
                except:
                    pass
            
            print(f"      ✅ Loaded {len(entries)} entries, {len(speakers)} speakers")
            return entries, len(speakers)
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"   📦 Downloading LibriTTS-R {subset} from HuggingFace...")
        print("   ⏳ This is a large dataset, download may take 10-30 minutes...")
        sys.stdout.flush()
        
        start_time = time.time()
        # LibriTTS-R: https://huggingface.co/datasets/mythicinfinity/libritts_r
        # Config: "clean", Splits: "train.clean.100", "train.clean.360"
        split_name = f"train.{subset.replace('-', '.')}"
        ds = load_dataset(
            "mythicinfinity/libritts_r",
            "clean",  # config name
            split=split_name,  # e.g., "train.clean.100"
            num_proc=4
        )
        print(f"   ✅ Dataset loaded in {time.time() - start_time:.1f}s ({len(ds)} samples)")
        sys.stdout.flush()
        
        print("   🔄 Processing audio files with {0} workers...".format(NUM_WORKERS))
        
        items_to_process = list(enumerate(ds))
        lock = __import__('threading').Lock()
        
        def process_libritts_item(args):
            idx, item = args
            try:
                audio = item["audio"]
                speaker = str(item["speaker_id"])
                text = item.get("text_normalized", item.get("text", ""))
                utterance_id = item.get("id", str(idx))
                
                speaker_dir = audio_dir / speaker
                with lock:
                    speaker_dir.mkdir(exist_ok=True)
                    speakers.add(f"libritts_{speaker}")
                
                audio_array = audio["array"]
                sr = audio["sampling_rate"]
                
                waveform = torch.tensor(audio_array).unsqueeze(0).float()
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                
                filename = f"{utterance_id}.wav"
                audio_path = speaker_dir / filename
                torchaudio.save(str(audio_path), waveform, target_sr)
                
                duration = waveform.shape[1] / target_sr
                
                return {
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "text": text,
                    "speaker_id": f"libritts_{speaker}",
                    "duration": round(duration, 3),
                    "sample_rate": target_sr,
                    "source": "libritts",
                    "language": "en",
                }
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_libritts_item, items_to_process),
                desc=f"LibriTTS {subset}",
                total=len(items_to_process)
            ))
        
        for r in results:
            if r is not None:
                entries.append(r)
        
        print(f"   ✅ LibriTTS-R {subset}: {len(entries)} samples, {len(speakers)} speakers")
        
    except Exception as e:
        print(f"   ❌ Error downloading LibriTTS-R: {e}")
        import traceback
        traceback.print_exc()
    
    return entries, len(speakers)


# ============================================================================
# MANIFEST CREATION
# ============================================================================

def create_manifests(
    entries: List[Dict],
    output_dir: Path,
    val_ratio: float = 0.05,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    min_text_length: int = 5,
) -> Tuple[int, int]:
    """Create train/val manifests for Stage 2."""
    print("\n" + "="*70)
    print("📝 Creating manifests for Stage 2...")
    print("="*70)
    
    # Debug: show sample entries
    if entries:
        print(f"\n   📊 Sample entry for debugging:")
        sample = entries[0]
        print(f"      duration: {sample.get('duration', 'N/A')}")
        print(f"      text length: {len(sample.get('text', ''))}")
        print(f"      speaker_id: {sample.get('speaker_id', 'N/A')}")
    
    # Filter
    filtered = []
    rejected_duration = 0
    rejected_text = 0
    for e in entries:
        dur = e.get("duration", 0)
        text = e.get("text", "")
        if dur is None or dur < min_duration or dur > max_duration:
            rejected_duration += 1
            continue
        if len(text) < min_text_length:
            rejected_text += 1
            continue
        filtered.append(e)
    
    print(f"\n   Total entries: {len(entries)}")
    print(f"   Rejected (duration): {rejected_duration}")
    print(f"   Rejected (text too short): {rejected_text}")
    print(f"   After filtering: {len(filtered)}")
    
    # Count speakers
    speakers = set(e["speaker_id"] for e in filtered)
    print(f"   Total speakers: {len(speakers)}")
    
    # Count by language
    languages = {}
    for e in filtered:
        lang = e.get("language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"   By language:")
    for lang, count in sorted(languages.items()):
        print(f"      {lang}: {count:,}")
    
    # Stratified split by speaker
    speaker_samples = {}
    for e in filtered:
        spk = e["speaker_id"]
        if spk not in speaker_samples:
            speaker_samples[spk] = []
        speaker_samples[spk].append(e)
    
    train_entries = []
    val_entries = []
    
    for spk, samples in speaker_samples.items():
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))
        val_entries.extend(samples[:n_val])
        train_entries.extend(samples[n_val:])
    
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    
    # Save manifests
    manifest_dir = output_dir / "manifests_stage2"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = manifest_dir / "train.json"
    val_path = manifest_dir / "val.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)
    
    # Stats
    train_hours = sum(e["duration"] for e in train_entries) / 3600
    val_hours = sum(e["duration"] for e in val_entries) / 3600
    train_speakers = len(set(e["speaker_id"] for e in train_entries))
    val_speakers = len(set(e["speaker_id"] for e in val_entries))
    
    print(f"\n   Train: {len(train_entries):,} samples ({train_hours:.1f}h, {train_speakers} speakers)")
    print(f"   Val:   {len(val_entries):,} samples ({val_hours:.1f}h, {val_speakers} speakers)")
    print(f"\n   Saved: {train_path}")
    print(f"   Saved: {val_path}")
    
    return len(train_entries), len(val_entries)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-speaker data for Stage 2 (Voice Cloning)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    preset_group = parser.add_mutually_exclusive_group(required=True)
    preset_group.add_argument("--minimal", action="store_true", 
                              help="OpenTTS only (5 speakers, ~24h)")
    preset_group.add_argument("--medium", action="store_true",
                              help="OpenTTS + VCTK (115 speakers, ~68h)")
    preset_group.add_argument("--large", action="store_true",
                              help="+ LibriTTS clean-100 (362 speakers, ~168h)")
    preset_group.add_argument("--full", action="store_true",
                              help="+ LibriTTS clean-360 (1266 speakers, ~500h)")
    
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--include-stage1", action="store_true",
                        help="Include Stage 1 manifest data (data/manifests/)")
    
    args = parser.parse_args()
    
    print_banner()
    check_dependencies()
    
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_entries = []
    total_speakers = 0
    
    # Include Stage 1 data if available and requested
    if args.include_stage1:
        stage1_manifest = output_dir / "manifests" / "train.json"
        if stage1_manifest.exists():
            print("📂 Including Stage 1 data...")
            with open(stage1_manifest, "r", encoding="utf-8") as f:
                stage1_data = json.load(f)
            for entry in stage1_data:
                entry["language"] = "uk"
                all_entries.append(entry)
            s1_speakers = len(set(e["speaker_id"] for e in stage1_data))
            print(f"   ✅ Added {len(stage1_data)} samples, {s1_speakers} speakers from Stage 1\n")
            total_speakers += s1_speakers
    
    # Always download OpenTTS
    entries, n_speakers = download_opentts(output_dir, args.sample_rate)
    all_entries.extend(entries)
    total_speakers += n_speakers
    
    # Medium+ includes VCTK
    if args.medium or args.large or args.full:
        entries, n_speakers = download_vctk(output_dir, args.sample_rate)
        all_entries.extend(entries)
        total_speakers += n_speakers
    
    # Large+ includes LibriTTS clean-100
    if args.large or args.full:
        entries, n_speakers = download_libritts(output_dir, args.sample_rate, "clean-100")
        all_entries.extend(entries)
        total_speakers += n_speakers
    
    # Full includes LibriTTS clean-360
    if args.full:
        entries, n_speakers = download_libritts(output_dir, args.sample_rate, "clean-360")
        all_entries.extend(entries)
        total_speakers += n_speakers
    
    # Create manifests
    if all_entries:
        train_count, val_count = create_manifests(all_entries, output_dir)
        
        actual_speakers = len(set(e["speaker_id"] for e in all_entries))
        
        print("\n" + "="*70)
        print("✅ STAGE 2 DATA PREPARATION COMPLETE!")
        print("="*70)
        print(f"   Total samples: {train_count + val_count:,}")
        print(f"   Total speakers: {actual_speakers}")
        print(f"   Train: {train_count:,}")
        print(f"   Val: {val_count:,}")
        print(f"\n   Manifests: {output_dir}/manifests_stage2/")
        print(f"\n   Next step:")
        print(f"   ./run_train_stage2.sh")
        print("="*70 + "\n")
    else:
        print("\n❌ No data downloaded!")
        sys.exit(1)


if __name__ == "__main__":
    main()
