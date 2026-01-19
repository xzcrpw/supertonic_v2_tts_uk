#!/usr/bin/env python3
"""
Dataset Downloader для Supertonic v2 TTS

Автоматично завантажує українські датасети для TTS.

✅ ПЕРЕВІРЕНІ ДАТАСЕТИ (січень 2026):

1. OpenTTS-UK Individual Voices (HuggingFace) - ~1 GB, ~24 години, 5 голосів
   - speech-uk/opentts-lada (201 MB, ~7h)
   - speech-uk/opentts-tetiana (156 MB, ~5h)  
   - speech-uk/opentts-mykyta (195 MB, ~6h)
   - speech-uk/opentts-oleksa (363 MB, ~4h)
   - speech-uk/opentts-kateryna (123 MB, ~2h)

2. EuroSpeech Ukrainian (HuggingFace) - ~50 GB, ~1,200 години
   - disco-eth/EuroSpeech (subset "ukraine")
   - Парламентське мовлення, висока якість

3. Ukrainian Podcasts (HuggingFace) - ~5 GB, ~51 година
   - taras-sereda/uk-pods
   - Природне мовлення, різні голоси

4. OpenTTS-UK Combined (HuggingFace) - ~12 MB metadata, ~24k семплів
   - Yehor/opentts-uk
   - Всі 5 голосів в одному датасеті

Usage:
    python scripts/download_datasets.py --quality      # Тільки OpenTTS (~1 GB)
    python scripts/download_datasets.py --medium       # OpenTTS + Podcasts (~6 GB)
    python scripts/download_datasets.py --full         # Все включаючи EuroSpeech (~60 GB)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import json

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# VERIFIED DATASET SOURCES (January 2026)
# ============================================================================

OPENTTS_VOICES = [
    # (repo_id, voice_name, description, size_mb, hours)
    ("speech-uk/opentts-lada", "lada", "Female, studio quality", 201, 7),
    ("speech-uk/opentts-tetiana", "tetiana", "Female, studio quality", 156, 5),
    ("speech-uk/opentts-mykyta", "mykyta", "Male, studio quality", 195, 6),
    ("speech-uk/opentts-oleksa", "oleksa", "Male, studio quality", 363, 4),
    ("speech-uk/opentts-kateryna", "kateryna", "Female, studio quality", 123, 2),
]

EUROSPEECH_INFO = {
    "repo_id": "disco-eth/EuroSpeech",
    "subset": "ukraine",
    "description": "Ukrainian Parliament recordings, high quality",
    "size_gb": 50,
    "hours": 1200,
    "url": "https://huggingface.co/datasets/disco-eth/EuroSpeech",
}

UK_PODS_INFO = {
    "repo_id": "taras-sereda/uk-pods",
    "description": "Ukrainian podcasts, natural speech",
    "size_gb": 5,
    "hours": 51,
    "url": "https://huggingface.co/datasets/taras-sereda/uk-pods",
}

OPENTTS_COMBINED = {
    "repo_id": "Yehor/opentts-uk",
    "description": "All 5 OpenTTS voices combined",
    "samples": 24000,
    "url": "https://huggingface.co/datasets/Yehor/opentts-uk",
}


def print_header(title: str, size: str = "", duration: str = ""):
    """Print section header."""
    print("\n" + "=" * 70)
    info = f"   {size}" if size else ""
    if duration:
        info += f" | {duration}"
    print(f"📥 {title}{info}")
    print("=" * 70)


def check_hf_available():
    """Check if HuggingFace hub is available."""
    if not HF_AVAILABLE:
        print("❌ huggingface_hub not installed!")
        print("   Install with: pip install huggingface_hub datasets")
        return False
    return True


def download_opentts_individual(data_dir: Path, voices: List[str] = None) -> Tuple[int, int]:
    """
    Download individual OpenTTS-UK voices from HuggingFace.
    
    Returns: (success_count, total_count)
    """
    print_header(
        "OpenTTS-UK Individual Voices",
        size="~1 GB total",
        duration="~24 hours, 5 speakers"
    )
    
    if not check_hf_available():
        return 0, len(OPENTTS_VOICES)
    
    output_dir = data_dir / "opentts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter voices if specified
    if voices:
        selected = [v for v in OPENTTS_VOICES if v[1] in voices]
    else:
        selected = OPENTTS_VOICES
    
    print(f"\nVoices to download: {[v[1] for v in selected]}")
    print(f"URLs:")
    for repo_id, name, desc, size_mb, hours in selected:
        print(f"  • https://huggingface.co/datasets/{repo_id}")
    
    success_count = 0
    
    for repo_id, voice_name, description, size_mb, hours in selected:
        voice_dir = output_dir / voice_name
        
        # Check if already downloaded
        if voice_dir.exists() and list(voice_dir.glob("**/*.wav")) or list(voice_dir.glob("**/*.parquet")):
            print(f"\n✓ {voice_name} already downloaded ({description})")
            success_count += 1
            continue
        
        try:
            print(f"\n⬇️  Downloading {voice_name} (~{size_mb} MB, ~{hours}h)...")
            print(f"   Source: https://huggingface.co/datasets/{repo_id}")
            
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(voice_dir),
                ignore_patterns=["*.md", ".git*", "*.gitattributes"]
            )
            
            print(f"✓ {voice_name} downloaded successfully")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to download {voice_name}: {e}")
    
    print(f"\n📊 Downloaded {success_count}/{len(selected)} voices")
    return success_count, len(selected)


def download_uk_pods(data_dir: Path) -> bool:
    """
    Download Ukrainian Podcasts dataset.
    
    Source: https://huggingface.co/datasets/taras-sereda/uk-pods
    Size: ~5 GB
    Duration: ~51 hours
    """
    print_header(
        "Ukrainian Podcasts (uk-pods)",
        size="~5 GB",
        duration="~51 hours"
    )
    
    print(f"\nURL: {UK_PODS_INFO['url']}")
    print(f"Description: {UK_PODS_INFO['description']}")
    
    if not check_hf_available():
        return False
    
    output_dir = data_dir / "uk-pods"
    
    # Check if already downloaded
    if output_dir.exists() and (output_dir / "clips.tar.gz").exists():
        print("\n✓ Already downloaded")
        return True
    
    try:
        print("\n⬇️  Downloading Ukrainian Podcasts...")
        
        snapshot_download(
            repo_id=UK_PODS_INFO["repo_id"],
            repo_type="dataset",
            local_dir=str(output_dir)
        )
        
        # Extract clips if tar exists
        clips_tar = output_dir / "clips.tar.gz"
        if clips_tar.exists():
            print("📦 Extracting clips.tar.gz...")
            import tarfile
            with tarfile.open(clips_tar, 'r:gz') as tar:
                tar.extractall(output_dir)
            print("✓ Extracted successfully")
        
        print("✓ Ukrainian Podcasts downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_eurospeech_uk(data_dir: Path, max_hours: int = None) -> bool:
    """
    Download EuroSpeech Ukrainian subset.
    
    Source: https://huggingface.co/datasets/disco-eth/EuroSpeech
    Subset: ukraine
    Size: ~50 GB (full), can limit with max_hours
    Duration: ~1,200 hours (CER < 20%)
    
    This is parliamentary speech - very clean, formal register.
    """
    print_header(
        "EuroSpeech Ukrainian (Parliament)",
        size="~50 GB" if not max_hours else f"~{max_hours // 24} GB (limited)",
        duration=f"~1,200 hours" if not max_hours else f"~{max_hours} hours"
    )
    
    print(f"\nURL: {EUROSPEECH_INFO['url']}")
    print(f"Subset: {EUROSPEECH_INFO['subset']}")
    print(f"Description: {EUROSPEECH_INFO['description']}")
    print(f"Quality: CER < 20% (high quality alignments)")
    
    if not check_hf_available():
        return False
    
    output_dir = data_dir / "eurospeech-uk"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already has data
    existing_files = list(output_dir.glob("**/*.parquet")) + list(output_dir.glob("**/*.wav"))
    if existing_files:
        print(f"\n✓ Already has {len(existing_files)} files downloaded")
        return True
    
    try:
        print("\n⬇️  Loading EuroSpeech Ukrainian dataset...")
        print("   This may take a while for large dataset...")
        
        # Load streaming to avoid memory issues
        dataset = load_dataset(
            EUROSPEECH_INFO["repo_id"],
            name=EUROSPEECH_INFO["subset"],
            split="train",
            streaming=True
        )
        
        # Save samples to disk
        manifest = []
        samples_dir = output_dir / "wavs"
        samples_dir.mkdir(exist_ok=True)
        
        total_duration = 0
        max_duration = (max_hours * 3600) if max_hours else float('inf')
        
        print("📝 Downloading and saving samples...")
        
        iterator = iter(dataset)
        count = 0
        
        if TQDM_AVAILABLE:
            pbar = tqdm(desc="Samples", unit=" samples")
        
        while total_duration < max_duration:
            try:
                sample = next(iterator)
            except StopIteration:
                break
            
            audio = sample.get("audio", {})
            text = sample.get("text", sample.get("transcript", ""))
            
            if not audio or not text:
                continue
            
            # Get audio array and sample rate
            audio_array = audio.get("array")
            sr = audio.get("sampling_rate", 16000)
            
            if audio_array is None:
                continue
            
            # Calculate duration
            duration = len(audio_array) / sr
            
            # Save audio file
            filename = f"euro_{count:08d}.wav"
            filepath = samples_dir / filename
            
            try:
                import soundfile as sf
                sf.write(str(filepath), audio_array, sr)
            except ImportError:
                # Fallback to scipy
                from scipy.io import wavfile
                import numpy as np
                wavfile.write(str(filepath), sr, (audio_array * 32767).astype(np.int16))
            
            manifest.append({
                "audio_filepath": str(filepath),
                "text": text,
                "duration": duration,
            })
            
            total_duration += duration
            count += 1
            
            if TQDM_AVAILABLE:
                pbar.update(1)
                pbar.set_postfix(hours=f"{total_duration/3600:.1f}")
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in manifest:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Downloaded {count} samples ({total_duration/3600:.1f} hours)")
        print(f"   Manifest: {manifest_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Download manually with streaming:")
        print("""
from datasets import load_dataset

dataset = load_dataset(
    "disco-eth/EuroSpeech", 
    name="ukraine",
    split="train",
    streaming=True
)

for sample in dataset:
    audio = sample["audio"]
    text = sample["text"]
    # Process...
""")
        return False


def download_opentts_combined(data_dir: Path) -> bool:
    """
    Download combined OpenTTS-UK dataset (all 5 voices).
    
    Source: https://huggingface.co/datasets/Yehor/opentts-uk
    This is metadata-only; actual audio is in individual voice repos.
    """
    print_header(
        "OpenTTS-UK Combined Dataset",
        size="~12 MB",
        duration="~24k samples, 5 voices"
    )
    
    print(f"\nURL: {OPENTTS_COMBINED['url']}")
    print(f"Description: {OPENTTS_COMBINED['description']}")
    
    if not check_hf_available():
        return False
    
    output_dir = data_dir / "opentts-combined"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print("\n✓ Already downloaded")
        return True
    
    try:
        print("\n⬇️  Downloading OpenTTS-UK combined...")
        
        snapshot_download(
            repo_id=OPENTTS_COMBINED["repo_id"],
            repo_type="dataset",
            local_dir=str(output_dir),
            ignore_patterns=["*.md", ".git*"]
        )
        
        print("✓ OpenTTS-UK combined downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def print_dataset_urls():
    """Print all verified dataset URLs."""
    print("\n" + "=" * 70)
    print("🔗 VERIFIED DATASET URLS (January 2026)")
    print("=" * 70)
    
    print("\n📦 OpenTTS-UK Individual Voices (~1 GB total, ~24 hours):")
    for repo_id, name, desc, size_mb, hours in OPENTTS_VOICES:
        print(f"   • {name}: https://huggingface.co/datasets/{repo_id}")
        print(f"     Size: {size_mb} MB | Duration: ~{hours}h | {desc}")
    
    print(f"\n📦 OpenTTS-UK Combined (metadata):")
    print(f"   • https://huggingface.co/datasets/{OPENTTS_COMBINED['repo_id']}")
    
    print(f"\n🎙️ Ukrainian Podcasts (~5 GB, ~51 hours):")
    print(f"   • https://huggingface.co/datasets/{UK_PODS_INFO['repo_id']}")
    
    print(f"\n🏛️ EuroSpeech Ukrainian (~50 GB, ~1,200 hours):")
    print(f"   • https://huggingface.co/datasets/{EUROSPEECH_INFO['repo_id']}")
    print(f"     Subset: {EUROSPEECH_INFO['subset']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Ukrainian TTS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_datasets.py --quality
      Download only OpenTTS voices (~1 GB, ~24 hours, best for TTS)
      
  python scripts/download_datasets.py --medium
      Download OpenTTS + Podcasts (~6 GB, ~75 hours)
      
  python scripts/download_datasets.py --full
      Download everything including EuroSpeech (~60 GB, ~1,300 hours)
      
  python scripts/download_datasets.py --full --eurospeech-hours 100
      Download OpenTTS + Podcasts + 100 hours of EuroSpeech (~15 GB)
      
  python scripts/download_datasets.py --urls
      Just print all dataset URLs without downloading
"""
    )
    
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--quality", action="store_true",
        help="Download only OpenTTS voices (~1 GB, best quality for TTS)"
    )
    parser.add_argument(
        "--medium", action="store_true",
        help="Download OpenTTS + Podcasts (~6 GB)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Download everything (~60 GB)"
    )
    parser.add_argument(
        "--eurospeech-hours", type=int, default=None,
        help="Limit EuroSpeech download to N hours (default: all ~1200 hours)"
    )
    parser.add_argument(
        "--voices", type=str, nargs="+",
        choices=["lada", "tetiana", "mykyta", "oleksa", "kateryna"],
        help="Download only specific OpenTTS voices"
    )
    parser.add_argument(
        "--urls", action="store_true",
        help="Just print dataset URLs without downloading"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("╔" + "═" * 68 + "╗")
    print("║         Supertonic v2 TTS - Ukrainian Dataset Downloader         ║")
    print("║                    Verified Sources (Jan 2026)                   ║")
    print("╚" + "═" * 68 + "╝")
    
    # Just print URLs
    if args.urls:
        print_dataset_urls()
        return
    
    # Default to quality if nothing specified
    if not any([args.quality, args.medium, args.full]):
        print("\n⚠️  No mode specified, using --quality (OpenTTS only)")
        args.quality = True
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Print what we're going to download
    print("\n📋 Download Plan:")
    if args.quality:
        print("   • OpenTTS Individual Voices (~1 GB, ~24 hours)")
    elif args.medium:
        print("   • OpenTTS Individual Voices (~1 GB, ~24 hours)")
        print("   • Ukrainian Podcasts (~5 GB, ~51 hours)")
    elif args.full:
        print("   • OpenTTS Individual Voices (~1 GB, ~24 hours)")
        print("   • Ukrainian Podcasts (~5 GB, ~51 hours)")
        hours_str = f"~{args.eurospeech_hours}h" if args.eurospeech_hours else "~1,200 hours"
        print(f"   • EuroSpeech Ukrainian ({hours_str})")
    
    print(f"\n📁 Output directory: {data_dir.absolute()}")
    
    # Download OpenTTS voices (always)
    success, total = download_opentts_individual(data_dir, voices=args.voices)
    results["OpenTTS Voices"] = success == total
    
    # Download podcasts (medium/full)
    if args.medium or args.full:
        results["Ukrainian Podcasts"] = download_uk_pods(data_dir)
    
    # Download EuroSpeech (full only)
    if args.full:
        results["EuroSpeech Ukrainian"] = download_eurospeech_uk(
            data_dir, 
            max_hours=args.eurospeech_hours
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Download Summary")
    print("=" * 70)
    
    for name, success in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {name}")
    
    # Calculate approximate size
    total_size = 1.0  # OpenTTS always ~1 GB
    if args.medium or args.full:
        total_size += 5.0  # Podcasts
    if args.full:
        if args.eurospeech_hours:
            total_size += (args.eurospeech_hours / 24)  # Rough estimate
        else:
            total_size += 50.0  # Full EuroSpeech
    
    print(f"\n💾 Approximate total size: ~{total_size:.0f} GB")
    print(f"📁 Data saved to: {data_dir.absolute()}")
    
    # Print URLs for reference
    print_dataset_urls()
    
    # Next steps
    print("\n" + "=" * 70)
    print("📋 Next Steps")
    print("=" * 70)
    print("""
1. Prepare training manifests:
   python scripts/prepare_manifest.py --data-dir data/raw

2. Start training on H100:
   ./scripts/train_h100.sh

3. Or train individual components:
   python train_autoencoder.py --config config/h100_optimized.yaml
   python train_tts.py --config config/h100_optimized.yaml
   python train_duration.py --config config/h100_optimized.yaml
""")


if __name__ == "__main__":
    main()
