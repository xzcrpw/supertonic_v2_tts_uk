#!/usr/bin/env python3
"""
Dataset Downloader для Supertonic v2 TTS

Автоматично завантажує українські датасети для TTS.

Датасети:
1. M-AILABS Ukrainian (~20 год, ~3GB)
2. OpenTTS-UK (~multiple voices)
3. Common Voice Ukrainian (~80 год) - потребує ручного скачування
4. Voice of America (~390 год) - опційно
5. Ukrainian Broadcast (~300 год) - опційно

Usage:
    python scripts/download_datasets.py --minimal   # Тільки базові (~50GB)
    python scripts/download_datasets.py --full      # Все (~500GB)
"""

import os
import sys
import argparse
import subprocess
import tarfile
from pathlib import Path
from typing import List, Optional
import urllib.request
import shutil

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def download_with_progress(url: str, destination: Path, description: str = "Downloading"):
    """Download file with progress bar."""
    try:
        from tqdm import tqdm
        
        # Get file size
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=description) as pbar:
            def report_hook(block_num, block_size, total_size):
                pbar.update(block_size)
            
            urllib.request.urlretrieve(url, str(destination), reporthook=report_hook)
            
    except ImportError:
        print(f"Downloading {description}...")
        urllib.request.urlretrieve(url, str(destination))


def download_mailabs_ukrainian(data_dir: Path) -> bool:
    """Download M-AILABS Ukrainian dataset (~3GB, ~20 hours)."""
    print("\n" + "="*60)
    print("📥 Downloading M-AILABS Ukrainian")
    print("   Size: ~3GB | Duration: ~20 hours | Speakers: 2")
    print("="*60)
    
    output_dir = data_dir / "ukrainian"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_dir / "uk_UK.tgz"
    extract_dir = output_dir / "uk_UK"
    
    if extract_dir.exists():
        print("✓ Already downloaded")
        return True
    
    url = "http://www.caito.de/data/Training/stt_tts/uk_UK.tgz"
    
    try:
        download_with_progress(url, archive_path, "M-AILABS Ukrainian")
        
        print("Extracting...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        
        archive_path.unlink()
        print("✓ M-AILABS Ukrainian downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading M-AILABS: {e}")
        return False


def download_opentts_uk(data_dir: Path) -> bool:
    """Download OpenTTS-UK from HuggingFace."""
    print("\n" + "="*60)
    print("📥 Downloading OpenTTS-UK")
    print("   Voices: LADA, TETIANA, KATERYNA, MYKYTA, OLEKSA")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("✗ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    output_dir = data_dir / "ukrainian" / "opentts-uk"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print("✓ Already downloaded")
        return True
    
    try:
        snapshot_download(
            repo_id="Yehor/opentts-uk",
            repo_type="dataset",
            local_dir=str(output_dir),
            ignore_patterns=["*.md", "*.txt", ".git*"]
        )
        print("✓ OpenTTS-UK downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading OpenTTS-UK: {e}")
        return False


def download_voice_of_america(data_dir: Path) -> bool:
    """Download Voice of America dataset (~390 hours)."""
    print("\n" + "="*60)
    print("📥 Downloading Voice of America")
    print("   Size: ~50GB | Duration: ~390 hours")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("✗ huggingface_hub not installed")
        return False
    
    output_dir = data_dir / "ukrainian" / "voice-of-america"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print("✓ Already downloaded")
        return True
    
    try:
        snapshot_download(
            repo_id="speech-uk/voice-of-america",
            repo_type="dataset",
            local_dir=str(output_dir)
        )
        print("✓ Voice of America downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading Voice of America: {e}")
        return False


def download_broadcast_speech(data_dir: Path) -> bool:
    """Download Ukrainian Broadcast Speech (~300 hours)."""
    print("\n" + "="*60)
    print("📥 Downloading Ukrainian Broadcast Speech")
    print("   Size: ~40GB | Duration: ~300 hours")
    print("="*60)
    
    if not HF_AVAILABLE:
        print("✗ huggingface_hub not installed")
        return False
    
    output_dir = data_dir / "ukrainian" / "broadcast-speech-uk"
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print("✓ Already downloaded")
        return True
    
    try:
        snapshot_download(
            repo_id="Yehor/broadcast-speech-uk",
            repo_type="dataset",
            local_dir=str(output_dir)
        )
        print("✓ Broadcast Speech downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading Broadcast Speech: {e}")
        return False


def download_ljspeech(data_dir: Path) -> bool:
    """Download LJSpeech (English, ~24 hours)."""
    print("\n" + "="*60)
    print("📥 Downloading LJSpeech (English)")
    print("   Size: ~2.6GB | Duration: ~24 hours | Speaker: 1")
    print("="*60)
    
    output_dir = data_dir / "english"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_dir / "LJSpeech-1.1.tar.bz2"
    extract_dir = output_dir / "LJSpeech-1.1"
    
    if extract_dir.exists():
        print("✓ Already downloaded")
        return True
    
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    
    try:
        download_with_progress(url, archive_path, "LJSpeech")
        
        print("Extracting...")
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(output_dir)
        
        archive_path.unlink()
        print("✓ LJSpeech downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading LJSpeech: {e}")
        return False


def print_common_voice_instructions():
    """Print instructions for downloading Common Voice."""
    print("\n" + "="*60)
    print("📥 Common Voice Ukrainian (Manual Download Required)")
    print("   Size: ~10GB | Duration: ~80 hours | Speakers: ~1000+")
    print("="*60)
    print("""
Common Voice requires manual download due to licensing:

1. Go to: https://commonvoice.mozilla.org/uk/datasets
2. Register/Login
3. Download the Ukrainian dataset
4. Extract to: data/raw/ukrainian/common_voice_uk/

Expected structure:
data/raw/ukrainian/common_voice_uk/
├── clips/
│   ├── common_voice_uk_12345.mp3
│   └── ...
├── train.tsv
├── dev.tsv
├── test.tsv
└── validated.tsv
""")


def main():
    parser = argparse.ArgumentParser(description="Download TTS datasets")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--minimal", action="store_true", help="Download only minimal datasets (~50GB)")
    parser.add_argument("--full", action="store_true", help="Download all datasets (~500GB)")
    parser.add_argument("--english", action="store_true", help="Include English datasets")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("╔" + "═"*58 + "╗")
    print("║       Supertonic v2 TTS - Dataset Downloader            ║")
    print("╚" + "═"*58 + "╝")
    
    results = {}
    
    # Always download these (minimal set)
    results["M-AILABS Ukrainian"] = download_mailabs_ukrainian(data_dir)
    results["OpenTTS-UK"] = download_opentts_uk(data_dir)
    
    # Print Common Voice instructions
    print_common_voice_instructions()
    
    # Full mode: download large datasets
    if args.full:
        results["Voice of America"] = download_voice_of_america(data_dir)
        results["Broadcast Speech"] = download_broadcast_speech(data_dir)
    
    # English datasets
    if args.english or args.full:
        results["LJSpeech"] = download_ljspeech(data_dir)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\nDatasets saved to: {data_dir.absolute()}")
    
    print("\n" + "="*60)
    print("📋 Next Steps")
    print("="*60)
    print("""
1. Download Common Voice Ukrainian manually (see instructions above)

2. Prepare manifests:
   python scripts/prepare_manifest.py --data-dir data/raw

3. Start training:
   python train_autoencoder.py --config config/default.yaml
""")


if __name__ == "__main__":
    main()
