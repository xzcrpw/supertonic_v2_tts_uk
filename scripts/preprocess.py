#!/usr/bin/env python3
"""
Preprocess Script
Генерує mel-спектрограми та зберігає їх на диск для прискорення навчання.
"""
import os
import sys
import argparse
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.data.preprocessing import AudioProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/a100_optimized.yaml")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    
    # Load manifest
    manifest_path = Path(args.manifest_dir) / "train.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path) as f:
        data = json.load(f)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Init processor
    processor = AudioProcessor(
        sample_rate=44100,
        n_mels=228,
        n_fft=2048,
        hop_length=512
    )
    
    print(f"🚀 Preprocessing {len(data)} files...")
    
    for item in tqdm(data):
        audio_path = item["audio_path"]
        file_id = Path(audio_path).stem
        save_path = output_dir / f"{file_id}.pt"
        
        if save_path.exists():
            continue
            
        try:
            # Load & Compute Mel
            audio = processor.load(audio_path)
            mel = processor.compute_mel(audio)
            
            # Save compressed
            torch.save(mel.clone(), save_path)
            
        except Exception as e:
            # print(f"Error processing {audio_path}: {e}")
            pass
            
    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    main()