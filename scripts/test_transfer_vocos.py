#!/usr/bin/env python3
"""
Test Transfer Vocos reconstruction quality.

Usage:
    python scripts/test_transfer_vocos.py --checkpoint checkpoints/transfer_vocos/checkpoint_400.pt
    python scripts/test_transfer_vocos.py --checkpoint checkpoints/transfer_vocos/checkpoint_400.pt --audio path/to/test.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.transfer_vocos import create_transfer_learning_adapter


def test_reconstruction(checkpoint_path: str, audio_path: str = None, output_dir: str = "test_outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get("config", {})
    pretrained_checkpoint = config.get("transfer", {}).get("pretrained_checkpoint", "checkpoints/autoencoder/checkpoint_80000.pt")
    vocos_model = config.get("vocos", {}).get("pretrained_model", "charactr/vocos-mel-24khz")
    
    # Create model
    print("Creating model...")
    model = create_transfer_learning_adapter(
        pretrained_checkpoint=pretrained_checkpoint,
        vocos_model=vocos_model,
        freeze_encoder=True,
        freeze_vocos=True,
        device=device
    )
    
    # Load adapter weights
    model.adapter.load_state_dict(checkpoint["adapter"])
    model.eval()
    
    iteration = checkpoint.get("iteration", "unknown")
    print(f"Loaded from iteration: {iteration}")
    
    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get test audio
    if audio_path:
        audio_files = [Path(audio_path)]
    else:
        # Use validation samples
        manifest_path = Path("data/manifests/val.json")
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                samples = json.load(f)[:5]  # First 5 samples
            audio_files = [Path(s["audio_path"]) for s in samples]
        else:
            print("No audio files found! Provide --audio path")
            return
    
    print(f"\nTesting {len(audio_files)} audio files...")
    
    for i, audio_file in enumerate(audio_files):
        if not audio_file.exists():
            print(f"  Skipping {audio_file} (not found)")
            continue
        
        print(f"\n[{i+1}] {audio_file.name}")
        
        # Load audio
        audio, sr = torchaudio.load(str(audio_file))
        if audio.dim() == 2:
            audio = audio.mean(dim=0)  # Mono
        
        # Resample to 24kHz
        if sr != 24000:
            audio = F.resample(audio, sr, 24000)
        
        # Limit length (10 sec max)
        max_len = 24000 * 10
        if len(audio) > max_len:
            audio = audio[:max_len]
        
        audio = audio.unsqueeze(0).to(device)
        
        # Reconstruct
        with torch.no_grad():
            recon_audio = model(audio)
        
        # Match lengths
        min_len = min(audio.shape[-1], recon_audio.shape[-1])
        audio = audio[..., :min_len]
        recon_audio = recon_audio[..., :min_len]
        
        # Calculate metrics
        l1_loss = torch.nn.functional.l1_loss(recon_audio, audio).item()
        mse_loss = torch.nn.functional.mse_loss(recon_audio, audio).item()
        
        print(f"  L1 Loss: {l1_loss:.4f}")
        print(f"  MSE Loss: {mse_loss:.6f}")
        
        # Save outputs
        orig_path = output_path / f"sample_{i+1}_original.wav"
        recon_path = output_path / f"sample_{i+1}_reconstructed.wav"
        
        torchaudio.save(str(orig_path), audio.cpu(), 24000)
        torchaudio.save(str(recon_path), recon_audio.cpu(), 24000)
        
        print(f"  Saved: {orig_path.name}, {recon_path.name}")
    
    print(f"\n✓ All outputs saved to: {output_path}/")
    print("\nListen and compare the original vs reconstructed files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to transfer_vocos checkpoint")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to test audio file (optional)")
    parser.add_argument("--output", type=str, default="test_outputs",
                        help="Output directory")
    
    args = parser.parse_args()
    test_reconstruction(args.checkpoint, args.audio, args.output)
