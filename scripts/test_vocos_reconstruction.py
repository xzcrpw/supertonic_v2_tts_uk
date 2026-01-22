#!/usr/bin/env python3
"""
Test Vocos Adapter Reconstruction

Перевіряє якість реконструкції encoder + Vocos decoder.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.vocos_wrapper import create_vocos_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Test Vocos Reconstruction")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to encoder checkpoint")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--output", type=str, default="test_vocos_recon.wav",
                        help="Output audio path")
    parser.add_argument("--config", type=str, default="config/vocos_adapter.yaml",
                        help="Config file")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Create model
    print(f"\nLoading Vocos Adapter...")
    model = create_vocos_autoencoder(
        encoder_hidden_dim=config.encoder.hidden_dim,
        encoder_blocks=config.encoder.num_blocks,
        vocos_model=config.vocos.pretrained_model,
        freeze_vocos=True
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint["encoder"])
    print(f"✓ Loaded encoder from iteration {checkpoint.get('iteration', 'unknown')}")
    
    model.eval()
    
    # Load audio
    print(f"\nLoading audio: {args.audio}")
    audio, sr = torchaudio.load(args.audio)
    
    # Mix to mono if stereo
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample to 24kHz
    audio = torchaudio.functional.resample(audio, sr, 24000)
    
    # Take first 5 seconds
    max_samples = 24000 * 5
    if audio.size(1) > max_samples:
        audio = audio[:, :max_samples]
    
    audio = audio.to(device)
    
    print(f"Audio shape: {audio.shape}, duration: {audio.size(1) / 24000:.2f}s")
    
    # Reconstruct
    print("\nReconstructing...")
    with torch.no_grad():
        latent = model.encode(audio)
        reconstructed = model.decode(latent)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Save
    output_path = Path(args.output)
    torchaudio.save(str(output_path), reconstructed.cpu(), 24000)
    
    print(f"\n✓ Saved reconstruction: {output_path}")
    print(f"Duration: {reconstructed.size(1) / 24000:.2f}s")
    
    # Compare original vs reconstruction
    print("\n" + "="*60)
    print("Comparison:")
    print("="*60)
    print(f"Original:        {args.audio}")
    print(f"Reconstruction:  {output_path}")
    print(f"\nПослухай обидва файли і порівняй якість!")


if __name__ == "__main__":
    main()
