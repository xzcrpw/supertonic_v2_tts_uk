#!/usr/bin/env python3
"""
Test PURE Vocos reconstruction (without our adapter).
This tests if Vocos itself works correctly.
"""

import torch
import torchaudio
from pathlib import Path
from vocos import Vocos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load Vocos
print("Loading Vocos...")
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
vocos.eval()

# Find test audio
import json
manifest = Path("data/manifests/val.json")
with open(manifest) as f:
    samples = json.load(f)[:3]

output_dir = Path("test_outputs/pure_vocos")
output_dir.mkdir(parents=True, exist_ok=True)

print("\nTesting PURE Vocos reconstruction...")
print("="*60)

for i, sample in enumerate(samples):
    audio_path = Path(sample["audio_path"])
    if not audio_path.exists():
        continue
    
    print(f"\n[{i+1}] {audio_path.name}")
    
    # Load audio
    audio, sr = torchaudio.load(str(audio_path))
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    
    # Resample to 24kHz
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    
    # Limit to 10 sec
    audio = audio[:24000*10]
    audio = audio.unsqueeze(0).to(device)
    
    # Pure Vocos reconstruction
    with torch.no_grad():
        # Extract mel using Vocos's own feature extractor
        mel = vocos.feature_extractor(audio)
        print(f"    Mel shape: {mel.shape}")
        
        # Decode back to audio
        recon = vocos.decode(mel)
    
    # Match lengths
    min_len = min(audio.shape[-1], recon.shape[-1])
    audio = audio[..., :min_len]
    recon = recon[..., :min_len]
    
    # Metrics
    l1 = torch.nn.functional.l1_loss(recon, audio).item()
    print(f"    L1 Loss: {l1:.4f}")
    
    # Save
    torchaudio.save(str(output_dir / f"{i+1}_original.wav"), audio.cpu(), 24000)
    torchaudio.save(str(output_dir / f"{i+1}_vocos_recon.wav"), recon.cpu(), 24000)

print(f"\n✓ Saved to {output_dir}/")
print("\nListen to these files - if Vocos itself sounds bad, that's a different problem.")
print("If Vocos sounds GOOD here, then our adapter is the problem.")
