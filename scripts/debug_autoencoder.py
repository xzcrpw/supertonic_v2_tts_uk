#!/usr/bin/env python3
"""
Diagnose original Supertonic Autoencoder quality.
Figure out WHY reconstruction is "metallic".
"""

import torch
import torchaudio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.speech_autoencoder import SpeechAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load checkpoint
checkpoint_path = "checkpoints/autoencoder/checkpoint_80000.pt"
print(f"\nLoading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Get config
config = checkpoint.get("config", {})
model_cfg = config.get("model", {}).get("autoencoder", {})
audio_cfg = config.get("audio", {})

print(f"Model config: {model_cfg}")
print(f"Audio config: {audio_cfg}")

# Create model with correct config
# Use default num_blocks (10) to match default decoder_dilations
model = SpeechAutoencoder(
    sample_rate=audio_cfg.get("sample_rate", 44100),
    n_fft=audio_cfg.get("n_fft", 2048),
    hop_length=audio_cfg.get("hop_length", 512),
    n_mels=model_cfg.get("in_channels", 228),
    latent_dim=model_cfg.get("latent_dim", 32),
    hidden_dim=model_cfg.get("hidden_dims", [512])[-1] if isinstance(model_cfg.get("hidden_dims"), list) else 512,
    # Keep default num_blocks=10 to match decoder_dilations=[1,2,4,1,2,4,1,1,1,1]
).to(device)

# Load weights - checkpoint stores components separately
if "encoder" in checkpoint and "decoder" in checkpoint:
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    if "mpd" in checkpoint:
        model.mpd.load_state_dict(checkpoint["mpd"])
    if "mrd" in checkpoint:
        model.mrd.load_state_dict(checkpoint["mrd"])
elif "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
elif "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    print("Available keys:", checkpoint.keys())
    raise KeyError("Cannot find model weights in checkpoint")

model.eval()
print(f"✓ Model loaded from iteration {checkpoint.get('iteration', '?')}")
print(f"Latent dim: {model.latent_dim}")
print(f"Compressed dim: {model.compressed_dim}")

# Test files
manifest = Path("data/manifests/val.json")
with open(manifest) as f:
    samples = json.load(f)[:5]

output_dir = Path("test_outputs/autoencoder_debug")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("Testing Original Autoencoder")
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
    
    # Resample to 44.1kHz
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
    
    # Limit to 10 sec
    audio = audio[:44100*10]
    audio = audio.unsqueeze(0).to(device)
    
    print(f"    Audio shape: {audio.shape}")
    
    # Full reconstruction
    with torch.no_grad():
        # Encode to latent
        latent = model.encode(audio)
        print(f"    Latent shape: {latent.shape}")
        print(f"    Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
        print(f"    Latent std: {latent.std():.3f}")
        
        # Decode back to audio
        recon_audio = model.decode(latent)
        print(f"    Recon audio shape: {recon_audio.shape}")
    
    # Match lengths
    min_len = min(audio.shape[-1], recon_audio.shape[-1])
    audio_crop = audio[..., :min_len]
    recon_crop = recon_audio[..., :min_len]
    
    # Audio loss
    l1_loss = torch.nn.functional.l1_loss(recon_crop, audio_crop).item()
    print(f"    Audio L1 loss: {l1_loss:.4f}")
    
    # Save
    orig_path = output_dir / f"{i+1}_original.wav"
    recon_path = output_dir / f"{i+1}_reconstructed.wav"
    latent_path = output_dir / f"{i+1}_latent.pt"
    
    torchaudio.save(str(orig_path), audio_crop.cpu(), 44100)
    torchaudio.save(str(recon_path), recon_crop.cpu(), 44100)
    torch.save(latent.cpu(), str(latent_path))
    
    print(f"    ✓ Saved")

print(f"\n✓ Outputs in: {output_dir}/")
print("\nКЛЮЧОВІ ПОКАЗНИКИ:")
print("- Latent range: має бути приблизно [-3, 3]")
print("- Latent std: має бути приблизно 1.0")
print("- Audio L1: <0.1 = добре, >0.3 = погано")
