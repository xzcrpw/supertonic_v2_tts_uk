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
from supertonic.data.preprocessing import AudioProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load checkpoint
checkpoint_path = "checkpoints/autoencoder/checkpoint_80000.pt"
print(f"\nLoading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Get config
config = checkpoint.get("config", {})
print(f"Config: {config.get('model', {})}")

# Create model
model_cfg = config.get("model", {})
model = SpeechAutoencoder(
    in_channels=model_cfg.get("in_channels", 228),
    latent_dim=model_cfg.get("latent_dim", 24),
    encoder_channels=model_cfg.get("encoder_channels", [64, 128, 256, 512]),
    decoder_channels=model_cfg.get("decoder_channels", [512, 256, 128, 64]),
    num_encoder_layers=model_cfg.get("num_encoder_layers", 4),
    num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
).to(device)

# Load weights
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
elif "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # Try to find the right key
    print("Available keys:", checkpoint.keys())
    raise KeyError("Cannot find model weights in checkpoint")

model.eval()
print(f"✓ Model loaded from iteration {checkpoint.get('iteration', '?')}")

# Audio processor
audio_cfg = config.get("audio", {})
audio_processor = AudioProcessor(
    sample_rate=audio_cfg.get("sample_rate", 44100),
    n_mels=audio_cfg.get("n_mels", 228),
    hop_length=audio_cfg.get("hop_length", 512),
    win_length=audio_cfg.get("win_length", 2048),
    n_fft=audio_cfg.get("n_fft", 2048)
)

print(f"\nAudio config:")
print(f"  sample_rate: {audio_processor.sample_rate}")
print(f"  n_mels: {audio_processor.n_mels}")
print(f"  hop_length: {audio_processor.hop_length}")

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
    
    # Keep original sample rate (44.1kHz)
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
    
    # Limit to 10 sec
    audio = audio[:44100*10]
    
    print(f"    Audio shape: {audio.shape}, sr: {sr}")
    
    # Compute mel spectrogram
    mel = audio_processor.compute_mel(audio.unsqueeze(0))
    mel = mel.to(device)
    print(f"    Mel shape: {mel.shape}")
    print(f"    Mel range: [{mel.min():.2f}, {mel.max():.2f}]")
    
    # Encode & Decode
    with torch.no_grad():
        latent = model.encode(mel)
        print(f"    Latent shape: {latent.shape}")
        print(f"    Latent range: [{latent.min():.2f}, {latent.max():.2f}]")
        
        recon_mel = model.decode(latent)
        print(f"    Recon mel shape: {recon_mel.shape}")
        print(f"    Recon mel range: [{recon_mel.min():.2f}, {recon_mel.max():.2f}]")
    
    # Mel loss
    min_len = min(mel.shape[-1], recon_mel.shape[-1])
    mel_loss = torch.nn.functional.l1_loss(
        recon_mel[..., :min_len], 
        mel[..., :min_len]
    ).item()
    print(f"    Mel L1 loss: {mel_loss:.4f}")
    
    # Convert mel back to audio (Griffin-Lim or vocoder)
    recon_audio = audio_processor.mel_to_audio(recon_mel.squeeze(0).cpu())
    
    print(f"    Recon audio shape: {recon_audio.shape}")
    
    # Save
    orig_path = output_dir / f"{i+1}_original.wav"
    recon_path = output_dir / f"{i+1}_reconstructed.wav"
    mel_orig_path = output_dir / f"{i+1}_mel_original.pt"
    mel_recon_path = output_dir / f"{i+1}_mel_reconstructed.pt"
    
    torchaudio.save(str(orig_path), audio.unsqueeze(0), 44100)
    torchaudio.save(str(recon_path), recon_audio.unsqueeze(0), 44100)
    torch.save(mel.cpu(), str(mel_orig_path))
    torch.save(recon_mel.cpu(), str(mel_recon_path))
    
    print(f"    ✓ Saved")

print(f"\n✓ Outputs in: {output_dir}/")
print("\nLISTEN to the files and check:")
print("1. If mel looks similar but audio is metallic → vocoder/Griffin-Lim problem")
print("2. If mel is very different → encoder/decoder problem")
print("3. Check latent range - should be normalized around 0")
