# scripts/check_latent_stats.py
import torch
import sys
from pathlib import Path
sys.path.insert(0, ".")
from supertonic.models.speech_autoencoder import LatentEncoder
from supertonic.data.preprocessing import AudioProcessor

def check_stats():
    device = "cuda"
    # Завантаж свій чекпоінт автоенкодера
    ckpt_path = "checkpoints/autoencoder/checkpoint_150000.pt"
    
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Init encoder
    encoder = LatentEncoder(input_dim=100, hidden_dim=512, output_dim=24, num_blocks=10).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    
    # Init processor
    processor = AudioProcessor(sample_rate=22050, n_mels=100)
    
    # Load sample audio (any wav file from your data)
    # Зміни шлях на реальний файл!
    audio_path = "data/audio/opentts/lada/lada_000001.wav" 
    audio = processor.load(audio_path).to(device)
    mel = processor.compute_mel(audio).unsqueeze(0)
    
    with torch.no_grad():
        latent = encoder(mel)
    
    print("\n📊 LATENT STATISTICS:")
    print(f"Mean: {latent.mean().item():.4f}")
    print(f"Std:  {latent.std().item():.4f}")
    print(f"Min:  {latent.min().item():.4f}")
    print(f"Max:  {latent.max().item():.4f}")
    
    if abs(latent.mean().item()) > 1.0 or latent.std().item() > 2.0:
        print("\n❌ CRITICAL: Latents are NOT normalized!")
        print("Flow Matching requires std ≈ 1.0. You must normalize latents during Stage 2 training.")
    else:
        print("\n✅ Latents look okay-ish (but check if they are centered).")

if __name__ == "__main__":
    check_stats()