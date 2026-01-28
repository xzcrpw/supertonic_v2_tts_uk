import torch
import sys
import os
from pathlib import Path

# Додаємо корінь проекту в шлях
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supertonic.models.speech_autoencoder import LatentEncoder
from supertonic.data.preprocessing import AudioProcessor

def strip_ddp_prefix(state_dict):
    """Видаляє префікс 'module.' з ключів чекпоінта."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def check_stats():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Шлях до чекпоінта
    ckpt_path = "checkpoints/autoencoder/checkpoint_150000.pt"
    
    # Шлях до тестового аудіо (вкажіть існуючий файл!)
    # Спробуємо знайти будь-який wav файл у data/
    audio_files = list(Path("data").rglob("*.wav"))
    if not audio_files:
        print("❌ Не знайдено жодного .wav файлу в папці data/")
        return
    audio_path = str(audio_files[0])
    print(f"🎵 Using audio: {audio_path}")
    
    print(f"📦 Loading {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"❌ Чекпоінт не знайдено: {ckpt_path}")
        return

    # Ініціалізація енкодера
    # Параметри мають співпадати з config/22khz_optimal.yaml
    encoder = LatentEncoder(
        input_dim=100,      # n_mels
        hidden_dim=512, 
        output_dim=24, 
        num_blocks=10,
        kernel_size=7
    ).to(device)
    
    # Завантаження ваг з фіксом DDP
    encoder_state = strip_ddp_prefix(ckpt["encoder"])
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    
    # Обробка аудіо
    processor = AudioProcessor(
        sample_rate=22050, 
        n_mels=100, 
        n_fft=1024, 
        hop_length=256
    )
    
    audio = processor.load(audio_path)
    # Обріжемо, якщо дуже довге, щоб не забити пам'ять
    if audio.shape[-1] > 22050 * 10:
        audio = audio[..., :22050 * 10]
        
    mel = processor.compute_mel(audio).unsqueeze(0).to(device)
    
    print("🔄 Encoding...")
    with torch.no_grad():
        latent = encoder(mel)
    
    print("\n📊 LATENT STATISTICS:")
    mean = latent.mean().item()
    std = latent.std().item()
    min_val = latent.min().item()
    max_val = latent.max().item()
    
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    print(f"  Min:  {min_val:.4f}")
    print(f"  Max:  {max_val:.4f}")
    
    print("\n🧐 VERDICT:")
    if abs(mean) > 1.0:
        print("⚠️  Mean is shifted (should be close to 0).")
    
    if std > 3.0 or std < 0.3:
        print(f"❌ CRITICAL: Std is {std:.4f}! Flow Matching expects Std ≈ 1.0.")
        print("   Ви повинні додати нормалізацію латентів у train_text_to_latent.py!")
        print(f"   Використовуйте: latent = (latent - {mean:.4f}) / {std:.4f}")
    else:
        print("✅ Latent stats look acceptable for training.")

if __name__ == "__main__":
    check_stats()