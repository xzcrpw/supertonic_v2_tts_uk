#!/usr/bin/env python3
"""
Аналіз "металічного" звуку в реконструкції.
Порівнює спектрограми оригіналу і реконструкції.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("test_outputs/autoencoder_debug")

# Знаходимо пари файлів
pairs = []
for i in range(1, 10):
    orig = output_dir / f"{i}_original.wav"
    recon = output_dir / f"{i}_reconstructed.wav"
    if orig.exists() and recon.exists():
        pairs.append((orig, recon, i))

if not pairs:
    print("❌ Не знайдено файлів. Спочатку запусти debug_autoencoder.py")
    exit(1)

print(f"Знайдено {len(pairs)} пар файлів")

# Аналізуємо першу пару детально
orig_path, recon_path, idx = pairs[0]

print(f"\nАналізую: {orig_path.name}")

# Завантажуємо аудіо
orig_audio, sr = torchaudio.load(str(orig_path))
recon_audio, _ = torchaudio.load(str(recon_path))

print(f"Sample rate: {sr}")
print(f"Original shape: {orig_audio.shape}")
print(f"Recon shape: {recon_audio.shape}")

# Обрізаємо до однакової довжини
min_len = min(orig_audio.shape[-1], recon_audio.shape[-1])
orig_audio = orig_audio[..., :min_len]
recon_audio = recon_audio[..., :min_len]

# Спектрограми
n_fft = 2048
hop_length = 512

spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=n_fft,
    hop_length=hop_length,
    power=2
)

orig_spec = spec_transform(orig_audio).squeeze().numpy()
recon_spec = spec_transform(recon_audio).squeeze().numpy()

# Логарифмуємо
orig_spec_db = 10 * np.log10(orig_spec + 1e-10)
recon_spec_db = 10 * np.log10(recon_spec + 1e-10)

# Різниця
diff_spec_db = recon_spec_db - orig_spec_db

# Частотні біни
freq_bins = np.fft.rfftfreq(n_fft, 1/sr)

# Статистика по частотних діапазонах
def analyze_freq_range(orig, recon, freq, low, high, name):
    mask = (freq >= low) & (freq < high)
    orig_mean = orig[mask].mean()
    recon_mean = recon[mask].mean()
    diff = recon_mean - orig_mean
    print(f"  {name:15} ({low:5}-{high:5} Hz): orig={orig_mean:6.1f}dB, recon={recon_mean:6.1f}dB, diff={diff:+5.1f}dB")
    return diff

print("\n" + "="*70)
print("ЧАСТОТНИЙ АНАЛІЗ (середня потужність по діапазонах)")
print("="*70)

diffs = []
diffs.append(analyze_freq_range(orig_spec_db, recon_spec_db, freq_bins, 0, 500, "Низькі"))
diffs.append(analyze_freq_range(orig_spec_db, recon_spec_db, freq_bins, 500, 2000, "Середні"))
diffs.append(analyze_freq_range(orig_spec_db, recon_spec_db, freq_bins, 2000, 5000, "Високі"))
diffs.append(analyze_freq_range(orig_spec_db, recon_spec_db, freq_bins, 5000, 10000, "Дуже високі"))
diffs.append(analyze_freq_range(orig_spec_db, recon_spec_db, freq_bins, 10000, 22050, "Ультрависокі"))

print("\n" + "="*70)
print("ДІАГНОСТИКА")
print("="*70)

if diffs[3] > 3 or diffs[4] > 3:
    print("⚠️  ПРОБЛЕМА: Надлишок енергії на високих частотах (>5kHz)")
    print("   → Це типова причина 'металічного' звуку")
    print("   → Можливі рішення:")
    print("      1. Знизити sample rate до 22050Hz")
    print("      2. Додати low-pass фільтр")
    print("      3. Більше тренувати discriminator")

if diffs[0] < -3:
    print("⚠️  ПРОБЛЕМА: Втрата низьких частот (<500Hz)")
    print("   → Голос звучить 'тонким'")

if abs(diffs[1]) > 3:
    print("⚠️  ПРОБЛЕМА: Середні частоти (500-2000Hz) не відповідають")
    print("   → Це основний діапазон мови")

# Зберігаємо візуалізацію
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Оригінал
im0 = axes[0].imshow(orig_spec_db, aspect='auto', origin='lower', 
                      extent=[0, min_len/sr, 0, sr/2], cmap='magma',
                      vmin=-80, vmax=0)
axes[0].set_title('Original Spectrogram')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_ylim(0, 10000)
plt.colorbar(im0, ax=axes[0], label='dB')

# Реконструкція
im1 = axes[1].imshow(recon_spec_db, aspect='auto', origin='lower',
                      extent=[0, min_len/sr, 0, sr/2], cmap='magma',
                      vmin=-80, vmax=0)
axes[1].set_title('Reconstructed Spectrogram')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_ylim(0, 10000)
plt.colorbar(im1, ax=axes[1], label='dB')

# Різниця
im2 = axes[2].imshow(diff_spec_db, aspect='auto', origin='lower',
                      extent=[0, min_len/sr, 0, sr/2], cmap='RdBu_r',
                      vmin=-20, vmax=20)
axes[2].set_title('Difference (Recon - Original): RED = too loud, BLUE = too quiet')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_ylim(0, 10000)
plt.colorbar(im2, ax=axes[2], label='dB difference')

plt.tight_layout()
plt.savefig(str(output_dir / 'spectral_analysis.png'), dpi=150)
print(f"\n✓ Збережено: {output_dir}/spectral_analysis.png")

# Додатковий аналіз: pitch/formants
print("\n" + "="*70)
print("РЕКОМЕНДАЦІЇ")
print("="*70)
print("""
Якщо 'метал' в основному на високих частотах:
  → Тренувати на 22050Hz замість 44100Hz
  → Або додати MultiResolution STFT loss з фокусом на низькі частоти

Якщо проблема в якості даних:
  → Відфільтрувати тільки чисті записи (OpenTTS, подкасти)
  → Прибрати європарламент або preprocessing (denoise)

Якщо g_loss=8-9 не зменшується:
  → Можливо discriminator занадто сильний
  → Спробувати зменшити вагу adversarial loss
""")
