#!/usr/bin/env python3
"""
Порівняння двох checkpoint'ів автоенкодера.
Генерує audio для одних і тих самих файлів і показує метрики.
"""

import torch
import torchaudio
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.speech_autoencoder import SpeechAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """Завантажує модель з checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", {})
    audio_cfg = config.get("audio", {})
    
    # Створюємо модель з дефолтними параметрами (latent_dim=24)
    model = SpeechAutoencoder(
        sample_rate=audio_cfg.get("sample_rate", 44100),
        n_fft=audio_cfg.get("n_fft", 2048),
        hop_length=audio_cfg.get("hop_length", 512),
        n_mels=228,
        latent_dim=24,
        hidden_dim=512,
    ).to(device)
    
    # Завантажуємо ваги
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    
    iteration = checkpoint.get("iteration", "?")
    return model, iteration


def analyze_frequency(audio, sr=44100):
    """Аналізує частотний спектр."""
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, power=2)
    spec = spec_transform(audio).squeeze().numpy()
    spec_db = 10 * np.log10(spec + 1e-10)
    
    freq_bins = np.fft.rfftfreq(2048, 1/sr)
    
    results = {}
    ranges = [
        ("low_0_500", 0, 500),
        ("mid_500_2000", 500, 2000),
        ("high_2000_5000", 2000, 5000),
        ("vhigh_5000_10000", 5000, 10000),
    ]
    
    for name, low, high in ranges:
        mask = (freq_bins >= low) & (freq_bins < high)
        results[name] = spec_db[mask].mean()
    
    return results


def compare_audio(original, recon1, recon2, sr=44100):
    """Порівнює два реконструйованих аудіо з оригіналом."""
    min_len = min(original.shape[-1], recon1.shape[-1], recon2.shape[-1])
    original = original[..., :min_len]
    recon1 = recon1[..., :min_len]
    recon2 = recon2[..., :min_len]
    
    # L1 loss
    l1_1 = torch.nn.functional.l1_loss(recon1, original).item()
    l1_2 = torch.nn.functional.l1_loss(recon2, original).item()
    
    # Frequency analysis
    orig_freq = analyze_frequency(original.cpu())
    recon1_freq = analyze_frequency(recon1.cpu())
    recon2_freq = analyze_frequency(recon2.cpu())
    
    freq_diff1 = {k: recon1_freq[k] - orig_freq[k] for k in orig_freq}
    freq_diff2 = {k: recon2_freq[k] - orig_freq[k] for k in orig_freq}
    
    return {
        "l1_1": l1_1,
        "l1_2": l1_2,
        "freq_diff1": freq_diff1,
        "freq_diff2": freq_diff2,
    }


def main():
    # Checkpoint paths
    ckpt1_path = "checkpoints/autoencoder/checkpoint_80000.pt"
    ckpt2_path = "checkpoints/autoencoder/checkpoint_90000.pt"
    
    print("="*70)
    print("ПОРІВНЯННЯ CHECKPOINT'ІВ")
    print("="*70)
    
    # Load models
    print(f"\nЗавантаження checkpoint 1: {ckpt1_path}")
    model1, iter1 = load_model(ckpt1_path)
    print(f"  → Iteration: {iter1}")
    
    print(f"\nЗавантаження checkpoint 2: {ckpt2_path}")
    model2, iter2 = load_model(ckpt2_path)
    print(f"  → Iteration: {iter2}")
    
    # Test files
    manifest = Path("data/manifests/val.json")
    with open(manifest) as f:
        samples = json.load(f)[:5]
    
    output_dir = Path("test_outputs/compare_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТИ")
    print("="*70)
    
    all_results = []
    
    for i, sample in enumerate(samples):
        audio_path = Path(sample["audio_path"])
        if not audio_path.exists():
            continue
        
        print(f"\n[{i+1}] {audio_path.name}")
        
        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)
        audio = audio[:44100*10].unsqueeze(0).to(device)
        
        # Reconstruct with both models
        with torch.no_grad():
            latent1 = model1.encode(audio)
            recon1 = model1.decode(latent1)
            
            latent2 = model2.encode(audio)
            recon2 = model2.decode(latent2)
        
        # Compare
        results = compare_audio(audio, recon1, recon2)
        all_results.append(results)
        
        print(f"    Audio L1 loss:")
        print(f"      Checkpoint {iter1}: {results['l1_1']:.4f}")
        print(f"      Checkpoint {iter2}: {results['l1_2']:.4f}")
        improvement = (results['l1_1'] - results['l1_2']) / results['l1_1'] * 100
        print(f"      {'📈 Покращення' if improvement > 0 else '📉 Погіршення'}: {abs(improvement):.1f}%")
        
        print(f"    Частотний баланс (різниця від оригіналу, dB):")
        print(f"      {'Діапазон':20} | {f'Ckpt {iter1}':>10} | {f'Ckpt {iter2}':>10} | Краще?")
        print(f"      {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-------")
        
        for key in results['freq_diff1']:
            d1 = results['freq_diff1'][key]
            d2 = results['freq_diff2'][key]
            better = "✅" if abs(d2) < abs(d1) else "❌" if abs(d2) > abs(d1) else "="
            name = key.replace("_", " ").replace("low", "Низькі").replace("mid", "Середні").replace("high", "Високі").replace("vhigh", "Дуже вис.")
            print(f"      {name:20} | {d1:>+10.2f} | {d2:>+10.2f} | {better}")
        
        # Save audio files
        min_len = min(audio.shape[-1], recon1.shape[-1], recon2.shape[-1])
        torchaudio.save(str(output_dir / f"{i+1}_original.wav"), audio[..., :min_len].cpu(), 44100)
        torchaudio.save(str(output_dir / f"{i+1}_ckpt{iter1}.wav"), recon1[..., :min_len].cpu(), 44100)
        torchaudio.save(str(output_dir / f"{i+1}_ckpt{iter2}.wav"), recon2[..., :min_len].cpu(), 44100)
    
    # Summary
    print("\n" + "="*70)
    print("ЗАГАЛЬНИЙ ПІДСУМОК")
    print("="*70)
    
    avg_l1_1 = np.mean([r['l1_1'] for r in all_results])
    avg_l1_2 = np.mean([r['l1_2'] for r in all_results])
    
    print(f"\nСередній Audio L1 loss:")
    print(f"  Checkpoint {iter1}: {avg_l1_1:.4f}")
    print(f"  Checkpoint {iter2}: {avg_l1_2:.4f}")
    
    improvement = (avg_l1_1 - avg_l1_2) / avg_l1_1 * 100
    if improvement > 0:
        print(f"  📈 Загальне покращення: {improvement:.1f}%")
    else:
        print(f"  📉 Загальне погіршення: {abs(improvement):.1f}%")
    
    # Average frequency differences
    print(f"\nСередня частотна різниця від оригіналу:")
    for key in all_results[0]['freq_diff1']:
        avg_d1 = np.mean([r['freq_diff1'][key] for r in all_results])
        avg_d2 = np.mean([r['freq_diff2'][key] for r in all_results])
        better = "✅" if abs(avg_d2) < abs(avg_d1) else "❌"
        name = key.replace("_", " ")
        print(f"  {name:20}: {avg_d1:+.2f} → {avg_d2:+.2f} dB  {better}")
    
    print(f"\n✓ Аудіо файли збережено в: {output_dir}/")
    print("\nПОСЛУХАЙ файли щоб оцінити різницю!")


if __name__ == "__main__":
    main()
