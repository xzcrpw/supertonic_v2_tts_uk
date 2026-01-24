#!/usr/bin/env python3
"""
ДЕТАЛЬНА ДІАГНОСТИКА РЕКОНСТРУКЦІЇ AUTOENCODER

Аналізує кожен етап pipeline:
1. Вхідне аудіо
2. Mel spectrogram
3. Encoder output (latent)
4. Decoder intermediate states
5. iSTFT magnitude/phase
6. Вихідне аудіо (до і після tanh)

Запуск:
    python scripts/diagnose_reconstruction.py --checkpoint checkpoints/autoencoder/checkpoint_2500.pt --audio data/opentts/audio/lada/lada_000001.wav
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder, ISTFTHead
from supertonic.data.preprocessing import AudioProcessor


def load_checkpoint_for_diagnosis(checkpoint_path: str, device: str = "cuda"):
    """Завантажує чекпоінт і створює моделі."""
    print(f"\n{'='*70}")
    print(f"ЗАВАНТАЖЕННЯ ЧЕКПОІНТА")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    
    # Detect params from weights
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    
    n_mels = encoder_state["input_conv.weight"].shape[1]
    n_fft = decoder_state["istft_head.window"].shape[0]
    
    # Guess hop_length
    if n_fft == 1024:
        hop_length = 256
        sample_rate = 22050
    else:
        hop_length = 512
        sample_rate = 44100
    
    print(f"  n_fft: {n_fft}")
    print(f"  hop_length: {hop_length}")
    print(f"  n_mels: {n_mels}")
    print(f"  sample_rate: {sample_rate}")
    
    # Create models
    encoder = LatentEncoder(
        input_dim=n_mels,
        hidden_dim=512,
        output_dim=24,
        num_blocks=10,
        kernel_size=7,
    ).to(device)
    
    decoder = LatentDecoder(
        input_dim=24,
        hidden_dim=512,
        num_blocks=10,
        kernel_size=7,
        dilations=[1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
        n_fft=n_fft,
        hop_length=hop_length,
        causal=True,
    ).to(device)
    
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    
    encoder.eval()
    decoder.eval()
    
    audio_processor = AudioProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    return encoder, decoder, audio_processor, {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "sample_rate": sample_rate
    }


def analyze_tensor(name: str, tensor: torch.Tensor, detailed: bool = True):
    """Аналізує тензор і виводить статистику."""
    t = tensor.detach().float()
    
    stats = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "abs_max": t.abs().max().item(),
        "num_zeros": (t == 0).sum().item(),
        "num_nan": torch.isnan(t).sum().item(),
        "num_inf": torch.isinf(t).sum().item(),
    }
    
    # Check for clipping (values at boundaries)
    if t.min() >= -1 and t.max() <= 1:
        num_clipped_low = (t <= -0.99).sum().item()
        num_clipped_high = (t >= 0.99).sum().item()
        stats["clipped_low"] = num_clipped_low
        stats["clipped_high"] = num_clipped_high
        stats["clipped_percent"] = 100 * (num_clipped_low + num_clipped_high) / t.numel()
    
    print(f"\n  📊 {name}:")
    print(f"     Shape: {stats['shape']}")
    print(f"     Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"     Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"     Abs Max: {stats['abs_max']:.6f}")
    
    if stats['num_nan'] > 0:
        print(f"     ⚠️  NaN values: {stats['num_nan']}")
    if stats['num_inf'] > 0:
        print(f"     ⚠️  Inf values: {stats['num_inf']}")
    if "clipped_percent" in stats and stats["clipped_percent"] > 1:
        print(f"     ⚠️  Clipped: {stats['clipped_percent']:.2f}% ({stats['clipped_low']} low, {stats['clipped_high']} high)")
    
    if detailed:
        # Percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        pct_values = [torch.quantile(t.flatten().float(), p/100).item() for p in percentiles]
        print(f"     Percentiles: {dict(zip(percentiles, [f'{v:.4f}' for v in pct_values]))}")
    
    return stats


def diagnose_decoder_internals(decoder, latent, device):
    """Діагностує внутрішні стани decoder крок за кроком."""
    print(f"\n{'='*70}")
    print("ДІАГНОСТИКА DECODER (крок за кроком)")
    print(f"{'='*70}")
    
    with torch.no_grad():
        # Step 1: Input projection
        x = decoder.input_conv(latent)
        analyze_tensor("После input_conv", x)
        
        x = decoder.input_norm(x)
        analyze_tensor("После input_norm (BatchNorm)", x)
        
        # Step 2: ConvNeXt blocks
        x = decoder.convnext(x)
        analyze_tensor("После ConvNeXt stack", x)
        
        # Step 3: Check which head type we have
        if hasattr(decoder, 'head'):
            # WaveNeXt head (new architecture) - expects [B, C, T]
            print(f"\n  --- WaveNeXt Head Details ---")
            
            head = decoder.head
            
            # BatchNorm (expects [B, C, T])
            x_norm = head.norm(x)
            analyze_tensor("After head.norm (BatchNorm)", x_norm)
            
            # Conv1d → head_dim
            x_conv = head.conv(x_norm)
            analyze_tensor("After head.conv (→ head_dim)", x_conv)
            
            # PReLU
            x_act = head.act(x_conv)
            analyze_tensor("After PReLU", x_act)
            
            # Transpose for Linear: [B, head_dim, T] → [B, T, head_dim]
            x_t = x_act.transpose(1, 2)
            
            # Linear → hop_length
            x_fc = head.fc(x_t)
            analyze_tensor("After head.fc (waveform frames)", x_fc)
            
            # Reshape to waveform
            batch_size, num_frames, _ = x_fc.shape
            audio_raw = x_fc.reshape(batch_size, num_frames * head.hop_length)
            analyze_tensor("Audio BEFORE tanh", audio_raw)
            
            audio_final = torch.tanh(audio_raw)
            analyze_tensor("Audio AFTER tanh", audio_final)
            
            return audio_raw, audio_final, None, None
            
        elif hasattr(decoder, 'istft_head'):
            # Legacy iSTFT head
            # Need to transpose for iSTFT head which expects [B, T, hidden]
            x = x.transpose(1, 2)
            print(f"\n  --- iSTFT Head Details (LEGACY) ---")
            
            istft = decoder.istft_head
            
            # Magnitude projection (before exp)
            mag_raw = istft.mag_proj(x)
            analyze_tensor("Magnitude (raw, before exp)", mag_raw)
            
            # Magnitude after exp
            mag = mag_raw.exp()
            analyze_tensor("Magnitude (after exp)", mag)
            
            # Phase projection
            phase_raw = istft.phase_proj(x)
            analyze_tensor("Phase (raw)", phase_raw)
            
            phase = torch.tanh(phase_raw) * 3.14159
            analyze_tensor("Phase (after tanh * π)", phase)
            
            # Complex spectrum
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)
            analyze_tensor("STFT Real part", real)
            analyze_tensor("STFT Imag part", imag)
            
            spec = torch.complex(real, imag)
            spec = spec.transpose(1, 2)
            
            # iSTFT
            audio_raw = torch.istft(
                spec,
                n_fft=istft.n_fft,
                hop_length=istft.hop_length,
                win_length=istft.n_fft,
                window=istft.window.to(device),
                center=True,
                normalized=False,
                onesided=True,
                length=None,
                return_complex=False
            )
            analyze_tensor("Audio BEFORE tanh", audio_raw)
            
            # After tanh
            audio_final = torch.tanh(audio_raw)
            analyze_tensor("Audio AFTER tanh", audio_final)
            
            return audio_raw, audio_final, mag, phase
        else:
            print("  ⚠️  Unknown head type!")
            return None, None, None, None


def diagnose_full_pipeline(checkpoint_path: str, audio_path: str, output_dir: str = "diagnostic_outputs"):
    """Повна діагностика pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")
    
    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load checkpoint
    encoder, decoder, audio_processor, params = load_checkpoint_for_diagnosis(checkpoint_path, device)
    
    # Load audio
    print(f"\n{'='*70}")
    print("ВХІДНЕ АУДІО")
    print(f"{'='*70}")
    
    audio, sr = torchaudio.load(audio_path)
    print(f"  Файл: {audio_path}")
    print(f"  Original SR: {sr}")
    
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    if sr != params["sample_rate"]:
        resampler = torchaudio.transforms.Resample(sr, params["sample_rate"])
        audio = resampler(audio)
    
    audio = audio.squeeze(0)
    analyze_tensor("Вхідне аудіо", audio)
    
    # Compute mel
    print(f"\n{'='*70}")
    print("MEL SPECTROGRAM")
    print(f"{'='*70}")
    
    mel = audio_processor.compute_mel(audio)
    analyze_tensor("Mel spectrogram", mel)
    
    # Prepare for model
    audio_batch = audio.unsqueeze(0).to(device)
    mel_batch = mel.unsqueeze(0).to(device)
    
    # Encoder
    print(f"\n{'='*70}")
    print("ENCODER")
    print(f"{'='*70}")
    
    with torch.no_grad():
        latent = encoder(mel_batch)
    analyze_tensor("Latent vectors", latent)
    
    # Decoder with detailed diagnostics
    audio_raw, audio_final, mag, phase = diagnose_decoder_internals(decoder, latent, device)
    
    # Compare with original
    print(f"\n{'='*70}")
    print("ПОРІВНЯННЯ З ОРИГІНАЛОМ")
    print(f"{'='*70}")
    
    min_len = min(audio.shape[-1], audio_final.shape[-1])
    orig = audio[:min_len]
    recon = audio_final[0, :min_len].cpu()
    
    l1 = F.l1_loss(recon, orig).item()
    mse = F.mse_loss(recon, orig).item()
    
    # SNR
    signal_power = (orig ** 2).mean()
    noise_power = ((orig - recon) ** 2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
    
    print(f"\n  L1 Loss: {l1:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  SNR: {snr:.2f} dB")
    
    # Amplitude comparison
    print(f"\n  Amplitude comparison:")
    print(f"    Original - max: {orig.abs().max():.4f}, mean: {orig.abs().mean():.4f}")
    print(f"    Reconstructed - max: {recon.abs().max():.4f}, mean: {recon.abs().mean():.4f}")
    print(f"    Ratio (recon/orig): {recon.abs().mean() / (orig.abs().mean() + 1e-8):.4f}")
    
    # Save diagnostic plots
    print(f"\n{'='*70}")
    print("ЗБЕРЕЖЕННЯ ДІАГНОСТИЧНИХ ГРАФІКІВ")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    # 1. Waveforms
    axes[0, 0].plot(orig.numpy(), alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title("Original Waveform")
    axes[0, 0].set_ylim(-1.1, 1.1)
    
    axes[0, 1].plot(recon.numpy(), alpha=0.7, linewidth=0.5, color='orange')
    axes[0, 1].set_title("Reconstructed Waveform")
    axes[0, 1].set_ylim(-1.1, 1.1)
    
    # 2. Mel spectrograms
    axes[1, 0].imshow(mel.numpy(), aspect='auto', origin='lower')
    axes[1, 0].set_title("Original Mel")
    axes[1, 0].set_ylabel("Mel bin")
    
    recon_mel = audio_processor.compute_mel(recon)
    axes[1, 1].imshow(recon_mel.numpy(), aspect='auto', origin='lower')
    axes[1, 1].set_title("Reconstructed Mel")
    
    # 3. Latent space
    latent_np = latent[0].cpu().numpy()
    axes[2, 0].imshow(latent_np, aspect='auto', origin='lower', cmap='RdBu')
    axes[2, 0].set_title(f"Latent vectors (range: [{latent_np.min():.2f}, {latent_np.max():.2f}])")
    axes[2, 0].set_ylabel("Latent dim")
    axes[2, 0].colorbar = plt.colorbar(axes[2, 0].images[0], ax=axes[2, 0])
    
    # 4. Magnitude spectrum (log scale)
    mag_np = mag[0].cpu().numpy()
    axes[2, 1].imshow(np.log10(mag_np.T + 1e-8), aspect='auto', origin='lower', cmap='magma')
    axes[2, 1].set_title(f"Magnitude (log10, range: [{mag_np.min():.2f}, {mag_np.max():.2f}])")
    axes[2, 1].set_ylabel("Freq bin")
    
    # 5. Audio before tanh
    audio_raw_np = audio_raw[0].cpu().numpy()
    axes[3, 0].plot(audio_raw_np[:10000], alpha=0.7, linewidth=0.5, color='red')
    axes[3, 0].set_title(f"Audio BEFORE tanh (first 10k samples, range: [{audio_raw_np.min():.2f}, {audio_raw_np.max():.2f}])")
    axes[3, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[3, 0].axhline(y=-1, color='black', linestyle='--', alpha=0.5)
    
    # 6. Histogram of raw audio values
    axes[3, 1].hist(audio_raw_np.flatten(), bins=100, alpha=0.7, color='red')
    axes[3, 1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
    axes[3, 1].axvline(x=-1, color='black', linestyle='--', alpha=0.5)
    axes[3, 1].set_title(f"Histogram of audio BEFORE tanh")
    axes[3, 1].set_xlabel("Amplitude")
    axes[3, 1].set_ylabel("Count")
    
    plt.tight_layout()
    plot_path = output_path / "diagnostic_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  📊 Saved: {plot_path}")
    
    # Save audio files
    audio_raw_path = output_path / "audio_before_tanh.wav"
    audio_final_path = output_path / "audio_after_tanh.wav"
    original_path = output_path / "original.wav"
    
    # Normalize raw audio for listening (it might be very loud)
    audio_raw_cpu = audio_raw[0].cpu()
    audio_raw_normalized = audio_raw_cpu / (audio_raw_cpu.abs().max() + 1e-8)
    
    torchaudio.save(str(audio_raw_path), audio_raw_normalized.unsqueeze(0), params["sample_rate"])
    torchaudio.save(str(audio_final_path), audio_final[0].cpu().unsqueeze(0), params["sample_rate"])
    torchaudio.save(str(original_path), orig.unsqueeze(0), params["sample_rate"])
    
    print(f"  🔊 Saved: {original_path}")
    print(f"  🔊 Saved: {audio_raw_path} (normalized)")
    print(f"  🔊 Saved: {audio_final_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 SUMMARY / ВИСНОВКИ")
    print(f"{'='*70}")
    
    # Detect issues
    issues = []
    
    if audio_raw.abs().max() > 10:
        issues.append(f"❌ Audio before tanh has extreme values: max={audio_raw.abs().max():.2f}")
    
    if mag.max() > 100:
        issues.append(f"❌ Magnitude spectrum too large: max={mag.max():.2f}")
    
    if latent.abs().max() > 10:
        issues.append(f"⚠️  Latent values might be too large: max={latent.abs().max():.2f}")
    
    clipped_percent = ((audio_final.abs() > 0.99).sum() / audio_final.numel() * 100).item()
    if clipped_percent > 5:
        issues.append(f"❌ {clipped_percent:.1f}% of output is clipped (saturated at ±1)")
    
    if snr < 0:
        issues.append(f"❌ Negative SNR ({snr:.1f} dB) - reconstruction is mostly noise")
    
    if len(issues) == 0:
        print("  ✅ No major issues detected")
    else:
        print("  Виявлені проблеми:")
        for issue in issues:
            print(f"    {issue}")
    
    print(f"\n  Рекомендації:")
    if mag.max() > 100:
        print("    - Magnitude занадто великий. Спробуй clamp exp() output")
        print("    - Або використай log1p замість exp")
    if audio_raw.abs().max() > 10:
        print("    - iSTFT генерує занадто гучний сигнал")
        print("    - Можливо потрібна нормалізація magnitude")
    if clipped_percent > 5:
        print("    - Багато кліпінгу. Модель ще вчиться контролювати амплітуду")
        print("    - Зачекай більше кроків або збільш waveform loss weight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose autoencoder reconstruction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to test audio")
    parser.add_argument("--output", type=str, default="diagnostic_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    diagnose_full_pipeline(args.checkpoint, args.audio, args.output)
