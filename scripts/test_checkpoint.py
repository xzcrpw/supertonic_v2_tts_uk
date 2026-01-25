#!/usr/bin/env python3
"""
SUPERTONIC V2 - INTERACTIVE CHECKPOINT TESTER
==============================================
Інтерактивний скрипт для тестування чекпоінтів autoencoder.

Можливості:
  - Вибір чекпоінта
  - Reconstruction аудіо
  - Порівняння якості (PESQ, STOI, спектрограми)
  - A/B тест різних чекпоінтів
  - Збереження результатів

Usage:
    python scripts/test_checkpoint.py
    python scripts/test_checkpoint.py --checkpoint checkpoints/autoencoder/step_5000.pt
    python scripts/test_checkpoint.py --audio test.wav
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import json
from datetime import datetime

import torch
import torchaudio
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from supertonic.data.preprocessing import AudioProcessor

# Optional quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    
try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple:
    """Load encoder and decoder from checkpoint."""
    from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
    
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get("config", {})
    
    # Get audio params from config
    audio_config = config.get("audio", {})
    n_fft = audio_config.get("n_fft", 1024)
    hop_length = audio_config.get("hop_length", 256)
    n_mels = audio_config.get("n_mels", 100)
    sample_rate = audio_config.get("sample_rate", 22050)
    
    # CRITICAL: Detect actual parameters from checkpoint weights (config might be wrong due to bug)
    
    # Detect n_mels from encoder input_conv weight shape: [hidden, n_mels, 1]
    if "encoder" in checkpoint:
        encoder_state = checkpoint["encoder"]
        if "input_conv.weight" in encoder_state:
            actual_n_mels = encoder_state["input_conv.weight"].shape[1]
            if actual_n_mels != n_mels:
                print(f"   ⚠️  Config says n_mels={n_mels}, but weights have n_mels={actual_n_mels}")
                n_mels = actual_n_mels
    
    # Detect n_fft from decoder istft_head.window shape
    if "decoder" in checkpoint:
        decoder_state = checkpoint["decoder"]
        if "istft_head.window" in decoder_state:
            actual_n_fft = decoder_state["istft_head.window"].shape[0]
            if actual_n_fft != n_fft:
                print(f"   ⚠️  Config says n_fft={n_fft}, but weights have n_fft={actual_n_fft}")
                n_fft = actual_n_fft
                # Adjust hop_length proportionally
                if actual_n_fft == 2048:
                    hop_length = 512
                    sample_rate = 44100
                elif actual_n_fft == 1024:
                    hop_length = 256
                    sample_rate = 22050
    
    print(f"   ⚠️  Using detected params: n_fft={n_fft}, hop={hop_length}, mels={n_mels}, sr={sample_rate}")
    print(f"   Audio config: n_fft={n_fft}, hop={hop_length}, mels={n_mels}, sr={sample_rate}")
    
    # Default architecture params
    encoder_params = {
        "input_dim": n_mels,
        "hidden_dim": 512,
        "output_dim": 24,
        "num_blocks": 10,
        "kernel_size": 7,
    }
    
    decoder_params = {
        "input_dim": 24,
        "hidden_dim": 512,
        "num_blocks": 10,
        "kernel_size": 7,
        "dilations": [1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
        "causal": True,
        "n_fft": n_fft,
        "hop_length": hop_length,
    }
    
    # Override with config if available (but preserve n_fft/hop_length we detected from weights!)
    detected_n_fft = n_fft
    detected_hop = hop_length
    detected_mels = n_mels
    
    if "model" in config and "autoencoder" in config["model"]:
        ae_config = config["model"]["autoencoder"]
        if "encoder" in ae_config:
            encoder_params.update(ae_config["encoder"])
        if "decoder" in ae_config:
            for k, v in ae_config["decoder"].items():
                decoder_params[k] = v
    
    # Force use detected values (they come from actual weights, not buggy config)
    encoder_params["input_dim"] = detected_mels
    decoder_params["n_fft"] = detected_n_fft
    decoder_params["hop_length"] = detected_hop
    
    print(f"   Decoder params: n_fft={decoder_params.get('n_fft')}, hop={decoder_params.get('hop_length')}")
    
    # Create models
    encoder = LatentEncoder(**encoder_params).to(device)
    decoder = LatentDecoder(**decoder_params).to(device)
    
    # Helper to strip DDP "module." prefix from state dict
    def strip_ddp_prefix(state_dict):
        """Remove 'module.' prefix added by DistributedDataParallel."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove "module." (7 chars)
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    # Load weights (handle DDP prefix)
    encoder.load_state_dict(strip_ddp_prefix(checkpoint["encoder"]))
    decoder.load_state_dict(strip_ddp_prefix(checkpoint["decoder"]))
    
    encoder.eval()
    decoder.eval()
    
    # Try both "iteration" (new) and "step" (legacy) keys
    step = checkpoint.get("iteration", checkpoint.get("step", 0))
    print(f"   ✅ Loaded step {step:,}")
    
    # Update config with detected values (for use in reconstruction)
    config["audio"] = {
        "n_fft": detected_n_fft,
        "hop_length": detected_hop,
        "n_mels": detected_mels,
        "sample_rate": sample_rate,
    }
    
    return encoder, decoder, step, config


def load_audio(audio_path: str, target_sr: int = 22050) -> torch.Tensor:
    """Load and resample audio."""
    audio, sr = torchaudio.load(audio_path)
    
    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    return audio.squeeze(0)


def compute_mel(audio: torch.Tensor, 
                sample_rate: int = 22050,
                n_fft: int = 1024,
                hop_length: int = 256,
                n_mels: int = 100) -> torch.Tensor:
    """Compute mel spectrogram using AudioProcessor (same as training!)."""
    audio_processor = AudioProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sample_rate / 2
    )
    
    return audio_processor.compute_mel(audio, log_scale=True)


def reconstruct_audio(encoder, decoder, audio: torch.Tensor, 
                      device: str = "cuda",
                      sample_rate: int = 22050,
                      n_fft: int = 1024,
                      hop_length: int = 256,
                      n_mels: int = 100) -> torch.Tensor:
    """Reconstruct audio through autoencoder."""
    with torch.no_grad():
        # Compute mel
        mel = compute_mel(audio, sample_rate=sample_rate, n_fft=n_fft, 
                         hop_length=hop_length, n_mels=n_mels)
        mel = mel.unsqueeze(0).to(device)  # [1, n_mels, T]
        
        # Encode
        latent = encoder(mel)  # [1, latent_dim, T']
        
        # Decode
        reconstructed = decoder(latent)  # [1, T]
        
    return reconstructed.squeeze(0).cpu()


def compute_metrics(original: torch.Tensor, 
                    reconstructed: torch.Tensor,
                    sample_rate: int = 22050) -> dict:
    """Compute quality metrics."""
    metrics = {}
    
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Convert to numpy
    orig_np = original.numpy()
    recon_np = reconstructed.numpy()
    
    # L1 loss
    metrics["l1_loss"] = float(np.mean(np.abs(orig_np - recon_np)))
    
    # MSE
    metrics["mse"] = float(np.mean((orig_np - recon_np) ** 2))
    
    # SNR
    signal_power = np.mean(orig_np ** 2)
    noise_power = np.mean((orig_np - recon_np) ** 2)
    if noise_power > 0:
        metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power))
    else:
        metrics["snr_db"] = float("inf")
    
    # PESQ (if available)
    if PESQ_AVAILABLE:
        try:
            # PESQ requires 16kHz or 8kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                orig_16k = resampler(original).numpy()
                recon_16k = resampler(reconstructed).numpy()
            else:
                orig_16k = orig_np
                recon_16k = recon_np
            
            metrics["pesq"] = float(pesq(16000, orig_16k, recon_16k, 'wb'))
        except Exception as e:
            metrics["pesq"] = f"Error: {e}"
    
    # STOI (if available)
    if STOI_AVAILABLE:
        try:
            metrics["stoi"] = float(stoi(orig_np, recon_np, sample_rate, extended=False))
        except Exception as e:
            metrics["stoi"] = f"Error: {e}"
    
    # Mel spectrogram loss
    orig_mel = compute_mel(original, sample_rate=sample_rate)
    recon_mel = compute_mel(reconstructed, sample_rate=sample_rate)
    min_mel_len = min(orig_mel.shape[1], recon_mel.shape[1])
    metrics["mel_l1"] = float(torch.mean(torch.abs(
        orig_mel[:, :min_mel_len] - recon_mel[:, :min_mel_len]
    )).item())
    
    return metrics


def plot_comparison(original: torch.Tensor,
                    reconstructed: torch.Tensor,
                    save_path: str,
                    sample_rate: int = 22050,
                    title: str = ""):
    """Plot waveform and spectrogram comparison."""
    if not MATPLOTLIB_AVAILABLE:
        print("   ⚠️  matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Waveforms
    time_orig = np.arange(len(original)) / sample_rate
    time_recon = np.arange(len(reconstructed)) / sample_rate
    
    axes[0, 0].plot(time_orig, original.numpy(), alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title("Original Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_ylim(-1, 1)
    
    axes[0, 1].plot(time_recon, reconstructed.numpy(), alpha=0.7, linewidth=0.5, color='orange')
    axes[0, 1].set_title("Reconstructed Waveform")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_ylim(-1, 1)
    
    # Spectrograms
    orig_mel = compute_mel(original, sample_rate=sample_rate).numpy()
    recon_mel = compute_mel(reconstructed, sample_rate=sample_rate).numpy()
    
    im1 = axes[1, 0].imshow(orig_mel, aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title("Original Mel Spectrogram")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Mel bin")
    plt.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[1, 1].imshow(recon_mel, aspect='auto', origin='lower', cmap='magma')
    axes[1, 1].set_title("Reconstructed Mel Spectrogram")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Mel bin")
    plt.colorbar(im2, ax=axes[1, 1])
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   📊 Saved plot: {save_path}")


def list_checkpoints(checkpoint_dir: str = "checkpoints/autoencoder") -> List[str]:
    """List available checkpoints, sorted by step number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    # Sort by step number (extract number from filename)
    def get_step(path):
        name = path.stem  # checkpoint_5000 -> checkpoint_5000
        parts = name.split("_")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0
    
    checkpoints = sorted(checkpoints, key=get_step)
    return [str(cp) for cp in checkpoints]


def list_audio_files(data_dir: str = "data", limit: int = 20) -> List[str]:
    """List some audio files for testing."""
    data_dir = Path(data_dir)
    audio_files = []
    
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        for f in data_dir.rglob(ext):
            audio_files.append(str(f))
            if len(audio_files) >= limit:
                break
        if len(audio_files) >= limit:
            break
    
    return audio_files


def interactive_menu():
    """Interactive menu for checkpoint testing."""
    print("\n" + "="*70)
    print("🎵 SUPERTONIC V2 - CHECKPOINT TESTER")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Audio params (will be updated from checkpoint config)
    audio_params = {
        "sample_rate": 22050,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 100,
    }
    
    encoder = None
    decoder = None
    current_step = 0
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    while True:
        print("\n" + "-"*50)
        print("📋 MENU:")
        print("   1. List checkpoints")
        print("   2. Load checkpoint")
        print("   3. List test audio files")
        print("   4. Test reconstruction")
        print("   5. Compare two checkpoints")
        print("   6. Batch test (multiple files)")
        print("   0. Exit")
        print("-"*50)
        
        choice = input("Choose option: ").strip()
        
        if choice == "0":
            print("\n👋 Bye!")
            break
            
        elif choice == "1":
            checkpoints = list_checkpoints()
            if not checkpoints:
                print("   ❌ No checkpoints found in checkpoints/autoencoder/")
            else:
                print(f"\n   📦 Found {len(checkpoints)} checkpoints:")
                for i, cp in enumerate(checkpoints):
                    print(f"      [{i}] {Path(cp).name}")
                    
        elif choice == "2":
            checkpoints = list_checkpoints()
            if not checkpoints:
                print("   ❌ No checkpoints found")
                continue
                
            print(f"\n   Available checkpoints:")
            for i, cp in enumerate(checkpoints):
                print(f"      [{i}] {Path(cp).name}")
            
            idx = input("   Enter index or path: ").strip()
            
            try:
                if idx.isdigit():
                    cp_path = checkpoints[int(idx)]
                else:
                    cp_path = idx
                    
                encoder, decoder, current_step, config = load_checkpoint(cp_path, device)
                
                # Update audio params from config
                if config and "audio" in config:
                    audio_params.update({
                        "sample_rate": config["audio"].get("sample_rate", 22050),
                        "n_fft": config["audio"].get("n_fft", 1024),
                        "hop_length": config["audio"].get("hop_length", 256),
                        "n_mels": config["audio"].get("n_mels", 100),
                    })
                    print(f"   📊 Audio params: {audio_params}")
                
                print(f"   ✅ Loaded checkpoint at step {current_step:,}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == "3":
            audio_files = list_audio_files()
            if not audio_files:
                print("   ❌ No audio files found in data/")
            else:
                print(f"\n   🎵 Found {len(audio_files)} audio files (showing first 20):")
                for i, af in enumerate(audio_files[:20]):
                    print(f"      [{i}] {Path(af).name}")
                    
        elif choice == "4":
            if encoder is None:
                print("   ❌ Load a checkpoint first (option 2)")
                continue
            
            audio_path = input("   Enter audio path (or index from list): ").strip()
            
            try:
                if audio_path.isdigit():
                    audio_files = list_audio_files()
                    audio_path = audio_files[int(audio_path)]
                
                sample_rate = audio_params["sample_rate"]
                
                print(f"\n   🎵 Loading: {audio_path}")
                audio = load_audio(audio_path, target_sr=sample_rate)
                print(f"   Duration: {len(audio)/sample_rate:.2f}s")
                
                # Limit length for testing
                max_samples = sample_rate * 10  # 10 seconds max
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                    print(f"   Truncated to {max_samples/sample_rate:.1f}s")
                
                print("\n   🔄 Reconstructing...")
                reconstructed = reconstruct_audio(
                    encoder, decoder, audio, device, 
                    sample_rate=audio_params["sample_rate"],
                    n_fft=audio_params["n_fft"],
                    hop_length=audio_params["hop_length"],
                    n_mels=audio_params["n_mels"]
                )
                
                print("\n   📊 Computing metrics...")
                metrics = compute_metrics(audio, reconstructed, sample_rate)
                
                print("\n   Results:")
                print(f"      L1 Loss:     {metrics['l1_loss']:.6f}")
                print(f"      MSE:         {metrics['mse']:.6f}")
                print(f"      SNR:         {metrics['snr_db']:.2f} dB")
                print(f"      Mel L1:      {metrics['mel_l1']:.4f}")
                if 'pesq' in metrics:
                    print(f"      PESQ:        {metrics['pesq']}")
                if 'stoi' in metrics:
                    print(f"      STOI:        {metrics['stoi']}")
                
                # Save outputs
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = Path(audio_path).stem
                
                # Save audio
                orig_path = output_dir / f"{base_name}_step{current_step}_original.wav"
                recon_path = output_dir / f"{base_name}_step{current_step}_reconstructed.wav"
                
                torchaudio.save(str(orig_path), audio.unsqueeze(0), sample_rate)
                torchaudio.save(str(recon_path), reconstructed.unsqueeze(0), sample_rate)
                
                print(f"\n   💾 Saved:")
                print(f"      {orig_path}")
                print(f"      {recon_path}")
                
                # Plot
                plot_path = output_dir / f"{base_name}_step{current_step}_comparison.png"
                plot_comparison(audio, reconstructed, str(plot_path), sample_rate, 
                              f"Step {current_step:,} - {base_name}")
                
                # Save metrics
                metrics_path = output_dir / f"{base_name}_step{current_step}_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump({
                        "audio_path": audio_path,
                        "step": current_step,
                        "metrics": metrics,
                        "timestamp": timestamp,
                    }, f, indent=2)
                print(f"      {metrics_path}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == "5":
            print("\n   🔄 Compare two checkpoints")
            checkpoints = list_checkpoints()
            
            if len(checkpoints) < 2:
                print("   ❌ Need at least 2 checkpoints")
                continue
            
            print("   Available checkpoints:")
            for i, cp in enumerate(checkpoints):
                print(f"      [{i}] {Path(cp).name}")
            
            try:
                idx1 = int(input("   First checkpoint index: ").strip())
                idx2 = int(input("   Second checkpoint index: ").strip())
                audio_idx = input("   Audio file (path or index): ").strip()
                
                if audio_idx.isdigit():
                    audio_files = list_audio_files()
                    audio_path = audio_files[int(audio_idx)]
                else:
                    audio_path = audio_idx
                
                results = []
                for idx in [idx1, idx2]:
                    cp_path = checkpoints[idx]
                    enc, dec, step, config = load_checkpoint(cp_path, device)
                    
                    # Get audio params from this checkpoint's config
                    ap = {
                        "sample_rate": config.get("audio", {}).get("sample_rate", 22050),
                        "n_fft": config.get("audio", {}).get("n_fft", 1024),
                        "hop_length": config.get("audio", {}).get("hop_length", 256),
                        "n_mels": config.get("audio", {}).get("n_mels", 100),
                    }
                    
                    # Load audio with this checkpoint's sample rate
                    audio = load_audio(audio_path, target_sr=ap["sample_rate"])
                    max_samples = ap["sample_rate"] * 10
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    
                    recon = reconstruct_audio(enc, dec, audio, device, **ap)
                    metrics = compute_metrics(audio, recon, ap["sample_rate"])
                    results.append({
                        "step": step,
                        "path": cp_path,
                        "metrics": metrics,
                        "reconstructed": recon,
                        "sample_rate": ap["sample_rate"],
                    })
                
                print("\n   📊 Comparison:")
                print(f"   {'Metric':<15} {'Step '+str(results[0]['step']):<15} {'Step '+str(results[1]['step']):<15} {'Better':<10}")
                print("   " + "-"*55)
                
                for metric in ["l1_loss", "mse", "mel_l1"]:
                    v1 = results[0]["metrics"][metric]
                    v2 = results[1]["metrics"][metric]
                    better = "←" if v1 < v2 else "→"
                    print(f"   {metric:<15} {v1:<15.6f} {v2:<15.6f} {better:<10}")
                
                for metric in ["snr_db"]:
                    if metric in results[0]["metrics"]:
                        v1 = results[0]["metrics"][metric]
                        v2 = results[1]["metrics"][metric]
                        better = "←" if v1 > v2 else "→"
                        print(f"   {metric:<15} {v1:<15.2f} {v2:<15.2f} {better:<10}")
                
                # Save both
                base_name = Path(audio_path).stem
                for r in results:
                    sr = r.get("sample_rate", 22050)
                    recon_path = output_dir / f"{base_name}_step{r['step']}_reconstructed.wav"
                    torchaudio.save(str(recon_path), r['reconstructed'].unsqueeze(0), sr)
                    print(f"\n   💾 Saved: {recon_path}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == "6":
            if encoder is None:
                print("   ❌ Load a checkpoint first (option 2)")
                continue
            
            audio_files = list_audio_files(limit=50)
            n = input(f"   How many files to test? (max {len(audio_files)}): ").strip()
            n = min(int(n), len(audio_files))
            
            sample_rate = audio_params["sample_rate"]
            
            all_metrics = []
            for i, af in enumerate(audio_files[:n]):
                print(f"\n   [{i+1}/{n}] {Path(af).name}")
                try:
                    audio = load_audio(af, target_sr=sample_rate)
                    max_samples = sample_rate * 10
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    
                    recon = reconstruct_audio(
                        encoder, decoder, audio, device,
                        sample_rate=audio_params["sample_rate"],
                        n_fft=audio_params["n_fft"],
                        hop_length=audio_params["hop_length"],
                        n_mels=audio_params["n_mels"]
                    )
                    metrics = compute_metrics(audio, recon, sample_rate)
                    metrics["file"] = af
                    all_metrics.append(metrics)
                    
                    print(f"      L1: {metrics['l1_loss']:.6f}, Mel L1: {metrics['mel_l1']:.4f}, SNR: {metrics['snr_db']:.1f}dB")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
            
            # Summary
            if all_metrics:
                print("\n   📊 SUMMARY:")
                avg_l1 = np.mean([m["l1_loss"] for m in all_metrics])
                avg_mel = np.mean([m["mel_l1"] for m in all_metrics])
                avg_snr = np.mean([m["snr_db"] for m in all_metrics])
                
                print(f"      Avg L1 Loss:  {avg_l1:.6f}")
                print(f"      Avg Mel L1:   {avg_mel:.4f}")
                print(f"      Avg SNR:      {avg_snr:.2f} dB")
                
                # Save summary
                summary_path = output_dir / f"batch_test_step{current_step}.json"
                with open(summary_path, "w") as f:
                    json.dump({
                        "step": current_step,
                        "n_files": n,
                        "avg_l1_loss": avg_l1,
                        "avg_mel_l1": avg_mel,
                        "avg_snr_db": avg_snr,
                        "files": all_metrics,
                    }, f, indent=2)
                print(f"\n   💾 Saved: {summary_path}")
        
        else:
            print("   ❌ Unknown option")


def main():
    parser = argparse.ArgumentParser(description="Test autoencoder checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--audio", type=str, help="Audio file to test")
    parser.add_argument("--output", type=str, default="test_outputs", help="Output directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # If no args or interactive flag, run interactive mode
    if args.interactive or (args.checkpoint is None and args.audio is None):
        interactive_menu()
        return
    
    # Quick test mode
    if args.checkpoint and args.audio:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        encoder, decoder, step, config = load_checkpoint(args.checkpoint, device)
        
        # Get audio params from config
        audio_params = {
            "sample_rate": config.get("audio", {}).get("sample_rate", 22050),
            "n_fft": config.get("audio", {}).get("n_fft", 1024),
            "hop_length": config.get("audio", {}).get("hop_length", 256),
            "n_mels": config.get("audio", {}).get("n_mels", 100),
        }
        sample_rate = audio_params["sample_rate"]
        
        print(f"\n🎵 Loading: {args.audio}")
        audio = load_audio(args.audio, target_sr=sample_rate)
        
        print("🔄 Reconstructing...")
        reconstructed = reconstruct_audio(encoder, decoder, audio, device, **audio_params)
        
        print("📊 Computing metrics...")
        metrics = compute_metrics(audio, reconstructed, sample_rate)
        
        print("\n📊 Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.6f}")
            else:
                print(f"   {k}: {v}")
        
        # Save
        base_name = Path(args.audio).stem
        recon_path = output_dir / f"{base_name}_step{step}_reconstructed.wav"
        torchaudio.save(str(recon_path), reconstructed.unsqueeze(0), sample_rate)
        print(f"\n💾 Saved: {recon_path}")
        
        # Plot
        plot_path = output_dir / f"{base_name}_step{step}_comparison.png"
        plot_comparison(audio, reconstructed, str(plot_path), sample_rate, f"Step {step:,}")


if __name__ == "__main__":
    main()
