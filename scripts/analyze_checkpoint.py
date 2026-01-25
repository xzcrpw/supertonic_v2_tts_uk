#!/usr/bin/env python3
"""
🎵 SUPERTONIC V2 - ADVANCED CHECKPOINT ANALYZER

Comprehensive analysis with TTS-relevant metrics:
- Mel-based metrics (most important for TTS)
- Multi-resolution spectral analysis
- Perceptual quality indicators
- Amplitude/dynamics analysis
- Batch comparison across checkpoints

Usage:
    python scripts/analyze_checkpoint.py --checkpoint checkpoints/autoencoder/checkpoint_20000.pt
    python scripts/analyze_checkpoint.py --checkpoint checkpoints/autoencoder/checkpoint_20000.pt --audio path/to/test.wav
    python scripts/analyze_checkpoint.py --compare checkpoints/autoencoder/  # Compare all checkpoints
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
from supertonic.data.preprocessing import AudioProcessor


# ═══════════════════════════════════════════════════════════════════════════════
# COLORS AND FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def header(text: str) -> str:
    line = "═" * 70
    return f"\n{Colors.CYAN}{line}\n{Colors.BOLD}  {text}\n{Colors.CYAN}{line}{Colors.RESET}"


def subheader(text: str) -> str:
    return f"\n{Colors.YELLOW}▶ {text}{Colors.RESET}"


def metric_bar(value: float, max_val: float, width: int = 30, invert: bool = False) -> str:
    """Create a visual progress bar for metrics."""
    ratio = min(value / max_val, 1.0)
    if invert:
        ratio = 1 - ratio
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    
    # Color based on quality
    if invert:
        color = Colors.GREEN if ratio > 0.7 else Colors.YELLOW if ratio > 0.4 else Colors.RED
    else:
        color = Colors.GREEN if ratio < 0.3 else Colors.YELLOW if ratio < 0.6 else Colors.RED
    
    return f"{color}{bar}{Colors.RESET}"


def quality_emoji(score: float, thresholds: Tuple[float, float] = (0.3, 0.6)) -> str:
    """Return emoji based on quality score (lower is better)."""
    if score < thresholds[0]:
        return "🟢"
    elif score < thresholds[1]:
        return "🟡"
    else:
        return "🔴"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioMetrics:
    """Comprehensive audio quality metrics."""
    # Mel-based (most important for TTS)
    mel_l1: float
    mel_mse: float
    mel_cosine_sim: float
    
    # Multi-resolution mel
    mel_l1_256: float   # Fine detail
    mel_l1_512: float   # Medium
    mel_l1_1024: float  # Coarse structure
    
    # Spectral
    spectral_convergence: float
    log_spectral_distance: float
    
    # Amplitude/Dynamics
    amplitude_ratio: float
    rms_ratio: float
    peak_ratio: float
    dynamic_range_diff: float
    
    # Waveform (less important)
    l1_loss: float
    mse_loss: float
    
    # Overall score (composite)
    overall_score: float


@dataclass
class CheckpointInfo:
    """Checkpoint metadata."""
    path: str
    step: int
    n_mels: int
    hop_length: int
    sample_rate: int


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def strip_ddp_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix from DDP state dict."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[LatentEncoder, LatentDecoder, AudioProcessor, CheckpointInfo]:
    """Load checkpoint and create models."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    encoder_state = strip_ddp_prefix(checkpoint["encoder"])
    decoder_state = strip_ddp_prefix(checkpoint["decoder"])
    
    # Detect params
    n_mels = encoder_state["input_conv.weight"].shape[1]
    
    if "head.fc.weight" in decoder_state:
        hop_length = decoder_state["head.fc.weight"].shape[0]
    else:
        hop_length = 256
    
    n_fft = 1024 if hop_length == 256 else 2048
    sample_rate = 22050 if hop_length == 256 else 44100
    
    # Extract step from filename
    step = 0
    try:
        step = int(Path(checkpoint_path).stem.split("_")[-1])
    except:
        pass
    
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
    
    info = CheckpointInfo(
        path=checkpoint_path,
        step=step,
        n_mels=n_mels,
        hop_length=hop_length,
        sample_rate=sample_rate
    )
    
    return encoder, decoder, audio_processor, info


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mel_spectrogram(audio: torch.Tensor, n_fft: int, hop_length: int, 
                            n_mels: int, sample_rate: int) -> torch.Tensor:
    """Compute mel spectrogram."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="slaney"
    )
    mel = transform(audio)
    return torch.log(mel.clamp(min=1e-5))


def compute_stft(audio: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    """Compute STFT magnitude."""
    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                      window=window, return_complex=True)
    return stft.abs()


def spectral_convergence(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> float:
    """Spectral convergence loss (lower is better)."""
    return (torch.norm(target_mag - pred_mag, p="fro") / torch.norm(target_mag, p="fro")).item()


def log_spectral_distance(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> float:
    """Log spectral distance (lower is better)."""
    pred_log = torch.log(pred_mag.clamp(min=1e-5))
    target_log = torch.log(target_mag.clamp(min=1e-5))
    return torch.mean((pred_log - target_log).pow(2)).sqrt().item()


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor, 
                   sample_rate: int = 22050) -> AudioMetrics:
    """Compute comprehensive audio quality metrics."""
    
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Mel-based metrics (MOST IMPORTANT)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Standard mel (n_fft=1024)
    mel_orig = compute_mel_spectrogram(orig, 1024, 256, 80, sample_rate)
    mel_recon = compute_mel_spectrogram(recon, 1024, 256, 80, sample_rate)
    
    mel_l1 = F.l1_loss(mel_recon, mel_orig).item()
    mel_mse = F.mse_loss(mel_recon, mel_orig).item()
    
    # Cosine similarity (1.0 = identical, 0.0 = orthogonal)
    mel_cosine = F.cosine_similarity(
        mel_orig.flatten().unsqueeze(0), 
        mel_recon.flatten().unsqueeze(0)
    ).item()
    
    # Multi-resolution mel
    mel_orig_256 = compute_mel_spectrogram(orig, 256, 64, 80, sample_rate)
    mel_recon_256 = compute_mel_spectrogram(recon, 256, 64, 80, sample_rate)
    mel_l1_256 = F.l1_loss(mel_recon_256, mel_orig_256).item()
    
    mel_orig_512 = compute_mel_spectrogram(orig, 512, 128, 80, sample_rate)
    mel_recon_512 = compute_mel_spectrogram(recon, 512, 128, 80, sample_rate)
    mel_l1_512 = F.l1_loss(mel_recon_512, mel_orig_512).item()
    
    mel_orig_1024 = compute_mel_spectrogram(orig, 1024, 256, 80, sample_rate)
    mel_recon_1024 = compute_mel_spectrogram(recon, 1024, 256, 80, sample_rate)
    mel_l1_1024 = F.l1_loss(mel_recon_1024, mel_orig_1024).item()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Spectral metrics
    # ═══════════════════════════════════════════════════════════════════════════
    
    stft_orig = compute_stft(orig, 1024, 256)
    stft_recon = compute_stft(recon, 1024, 256)
    
    sc = spectral_convergence(stft_recon, stft_orig)
    lsd = log_spectral_distance(stft_recon, stft_orig)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Amplitude/Dynamics
    # ═══════════════════════════════════════════════════════════════════════════
    
    orig_abs = orig.abs()
    recon_abs = recon.abs()
    
    amplitude_ratio = (recon_abs.mean() / (orig_abs.mean() + 1e-8)).item()
    rms_ratio = (recon.pow(2).mean().sqrt() / (orig.pow(2).mean().sqrt() + 1e-8)).item()
    peak_ratio = (recon_abs.max() / (orig_abs.max() + 1e-8)).item()
    
    # Dynamic range (dB)
    orig_dr = 20 * np.log10((orig_abs.max() / (orig_abs[orig_abs > 0].min() + 1e-8)).item() + 1e-8)
    recon_dr = 20 * np.log10((recon_abs.max() / (recon_abs[recon_abs > 0].min() + 1e-8)).item() + 1e-8)
    dynamic_range_diff = abs(orig_dr - recon_dr)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Waveform metrics (less important for TTS)
    # ═══════════════════════════════════════════════════════════════════════════
    
    l1_loss = F.l1_loss(recon, orig).item()
    mse_loss = F.mse_loss(recon, orig).item()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Overall score (weighted composite)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Lower is better, normalized roughly to 0-1 range
    overall = (
        mel_l1 * 0.4 +                    # 40% - most important
        sc * 0.2 +                         # 20% - spectral shape
        (1 - mel_cosine) * 0.2 +          # 20% - similarity
        abs(1 - amplitude_ratio) * 0.1 +  # 10% - loudness
        lsd * 0.1                          # 10% - log spectral
    )
    
    return AudioMetrics(
        mel_l1=mel_l1,
        mel_mse=mel_mse,
        mel_cosine_sim=mel_cosine,
        mel_l1_256=mel_l1_256,
        mel_l1_512=mel_l1_512,
        mel_l1_1024=mel_l1_1024,
        spectral_convergence=sc,
        log_spectral_distance=lsd,
        amplitude_ratio=amplitude_ratio,
        rms_ratio=rms_ratio,
        peak_ratio=peak_ratio,
        dynamic_range_diff=dynamic_range_diff,
        l1_loss=l1_loss,
        mse_loss=mse_loss,
        overall_score=overall
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_audio(encoder: LatentEncoder, decoder: LatentDecoder,
                     audio_processor: AudioProcessor, audio: torch.Tensor,
                     device: str = "cuda") -> torch.Tensor:
    """Reconstruct audio through encoder-decoder."""
    with torch.no_grad():
        mel = audio_processor.compute_mel(audio).unsqueeze(0).to(device)
        latent = encoder(mel)
        reconstructed = decoder(latent)
    return reconstructed.squeeze(0).cpu()


def analyze_single(checkpoint_path: str, audio_path: str, output_dir: str = "analysis_output",
                  device: str = "cuda") -> AudioMetrics:
    """Analyze single audio file with checkpoint."""
    
    print(header("SUPERTONIC V2 - CHECKPOINT ANALYZER"))
    print(f"\n  {Colors.DIM}Checkpoint:{Colors.RESET} {checkpoint_path}")
    print(f"  {Colors.DIM}Audio:{Colors.RESET} {audio_path}")
    
    # Load checkpoint
    print(subheader("Loading checkpoint..."))
    encoder, decoder, audio_processor, info = load_checkpoint(checkpoint_path, device)
    print(f"  ✓ Step: {Colors.BOLD}{info.step:,}{Colors.RESET}")
    print(f"  ✓ Config: {info.n_mels} mels, hop={info.hop_length}, sr={info.sample_rate}")
    
    # Load audio
    print(subheader("Loading audio..."))
    audio, sr = torchaudio.load(audio_path)
    if sr != info.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, info.sample_rate)
    audio = audio.mean(dim=0)  # Mono
    
    duration = len(audio) / info.sample_rate
    print(f"  ✓ Duration: {duration:.2f}s ({len(audio):,} samples)")
    
    # Reconstruct
    print(subheader("Reconstructing..."))
    reconstructed = reconstruct_audio(encoder, decoder, audio_processor, audio, device)
    
    # Compute metrics
    print(subheader("Computing metrics..."))
    metrics = compute_metrics(audio, reconstructed, info.sample_rate)
    
    # Display results
    print(header("RESULTS"))
    
    # Mel metrics (most important)
    print(f"\n  {Colors.BOLD}📊 MEL METRICS (Primary){Colors.RESET}")
    print(f"  ├─ Mel L1:        {metrics.mel_l1:.4f}  {metric_bar(metrics.mel_l1, 1.0)} {quality_emoji(metrics.mel_l1)}")
    print(f"  ├─ Mel MSE:       {metrics.mel_mse:.4f}  {metric_bar(metrics.mel_mse, 0.5)}")
    print(f"  └─ Mel Cosine:    {metrics.mel_cosine_sim:.4f}  {metric_bar(1-metrics.mel_cosine_sim, 1.0, invert=True)} {'🟢' if metrics.mel_cosine_sim > 0.9 else '🟡' if metrics.mel_cosine_sim > 0.7 else '🔴'}")
    
    # Multi-resolution
    print(f"\n  {Colors.BOLD}🔍 MULTI-RESOLUTION MEL{Colors.RESET}")
    print(f"  ├─ Fine (256):    {metrics.mel_l1_256:.4f}  {metric_bar(metrics.mel_l1_256, 1.0)}")
    print(f"  ├─ Medium (512):  {metrics.mel_l1_512:.4f}  {metric_bar(metrics.mel_l1_512, 1.0)}")
    print(f"  └─ Coarse (1024): {metrics.mel_l1_1024:.4f}  {metric_bar(metrics.mel_l1_1024, 1.0)}")
    
    # Spectral
    print(f"\n  {Colors.BOLD}📈 SPECTRAL METRICS{Colors.RESET}")
    print(f"  ├─ Spectral Conv: {metrics.spectral_convergence:.4f}  {metric_bar(metrics.spectral_convergence, 1.0)}")
    print(f"  └─ Log Spectral:  {metrics.log_spectral_distance:.4f}  {metric_bar(metrics.log_spectral_distance, 2.0)}")
    
    # Amplitude
    print(f"\n  {Colors.BOLD}🔊 AMPLITUDE/DYNAMICS{Colors.RESET}")
    amp_quality = '🟢' if 0.8 < metrics.amplitude_ratio < 1.2 else '🟡' if 0.6 < metrics.amplitude_ratio < 1.4 else '🔴'
    print(f"  ├─ Amplitude:     {metrics.amplitude_ratio:.2%}  {amp_quality}")
    print(f"  ├─ RMS Ratio:     {metrics.rms_ratio:.2%}")
    print(f"  ├─ Peak Ratio:    {metrics.peak_ratio:.2%}")
    print(f"  └─ DR Diff:       {metrics.dynamic_range_diff:.1f} dB")
    
    # Overall
    print(f"\n  {Colors.BOLD}{'─'*50}{Colors.RESET}")
    overall_emoji = '🟢' if metrics.overall_score < 0.3 else '🟡' if metrics.overall_score < 0.5 else '🔴'
    print(f"  {Colors.BOLD}OVERALL SCORE: {metrics.overall_score:.4f}{Colors.RESET}  {overall_emoji}")
    print(f"  {Colors.DIM}(Lower is better, weighted: 40% Mel, 20% Spectral, 20% Cosine, 10% Amp, 10% LSD){Colors.RESET}")
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    step_str = f"step{info.step}"
    audio_name = Path(audio_path).stem
    
    # Save audio files
    orig_path = output_path / f"{audio_name}_{step_str}_original.wav"
    recon_path = output_path / f"{audio_name}_{step_str}_reconstructed.wav"
    
    torchaudio.save(str(orig_path), audio.unsqueeze(0), info.sample_rate)
    torchaudio.save(str(recon_path), reconstructed.unsqueeze(0), info.sample_rate)
    
    # Save metrics JSON
    metrics_path = output_path / f"{audio_name}_{step_str}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    
    print(subheader("Saved outputs"))
    print(f"  ├─ {orig_path}")
    print(f"  ├─ {recon_path}")
    print(f"  └─ {metrics_path}")
    
    return metrics


def compare_checkpoints(checkpoint_dir: str, audio_path: str, device: str = "cuda"):
    """Compare multiple checkpoints on same audio."""
    
    print(header("CHECKPOINT COMPARISON"))
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), 
                        key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0)
    
    if not checkpoints:
        print(f"  {Colors.RED}No checkpoints found in {checkpoint_dir}{Colors.RESET}")
        return
    
    print(f"\n  Found {len(checkpoints)} checkpoints")
    print(f"  Audio: {audio_path}")
    
    # Load audio once
    audio, sr = torchaudio.load(audio_path)
    
    results = []
    
    print(f"\n  {Colors.BOLD}{'Step':>8} {'Mel L1':>10} {'Cosine':>10} {'Amp':>10} {'Overall':>10}{Colors.RESET}")
    print(f"  {'-'*50}")
    
    for cp in checkpoints:
        try:
            encoder, decoder, audio_processor, info = load_checkpoint(str(cp), device)
            
            # Resample if needed
            audio_proc = audio.clone()
            if sr != info.sample_rate:
                audio_proc = torchaudio.functional.resample(audio_proc, sr, info.sample_rate)
            audio_proc = audio_proc.mean(dim=0)
            
            reconstructed = reconstruct_audio(encoder, decoder, audio_processor, audio_proc, device)
            metrics = compute_metrics(audio_proc, reconstructed, info.sample_rate)
            
            results.append({
                "step": info.step,
                "metrics": asdict(metrics)
            })
            
            # Color coding
            mel_color = Colors.GREEN if metrics.mel_l1 < 0.4 else Colors.YELLOW if metrics.mel_l1 < 0.6 else Colors.RED
            cos_color = Colors.GREEN if metrics.mel_cosine_sim > 0.9 else Colors.YELLOW if metrics.mel_cosine_sim > 0.7 else Colors.RED
            amp_color = Colors.GREEN if 0.8 < metrics.amplitude_ratio < 1.2 else Colors.YELLOW
            overall_color = Colors.GREEN if metrics.overall_score < 0.3 else Colors.YELLOW if metrics.overall_score < 0.5 else Colors.RED
            
            print(f"  {info.step:>8,} {mel_color}{metrics.mel_l1:>10.4f}{Colors.RESET} "
                  f"{cos_color}{metrics.mel_cosine_sim:>10.4f}{Colors.RESET} "
                  f"{amp_color}{metrics.amplitude_ratio:>9.0%}{Colors.RESET} "
                  f"{overall_color}{metrics.overall_score:>10.4f}{Colors.RESET}")
            
            # Free memory
            del encoder, decoder
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  {info.step:>8} {Colors.RED}Error: {e}{Colors.RESET}")
    
    # Show improvement
    if len(results) >= 2:
        first = results[0]["metrics"]
        last = results[-1]["metrics"]
        
        mel_improvement = (first["mel_l1"] - last["mel_l1"]) / first["mel_l1"] * 100
        cos_improvement = (last["mel_cosine_sim"] - first["mel_cosine_sim"]) / (1 - first["mel_cosine_sim"] + 1e-8) * 100
        amp_improvement = abs(1 - last["amplitude_ratio"]) < abs(1 - first["amplitude_ratio"])
        
        print(f"\n  {Colors.BOLD}📈 IMPROVEMENT (first → last){Colors.RESET}")
        print(f"  ├─ Mel L1:    {'+' if mel_improvement > 0 else ''}{mel_improvement:.1f}%")
        print(f"  ├─ Cosine:    {'+' if cos_improvement > 0 else ''}{cos_improvement:.1f}%")
        print(f"  └─ Amplitude: {'✓ Improved' if amp_improvement else '✗ Worsened'}")
    
    # Save comparison
    output_path = Path("analysis_output") / "comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Saved: {output_path}")


def list_test_audio(data_dir: str = "data", limit: int = 20) -> List[str]:
    """List available test audio files."""
    data_path = Path(data_dir)
    audio_files = []
    
    for pattern in ["**/*.wav", "**/*.flac", "**/*.mp3"]:
        for f in data_path.glob(pattern):
            audio_files.append(str(f))
            if len(audio_files) >= limit:
                break
        if len(audio_files) >= limit:
            break
    
    return sorted(audio_files)


def interactive_menu(device: str = "cuda"):
    """Interactive mode."""
    
    print(header("SUPERTONIC V2 - CHECKPOINT ANALYZER"))
    print(f"\n  Device: {Colors.CYAN}{device}{Colors.RESET}")
    
    checkpoint_dir = Path("checkpoints/autoencoder")
    current_checkpoint = None
    
    while True:
        print(f"\n{'─'*50}")
        print(f"  {Colors.BOLD}MENU{Colors.RESET}")
        print(f"  1. List checkpoints")
        print(f"  2. Analyze checkpoint + audio")
        print(f"  3. Compare all checkpoints")
        print(f"  4. List test audio files")
        print(f"  0. Exit")
        print(f"{'─'*50}")
        
        choice = input("  Choice: ").strip()
        
        if choice == "0":
            print(f"\n  {Colors.CYAN}Goodbye! 👋{Colors.RESET}\n")
            break
            
        elif choice == "1":
            checkpoints = sorted(checkpoint_dir.glob("*.pt"),
                               key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0)
            print(f"\n  {Colors.BOLD}Found {len(checkpoints)} checkpoints:{Colors.RESET}")
            for i, cp in enumerate(checkpoints):
                step = cp.stem.split("_")[-1]
                print(f"    [{i:>2}] {cp.name}")
                
        elif choice == "2":
            # Select checkpoint
            checkpoints = sorted(checkpoint_dir.glob("*.pt"),
                               key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0)
            print(f"\n  {Colors.BOLD}Checkpoints:{Colors.RESET}")
            for i, cp in enumerate(checkpoints):
                print(f"    [{i:>2}] {cp.name}")
            
            idx = input("  Checkpoint index (or path): ").strip()
            try:
                if idx.isdigit():
                    cp_path = str(checkpoints[int(idx)])
                else:
                    cp_path = idx
            except:
                print(f"  {Colors.RED}Invalid selection{Colors.RESET}")
                continue
            
            # Select audio
            audio_files = list_test_audio()
            print(f"\n  {Colors.BOLD}Sample audio files:{Colors.RESET}")
            for i, af in enumerate(audio_files[:10]):
                print(f"    [{i:>2}] {af}")
            
            audio_idx = input("  Audio index (or path): ").strip()
            try:
                if audio_idx.isdigit():
                    audio_path = audio_files[int(audio_idx)]
                else:
                    audio_path = audio_idx
            except:
                print(f"  {Colors.RED}Invalid selection{Colors.RESET}")
                continue
            
            # Analyze
            analyze_single(cp_path, audio_path, device=device)
            
        elif choice == "3":
            audio_files = list_test_audio()
            print(f"\n  {Colors.BOLD}Sample audio files:{Colors.RESET}")
            for i, af in enumerate(audio_files[:10]):
                print(f"    [{i:>2}] {af}")
            
            audio_idx = input("  Audio index (or path): ").strip()
            try:
                if audio_idx.isdigit():
                    audio_path = audio_files[int(audio_idx)]
                else:
                    audio_path = audio_idx
            except:
                print(f"  {Colors.RED}Invalid selection{Colors.RESET}")
                continue
            
            compare_checkpoints(str(checkpoint_dir), audio_path, device)
            
        elif choice == "4":
            audio_files = list_test_audio(limit=50)
            print(f"\n  {Colors.BOLD}Found {len(audio_files)} audio files:{Colors.RESET}")
            for i, af in enumerate(audio_files):
                print(f"    [{i:>2}] {af}")
        
        else:
            print(f"  {Colors.RED}Unknown option{Colors.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze TTS autoencoder checkpoints")
    parser.add_argument("--checkpoint", "-c", type=str, help="Path to checkpoint")
    parser.add_argument("--audio", "-a", type=str, help="Path to test audio")
    parser.add_argument("--compare", type=str, help="Directory of checkpoints to compare")
    parser.add_argument("--output", "-o", type=str, default="analysis_output", help="Output directory")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    if args.interactive or (not args.checkpoint and not args.compare):
        interactive_menu(device)
    elif args.compare:
        if not args.audio:
            print("Error: --audio required for comparison")
            return
        compare_checkpoints(args.compare, args.audio, device)
    elif args.checkpoint:
        if not args.audio:
            print("Error: --audio required")
            return
        analyze_single(args.checkpoint, args.audio, args.output, device)


if __name__ == "__main__":
    main()
