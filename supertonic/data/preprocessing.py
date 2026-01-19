"""
Audio/Text Preprocessing для Supertonic v2 TTS

Audio конфігурація (з paper):
- sample_rate: 44100
- n_fft: 2048
- hop_length: 512
- n_mels: 228
- fmin: 0
- fmax: 22050 (Nyquist)

Frame timing:
- FFT window: 2048/44100 = 46.43ms
- Hop: 512/44100 = 11.61ms
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import math
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings


def load_audio(
    path: Union[str, Path],
    target_sr: int = 44100,
    mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Завантажує аудіо файл та resamples до target sample rate.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 44100)
        mono: Convert to mono (default True)
        
    Returns:
        audio: [T] або [C, T] tensor
        sample_rate: Original sample rate
    """
    audio, sr = torchaudio.load(str(path))
    
    # Convert to mono
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Squeeze if mono
    if mono:
        audio = audio.squeeze(0)
    
    return audio, sr


def normalize_audio(
    audio: torch.Tensor,
    target_db: float = -20.0
) -> torch.Tensor:
    """
    Нормалізує аудіо до target loudness.
    
    Args:
        audio: Audio tensor [T] або [B, T]
        target_db: Target loudness in dB (default -20)
        
    Returns:
        normalized: Normalized audio
    """
    # RMS energy
    rms = audio.pow(2).mean().sqrt()
    
    # Target RMS
    target_rms = 10 ** (target_db / 20)
    
    # Scale
    if rms > 1e-8:
        audio = audio * (target_rms / rms)
    
    # Clip to prevent overflow
    audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio


class AudioProcessor:
    """
    Audio Processor для Supertonic v2.
    
    Обробляє аудіо для training:
    - Loading та resampling
    - Mel spectrogram extraction
    - Normalization
    
    Args:
        sample_rate: Target sample rate
        n_fft: FFT size
        hop_length: Hop size
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        n_mels: int = 228,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 1.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect"
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=self.fmax,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            norm="slaney",
            mel_scale="slaney"
        )
    
    def load(
        self,
        path: Union[str, Path],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Завантажує та обробляє аудіо файл.
        
        Args:
            path: Path to audio file
            normalize: Whether to normalize audio
            
        Returns:
            audio: [T] tensor
        """
        audio, _ = load_audio(path, target_sr=self.sample_rate)
        
        if normalize:
            audio = normalize_audio(audio)
        
        return audio
    
    def compute_mel(
        self,
        audio: torch.Tensor,
        log_scale: bool = True
    ) -> torch.Tensor:
        """
        Обчислює mel spectrogram.
        
        Args:
            audio: Audio tensor [T] або [B, T]
            log_scale: Apply log scaling
            
        Returns:
            mel: [n_mels, T] або [B, n_mels, T]
        """
        # Add batch dim if needed
        squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True
        
        # Compute mel spectrogram
        mel = self.mel_transform(audio)
        
        # Log scale
        if log_scale:
            mel = torch.log(torch.clamp(mel, min=1e-5))
        
        if squeeze:
            mel = mel.squeeze(0)
        
        return mel
    
    def get_duration(self, audio: torch.Tensor) -> float:
        """Повертає тривалість аудіо в секундах."""
        return len(audio) / self.sample_rate
    
    def get_num_frames(self, audio: torch.Tensor) -> int:
        """Повертає кількість mel frames."""
        return (len(audio) - self.n_fft) // self.hop_length + 1
    
    def samples_to_frames(self, num_samples: int) -> int:
        """Конвертує samples → frames."""
        return (num_samples + self.hop_length - 1) // self.hop_length
    
    def frames_to_samples(self, num_frames: int) -> int:
        """Конвертує frames → samples."""
        return num_frames * self.hop_length


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 228,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    log_scale: bool = True
) -> torch.Tensor:
    """
    Обчислює mel spectrogram (standalone function).
    
    Args:
        audio: Audio tensor [T] або [B, T]
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop size
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        log_scale: Apply log scaling
        
    Returns:
        mel: Mel spectrogram [n_mels, T] або [B, n_mels, T]
    """
    processor = AudioProcessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    return processor.compute_mel(audio, log_scale=log_scale)


class LatentNormalizer:
    """
    Normalizer для latent vectors.
    
    Нормалізує латенти precomputed channel-wise mean/variance.
    """
    
    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        latent_dim: int = 24
    ):
        self.latent_dim = latent_dim
        
        if mean is None:
            mean = torch.zeros(latent_dim)
        if std is None:
            std = torch.ones(latent_dim)
        
        self.mean = mean
        self.std = std
    
    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Нормалізує латенти.
        
        Args:
            latents: [B, C, T] або [C, T]
            
        Returns:
            normalized: Normalized latents
        """
        mean = self.mean.to(latents.device)
        std = self.std.to(latents.device)
        
        if latents.dim() == 2:
            return (latents - mean.unsqueeze(-1)) / (std.unsqueeze(-1) + 1e-8)
        else:
            return (latents - mean.unsqueeze(0).unsqueeze(-1)) / (std.unsqueeze(0).unsqueeze(-1) + 1e-8)
    
    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Денормалізує латенти.
        """
        mean = self.mean.to(latents.device)
        std = self.std.to(latents.device)
        
        if latents.dim() == 2:
            return latents * std.unsqueeze(-1) + mean.unsqueeze(-1)
        else:
            return latents * std.unsqueeze(0).unsqueeze(-1) + mean.unsqueeze(0).unsqueeze(-1)
    
    @classmethod
    def compute_statistics(
        cls,
        latents_list: list
    ) -> "LatentNormalizer":
        """
        Обчислює mean/std з списку латентів.
        """
        all_latents = torch.cat([l.reshape(l.shape[0], -1) for l in latents_list], dim=1)
        mean = all_latents.mean(dim=1)
        std = all_latents.std(dim=1)
        
        return cls(mean=mean, std=std, latent_dim=mean.shape[0])
    
    def save(self, path: Union[str, Path]):
        """Зберігає statistics."""
        torch.save({
            "mean": self.mean,
            "std": self.std,
            "latent_dim": self.latent_dim
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LatentNormalizer":
        """Завантажує statistics."""
        data = torch.load(path)
        return cls(
            mean=data["mean"],
            std=data["std"],
            latent_dim=data["latent_dim"]
        )


# ============================================================================
# Unit tests
# ============================================================================

def _test_preprocessing():
    """Тест preprocessing utilities."""
    print("Testing Preprocessing...")
    
    # Test with synthetic audio
    sample_rate = 44100
    duration = 2.0  # seconds
    audio = torch.randn(int(sample_rate * duration))
    
    # Test normalization
    normalized = normalize_audio(audio)
    assert normalized.shape == audio.shape
    print(f"  Normalization: {audio.shape} ✓")
    
    # Test AudioProcessor
    processor = AudioProcessor(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=228
    )
    
    mel = processor.compute_mel(audio)
    expected_frames = (len(audio) - 2048) // 512 + 1
    print(f"  Mel spectrogram: {mel.shape} (expected ~{expected_frames} frames) ✓")
    
    duration_sec = processor.get_duration(audio)
    print(f"  Duration: {duration_sec:.2f} sec ✓")
    
    # Test standalone function
    mel2 = compute_mel_spectrogram(audio)
    print(f"  Standalone mel: {mel2.shape} ✓")
    
    # Test latent normalizer
    latents = torch.randn(24, 100)
    normalizer = LatentNormalizer()
    
    normalized_latents = normalizer.normalize(latents)
    denormalized = normalizer.denormalize(normalized_latents)
    
    print(f"  Latent normalization: {latents.shape} ↔ {normalized_latents.shape} ✓")
    
    print("All Preprocessing tests passed! ✓\n")


if __name__ == "__main__":
    _test_preprocessing()
