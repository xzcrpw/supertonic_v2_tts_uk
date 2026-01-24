"""
Speech Autoencoder - Vocos-based архітектура для Supertonic v2

Кодує аудіо в низькорозмірний латентний простір та реконструює 44.1kHz waveform.

Архітектура (~47M параметрів):
- Latent Encoder: mel(228) → Conv+BN(512) → 10 ConvNeXt → Linear+LN(24)
- Latent Decoder: latent(24) → Conv+BN(512) → 10 dilated ConvNeXt → iSTFT
- Discriminators: MPD (periods [2,3,5,7,11]) + MRD (FFT [512,1024,2048])

Latent Space:
- 24-dimensional continuous space (не VQ-VAE!)
- Temporal compression Kc=6 (6 фреймів → 1 вектор 144-dim)

Референс: Vocos paper + Supertonic v2 paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict
from einops import rearrange

from supertonic.models.convnext import ConvNeXtBlock, ConvNeXtStack, LayerNorm1d


class MelSpectrogram(nn.Module):
    """
    Mel Spectrogram extractor для Supertonic v2.

    Конфігурація:
    - sample_rate: 44100
    - n_fft: 2048
    - hop_length: 512
    - n_mels: 228

    Frame timing:
    - FFT window: 2048/44100 = 46.43ms
    - Hop: 512/44100 = 11.61ms
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        center: bool = True,
        power: float = 1.0,  # 1 = magnitude, 2 = power
        normalized: bool = False,
        norm: str = "slaney",
        mel_scale: str = "slaney"
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.center = center
        self.power = power
        self.normalized = normalized

        # Створюємо mel filterbank
        mel_fb = self._create_mel_filterbank(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            norm=norm
        )
        self.register_buffer("mel_fb", mel_fb)

        # Hann window
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def _create_mel_filterbank(
        self,
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        f_min: float,
        f_max: float,
        norm: str
    ) -> torch.Tensor:
        """Створює mel filterbank matrix."""
        # Mel scale conversion
        def hz_to_mel(f):
            return 2595 * math.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        # FFT bins
        n_freqs = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

        # Mel points
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])

        # Create filterbank
        mel_fb = torch.zeros(n_mels, n_freqs)

        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            # Rising edge
            rising = (fft_freqs - left) / (center - left)
            # Falling edge
            falling = (right - fft_freqs) / (right - center)

            mel_fb[i] = torch.maximum(
                torch.zeros_like(fft_freqs),
                torch.minimum(rising, falling)
            )

        # Slaney normalization
        if norm == "slaney":
            enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
            mel_fb *= enorm.unsqueeze(1)

        return mel_fb

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Обчислює mel spectrogram.

        Args:
            audio: [B, T] або [B, 1, T]

        Returns:
            mel: [B, n_mels, T_frames]
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # STFT
        if self.center:
            pad_amount = self.n_fft // 2
            audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')

        # Manual STFT для сумісності
        # Unfold audio into frames
        frames = audio.unfold(1, self.n_fft, self.hop_length)  # [B, T_frames, n_fft]

        # Apply window
        frames = frames * self.window

        # FFT
        spec = torch.fft.rfft(frames, dim=-1)  # [B, T_frames, n_fft//2+1]
        spec = spec.transpose(1, 2)  # [B, n_fft//2+1, T_frames]

        # Magnitude
        if self.power == 1:
            spec_mag = spec.abs()
        else:
            spec_mag = spec.abs().pow(self.power)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_fb, spec_mag)  # [B, n_mels, T_frames]

        # Log mel
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel


class LatentEncoder(nn.Module):
    """
    Latent Encoder - кодує mel spectrogram в 24-dim латентний простір.

    Архітектура:
    1. Conv1d(228 → 512) + BatchNorm
    2. 10 ConvNeXt blocks (intermediate: 2048, kernel: 7)
    3. Linear(512 → 24) + LayerNorm

    Output: 24-dimensional latent vectors
    """

    def __init__(
        self,
        input_dim: int = 228,
        hidden_dim: int = 512,
        output_dim: int = 24,
        num_blocks: int = 10,
        kernel_size: int = 7,
        intermediate_mult: int = 4,
        gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial projection: mel → hidden
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.input_norm = nn.BatchNorm1d(hidden_dim)

        # ConvNeXt stack
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_blocks,
            intermediate_dim=hidden_dim * intermediate_mult,
            kernel_size=kernel_size,
            dilations=None,  # Всі dilation = 1
            causal=False,
            gradient_checkpointing=gradient_checkpointing
        )

        # Output projection: hidden → latent
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram to latent vectors.

        Args:
            mel: Mel spectrogram [B, n_mels, T]

        Returns:
            latent: Latent vectors [B, latent_dim, T]
        """
        # Initial projection
        x = self.input_conv(mel)  # [B, hidden, T]
        x = self.input_norm(x)

        # ConvNeXt blocks
        x = self.convnext(x)  # [B, hidden, T]

        # Output projection
        x = x.transpose(1, 2)  # [B, T, hidden]
        x = self.output_linear(x)  # [B, T, latent]
        x = self.output_norm(x)
        x = x.transpose(1, 2)  # [B, latent, T]

        return x


class ISTFTHead(nn.Module):
    """
    iSTFT Head для реконструкції waveform з frame-level features.

    Генерує STFT magnitude та phase, потім застосовує iSTFT.

    Args:
        input_dim: Input feature dimension
        n_fft: FFT size
        hop_length: Hop size
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1

        # Project to magnitude and phase
        self.mag_proj = nn.Linear(input_dim, self.n_freqs)
        self.phase_proj = nn.Linear(input_dim, self.n_freqs)

        # Hann window for iSTFT
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert frame features to waveform via iSTFT.

        Args:
            x: Frame features [B, T, D]

        Returns:
            audio: Waveform [B, T_audio]
        """
        batch_size, num_frames, _ = x.shape

        # Predict magnitude (exp для positive values)
        # CRITICAL: Clamp raw values to prevent magnitude explosion
        # exp(5) ≈ 148, exp(4) ≈ 55 - reasonable max magnitudes for audio
        mag_raw = self.mag_proj(x)
        mag = mag_raw.clamp(min=-20.0, max=5.0).exp()  # [B, T, n_freqs]

        # Predict phase (normalized to [-π, π])
        phase = self.phase_proj(x)  # [B, T, n_freqs]
        phase = torch.tanh(phase) * math.pi

        # Complex STFT
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        spec = torch.complex(real, imag)  # [B, T, n_freqs]
        spec = spec.transpose(1, 2)  # [B, n_freqs, T]

        # iSTFT
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
            return_complex=False
        )

        return audio


class LatentDecoder(nn.Module):
    """
    Latent Decoder - декодує латенти в waveform.

    Архітектура:
    1. Conv1d(24 → 512) + BatchNorm
    2. 10 dilated ConvNeXt blocks (dilations: [1,2,4,1,2,4,1,1,1,1])
    3. iSTFT head → waveform

    Всі конволюції КАУЗАЛЬНІ для streaming підтримки.
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 512,
        num_blocks: int = 10,
        kernel_size: int = 7,
        intermediate_mult: int = 4,
        dilations: Optional[List[int]] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        causal: bool = True,
        gradient_checkpointing: bool = False
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initial projection: latent → hidden
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.input_norm = nn.BatchNorm1d(hidden_dim)

        # Dilated ConvNeXt stack (causal!)
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_blocks,
            intermediate_dim=hidden_dim * intermediate_mult,
            kernel_size=kernel_size,
            dilations=dilations,
            causal=causal,
            gradient_checkpointing=gradient_checkpointing
        )

        # iSTFT head
        self.istft_head = ISTFTHead(
            input_dim=hidden_dim,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to waveform.

        Args:
            latent: Latent vectors [B, latent_dim, T]

        Returns:
            audio: Waveform [B, T_audio]
        """
        # Initial projection
        x = self.input_conv(latent)  # [B, hidden, T]
        x = self.input_norm(x)

        # ConvNeXt blocks
        x = self.convnext(x)  # [B, hidden, T]

        # iSTFT
        x = x.transpose(1, 2)  # [B, T, hidden]
        audio = self.istft_head(x)  # [B, T_audio]
        
        # CRITICAL: Clamp output to prevent clipping/distortion
        audio = torch.tanh(audio)

        return audio


class PeriodDiscriminator(nn.Module):
    """
    Period Discriminator для GAN training.

    Аналізує аудіо з періодичним sampling для виявлення
    періодичних артефактів.

    Args:
        period: Sampling period
        channels: Number of channels per layer
    """

    def __init__(
        self,
        period: int,
        channels: List[int] = [32, 128, 512, 1024, 1024]
    ):
        super().__init__()

        self.period = period

        layers = []
        in_ch = 1

        for i, out_ch in enumerate(channels):
            stride = 3 if i < len(channels) - 1 else 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch,
                        kernel_size=(5, 1),
                        stride=(stride, 1),
                        padding=(2, 0)
                    ),
                    nn.LeakyReLU(0.1)
                )
            )
            in_ch = out_ch

        # Final conv
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 1), padding=(1, 0))
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Audio [B, 1, T]

        Returns:
            output: Discriminator output
            features: Intermediate features for feature matching
        """
        batch_size, _, t = x.shape

        # Pad to multiple of period
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode='reflect')
            t = t + pad

        # Reshape to 2D: [B, 1, T/p, p]
        x = x.view(batch_size, 1, t // self.period, self.period)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return x.flatten(1, -1), features[:-1]


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) для Supertonic v2.

    Periods: [2, 3, 5, 7, 11] - покривають різні частотні компоненти.
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11]
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period=p) for p in periods
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through all period discriminators.

        Args:
            x: Audio [B, T] або [B, 1, T]

        Returns:
            outputs: List of discriminator outputs
            features: List of feature lists
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        outputs = []
        features = []

        for disc in self.discriminators:
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)

        return outputs, features


class ScaleDiscriminator(nn.Module):
    """
    Scale Discriminator для Multi-Resolution Discriminator.

    Args:
        channels: Number of channels per layer
    """

    def __init__(
        self,
        channels: List[int] = [128, 128, 256, 512, 1024, 1024, 1024]
    ):
        super().__init__()

        layers = []
        in_ch = 1

        for i, out_ch in enumerate(channels):
            if i == 0:
                layer = nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=15, stride=1, padding=7
                )
            elif i == len(channels) - 1:
                layer = nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=5, stride=1, padding=2
                )
            else:
                layer = nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=41, stride=4, padding=20, groups=4
                )

            layers.append(nn.Sequential(layer, nn.LeakyReLU(0.1)))
            in_ch = out_ch

        # Final conv
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=3, padding=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass."""
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return x.flatten(1, -1), features[:-1]


class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-Resolution Discriminator (MRD) для Supertonic v2.

    FFT sizes: [512, 1024, 2048] - різні temporal resolutions.
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048]
    ):
        super().__init__()

        self.fft_sizes = fft_sizes

        # Pooling для downsampling
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=fft // 256, stride=fft // 256)
            for fft in fft_sizes
        ])

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in fft_sizes
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass through all scale discriminators."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        outputs = []
        features = []

        for pool, disc in zip(self.pools, self.discriminators):
            x_pooled = pool(x)
            out, feat = disc(x_pooled)
            outputs.append(out)
            features.append(feat)

        return outputs, features


class SpeechAutoencoder(nn.Module):
    """
    Speech Autoencoder - головний модуль для кодування/декодування аудіо.

    Архітектура (~47M параметрів):
    - MelSpectrogram: audio → mel(228)
    - LatentEncoder: mel → latent(24)
    - LatentDecoder: latent → audio

    Для GAN training:
    - MultiPeriodDiscriminator (MPD)
    - MultiResolutionDiscriminator (MRD)

    Temporal compression Kc=6:
    - Стек 6 фреймів → 144-dim вектор (для flow-matching)

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop size
        n_mels: Number of mel bands
        latent_dim: Latent space dimension
        temporal_compression: Kc factor
        hidden_dim: Hidden dimension for encoder/decoder
        num_blocks: Number of ConvNeXt blocks
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        latent_dim: int = 24,
        temporal_compression: int = 6,
        hidden_dim: int = 512,
        num_encoder_blocks: int = 10,
        num_decoder_blocks: int = 10,
        decoder_dilations: Optional[List[int]] = None
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.temporal_compression = temporal_compression
        self.compressed_dim = latent_dim * temporal_compression  # 24 * 6 = 144

        if decoder_dilations is None:
            decoder_dilations = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

        # Mel spectrogram extractor
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Encoder
        self.encoder = LatentEncoder(
            input_dim=n_mels,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_blocks=num_encoder_blocks
        )

        # Decoder
        self.decoder = LatentDecoder(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_decoder_blocks,
            dilations=decoder_dilations,
            n_fft=n_fft,
            hop_length=hop_length,
            causal=True
        )

        # Discriminators (для GAN training)
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

        # Statistics для латентної нормалізації
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent vectors.

        Args:
            audio: Waveform [B, T]

        Returns:
            latent: [B, latent_dim, T_frames]
        """
        mel = self.mel_spec(audio)
        latent = self.encoder(mel)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to audio.

        Args:
            latent: [B, latent_dim, T_frames]

        Returns:
            audio: [B, T]
        """
        return self.decoder(latent)

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode → decode.

        Args:
            audio: Waveform [B, T]

        Returns:
            Dict with 'latent', 'audio_recon', 'mel', 'mel_recon'
        """
        # Encode
        mel = self.mel_spec(audio)
        latent = self.encoder(mel)

        # Decode
        audio_recon = self.decoder(latent)

        # Mel of reconstructed audio
        mel_recon = self.mel_spec(audio_recon)

        return {
            'latent': latent,
            'audio_recon': audio_recon,
            'mel': mel,
            'mel_recon': mel_recon
        }

    def compress_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Temporal compression: stack Kc frames into one vector.

        Args:
            latent: [B, C, T] where C=24

        Returns:
            compressed: [B, C*Kc, T/Kc] where C*Kc=144
        """
        batch_size, c, t = latent.shape
        kc = self.temporal_compression

        # Pad to multiple of Kc
        if t % kc != 0:
            pad = kc - (t % kc)
            latent = F.pad(latent, (0, pad))
            t = t + pad

        # Reshape: [B, C, T] → [B, C, T/Kc, Kc] → [B, C*Kc, T/Kc]
        latent = latent.view(batch_size, c, t // kc, kc)
        latent = latent.permute(0, 1, 3, 2)  # [B, C, Kc, T/Kc]
        latent = latent.reshape(batch_size, c * kc, t // kc)

        return latent

    def decompress_latent(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Temporal decompression: unstack Kc frames.

        Args:
            compressed: [B, C*Kc, T/Kc] where C*Kc=144

        Returns:
            latent: [B, C, T] where C=24
        """
        batch_size, ckc, t_compressed = compressed.shape
        kc = self.temporal_compression
        c = ckc // kc

        # Reshape: [B, C*Kc, T/Kc] → [B, C, Kc, T/Kc] → [B, C, T]
        compressed = compressed.view(batch_size, c, kc, t_compressed)
        latent = compressed.permute(0, 1, 3, 2)  # [B, C, T/Kc, Kc]
        latent = latent.reshape(batch_size, c, t_compressed * kc)

        return latent

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalize latent using precomputed statistics."""
        mean = self.latent_mean.view(1, -1, 1)
        std = self.latent_std.view(1, -1, 1)
        return (latent - mean) / (std + 1e-8)

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Denormalize latent using precomputed statistics."""
        mean = self.latent_mean.view(1, -1, 1)
        std = self.latent_std.view(1, -1, 1)
        return latent * std + mean

    def update_latent_statistics(self, latents: torch.Tensor):
        """Update running statistics for latent normalization."""
        # Channel-wise mean and std
        mean = latents.mean(dim=(0, 2))
        std = latents.std(dim=(0, 2))

        # EMA update
        momentum = 0.1
        self.latent_mean = (1 - momentum) * self.latent_mean + momentum * mean
        self.latent_std = (1 - momentum) * self.latent_std + momentum * std

    def discriminate(
        self,
        audio: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Apply discriminators.

        Returns:
            mpd_outputs, mpd_features, mrd_outputs, mrd_features
        """
        mpd_out, mpd_feat = self.mpd(audio)
        mrd_out, mrd_feat = self.mrd(audio)
        return mpd_out, mpd_feat, mrd_out, mrd_feat


# ============================================================================
# Unit tests
# ============================================================================

def _test_mel_spectrogram():
    """Тест MelSpectrogram."""
    print("Testing MelSpectrogram...")

    mel_spec = MelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=228
    )

    # Random audio
    batch_size = 2
    duration = 2.0  # seconds
    audio = torch.randn(batch_size, int(44100 * duration))

    mel = mel_spec(audio)

    expected_frames = int(44100 * duration / 512) + 1
    print(f"  Audio: {audio.shape}")
    print(f"  Mel: {mel.shape}")
    print(f"  Expected frames: ~{expected_frames}")

    assert mel.shape[1] == 228, f"Expected 228 mels, got {mel.shape[1]}"
    print("MelSpectrogram tests passed! ✓\n")


def _test_latent_encoder():
    """Тест LatentEncoder."""
    print("Testing LatentEncoder...")

    encoder = LatentEncoder(
        input_dim=228,
        hidden_dim=512,
        output_dim=24,
        num_blocks=10
    )

    batch_size = 2
    seq_len = 100
    mel = torch.randn(batch_size, 228, seq_len)

    latent = encoder(mel)

    assert latent.shape == (batch_size, 24, seq_len)
    print(f"  Mel: {mel.shape} -> Latent: {latent.shape} ✓")

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("LatentEncoder tests passed! ✓\n")


def _test_latent_decoder():
    """Тест LatentDecoder."""
    print("Testing LatentDecoder...")

    decoder = LatentDecoder(
        input_dim=24,
        hidden_dim=512,
        num_blocks=10,
        dilations=[1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
        n_fft=2048,
        hop_length=512,
        causal=True
    )

    batch_size = 2
    seq_len = 100
    latent = torch.randn(batch_size, 24, seq_len)

    audio = decoder(latent)

    expected_len = (seq_len - 1) * 512
    print(f"  Latent: {latent.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Expected audio length: ~{expected_len}")

    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("LatentDecoder tests passed! ✓\n")


def _test_discriminators():
    """Тест дискримінаторів."""
    print("Testing Discriminators...")

    mpd = MultiPeriodDiscriminator(periods=[2, 3, 5, 7, 11])
    mrd = MultiResolutionDiscriminator(fft_sizes=[512, 1024, 2048])

    batch_size = 2
    audio = torch.randn(batch_size, 44100)  # 1 second

    mpd_out, mpd_feat = mpd(audio)
    print(f"  MPD outputs: {len(mpd_out)}, shapes: {[o.shape for o in mpd_out]}")

    mrd_out, mrd_feat = mrd(audio)
    print(f"  MRD outputs: {len(mrd_out)}, shapes: {[o.shape for o in mrd_out]}")

    mpd_params = sum(p.numel() for p in mpd.parameters())
    mrd_params = sum(p.numel() for p in mrd.parameters())
    print(f"  MPD parameters: {mpd_params:,}")
    print(f"  MRD parameters: {mrd_params:,}")

    print("Discriminator tests passed! ✓\n")


def _test_speech_autoencoder():
    """Тест повного Speech Autoencoder."""
    print("Testing SpeechAutoencoder...")

    autoencoder = SpeechAutoencoder(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=228,
        latent_dim=24,
        temporal_compression=6,
        hidden_dim=512,
        num_encoder_blocks=10,
        num_decoder_blocks=10
    )

    batch_size = 2
    duration = 1.0
    audio = torch.randn(batch_size, int(44100 * duration))

    # Full forward
    output = autoencoder(audio)
    print(f"  Input audio: {audio.shape}")
    print(f"  Latent: {output['latent'].shape}")
    print(f"  Mel: {output['mel'].shape}")
    print(f"  Reconstructed audio: {output['audio_recon'].shape}")

    # Test compression
    latent = output['latent']
    compressed = autoencoder.compress_latent(latent)
    print(f"  Compressed latent: {latent.shape} -> {compressed.shape}")

    decompressed = autoencoder.decompress_latent(compressed)
    print(f"  Decompressed: {compressed.shape} -> {decompressed.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in autoencoder.parameters())
    encoder_params = sum(p.numel() for p in autoencoder.encoder.parameters())
    decoder_params = sum(p.numel() for p in autoencoder.decoder.parameters())

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")

    print("\nSpeechAutoencoder tests passed! ✓\n")


if __name__ == "__main__":
    _test_mel_spectrogram()
    _test_latent_encoder()
    _test_latent_decoder()
    _test_discriminators()
    _test_speech_autoencoder()
    print("All Speech Autoencoder tests passed! ✓")
