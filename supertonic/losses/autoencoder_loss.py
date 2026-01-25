"""
Autoencoder Loss - Multi-resolution reconstruction + GAN losses

Based on SupertonicTTS paper (arXiv:2503.23108v3)

Paper Loss formula (Appendix B.1):
    L_G = λ_recon × L_recon + λ_adv × L_adv + λ_fm × L_fm

Where:
- L_recon: Multi-resolution spectral L1 loss in mel domain
          FFT sizes: [1024, 2048, 4096] @ 44.1kHz
          Mel bands: [64, 128, 128] (different per resolution!)
          λ_recon = 45

- L_adv: LSGAN adversarial loss
         L_adv(G;D) = E[(D(G(x)) - 1)^2]
         λ_adv = 1

- L_fm: Feature matching L1 loss
        L_fm = (1/L) Σ ||phi_l(G(x)) - phi_l(x)||_1
        λ_fm = 0.1

Discriminators (paper uses VERY lightweight - Table 7):
- MPD: Multi-Period Discriminator (periods [2, 3, 5, 7, 11])
       Hidden sizes: [16, 64, 256, 512, 512, 1]
- MRD: Multi-Resolution Discriminator (FFT [512, 1024, 2048])
       All conv layers have hidden_channels = 16 (!)

For 22kHz we scale FFT sizes proportionally:
    [1024, 2048, 4096] @ 44.1kHz -> [512, 1024, 2048] @ 22kHz

Reference: HiFi-GAN, Vocos, SupertonicTTS paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


class MultiResolutionSTFT(nn.Module):
    """
    Multi-Resolution STFT для обчислення mel spectrogram на різних scales.
    
    Paper (Appendix B.1) використовує РІЗНІ n_mels для кожного FFT size:
    - FFT 1024 -> 64 mels
    - FFT 2048 -> 128 mels  
    - FFT 4096 -> 128 mels
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
        sample_rate: int = 22050,
        n_mels: Optional[List[int]] = None  # Paper: different per resolution!
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [s // 4 for s in fft_sizes]
        self.win_sizes = win_sizes or fft_sizes
        self.sample_rate = sample_rate
        
        # Paper uses different n_mels per resolution: [64, 128, 128]
        if n_mels is None:
            self.n_mels_list = [64, 128, 128]  # Paper default
        elif isinstance(n_mels, int):
            self.n_mels_list = [n_mels] * len(fft_sizes)
        else:
            self.n_mels_list = n_mels
        
        # Створюємо mel filterbanks для кожного resolution
        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            n_mels_i = self.n_mels_list[i] if i < len(self.n_mels_list) else self.n_mels_list[-1]
            mel_fb = self._create_mel_filterbank(
                n_fft=fft_size,
                n_mels=n_mels_i,
                sample_rate=sample_rate
            )
            self.register_buffer(f"mel_fb_{i}", mel_fb)
            self.register_buffer(f"n_mels_{i}", torch.tensor(n_mels_i))
            
            # Hann window
            window = torch.hann_window(win_size)
            self.register_buffer(f"window_{i}", window)
    
    def _create_mel_filterbank(
        self,
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ) -> torch.Tensor:
        """Створює mel filterbank."""
        f_max = f_max or sample_rate / 2
        
        def hz_to_mel(f):
            return 2595 * math.log10(1 + f / 700)
        
        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)
        
        n_freqs = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)
        
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])
        
        mel_fb = torch.zeros(n_mels, n_freqs)
        
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            rising = (fft_freqs - left) / (center - left + 1e-8)
            falling = (right - fft_freqs) / (right - center + 1e-8)
            
            mel_fb[i] = torch.maximum(
                torch.zeros_like(fft_freqs),
                torch.minimum(rising, falling)
            )
        
        # Slaney normalization
        enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels] + 1e-8)
        mel_fb *= enorm.unsqueeze(1)
        
        return mel_fb
    
    def forward(
        self,
        audio: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Обчислює mel spectrograms на всіх resolutions.
        
        Args:
            audio: [B, T] або [B, 1, T]
            
        Returns:
            List of mel spectrograms
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        mels = []
        
        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            mel_fb = getattr(self, f"mel_fb_{i}")
            window = getattr(self, f"window_{i}")
            
            # Ensure window is on same device as audio
            if window.device != audio.device:
                window = window.to(audio.device)
            
            # STFT
            spec = torch.stft(
                audio,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                center=True,
                pad_mode='reflect',
                normalized=False,
                onesided=True,
                return_complex=True
            )
            
            # Magnitude
            mag = spec.abs()  # [B, n_freqs, T]
            
            # Ensure mel_fb is on same device as mag
            if mel_fb.device != mag.device:
                mel_fb = mel_fb.to(mag.device)
            
            # Mel
            mel = torch.matmul(mel_fb, mag)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            mels.append(mel)
        
        return mels


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence + Log Magnitude Loss (HiFi-GAN style).
    
    Працює на ЛІНІЙНОМУ спектрі (не Mel!) для кращої якості фази.
    SC = ||mag_real - mag_fake||_F / ||mag_real||_F
    LM = ||log(mag_real) - log(mag_fake)||_1
    
    Це критично для усунення "металевого" звуку!
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [s // 4 for s in fft_sizes]
        self.win_sizes = win_sizes or fft_sizes
        
        # Register windows
        for i, win_size in enumerate(self.win_sizes):
            window = torch.hann_window(win_size)
            self.register_buffer(f"window_{i}", window)
    
    def _stft_magnitude(self, audio: torch.Tensor, fft_size: int, 
                        hop_size: int, win_size: int, window: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        spec = torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        return spec.abs()
    
    def forward(self, real: torch.Tensor, generated: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sc_loss: Spectral Convergence loss
            mag_loss: Log Magnitude loss  
        """
        if real.dim() == 3:
            real = real.squeeze(1)
        if generated.dim() == 3:
            generated = generated.squeeze(1)
            
        # Match lengths
        min_len = min(real.size(-1), generated.size(-1))
        real = real[..., :min_len]
        generated = generated[..., :min_len]
        
        sc_loss = 0.0
        mag_loss = 0.0
        
        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            window = getattr(self, f"window_{i}")
            if window.device != real.device:
                window = window.to(real.device)
            
            mag_real = self._stft_magnitude(real, fft_size, hop_size, win_size, window)
            mag_fake = self._stft_magnitude(generated, fft_size, hop_size, win_size, window)
            
            # Spectral Convergence: Frobenius norm ratio
            sc_loss += torch.norm(mag_real - mag_fake, p="fro") / (torch.norm(mag_real, p="fro") + 1e-9)
            
            # Log Magnitude L1
            log_mag_real = torch.log(mag_real.clamp(min=1e-5))
            log_mag_fake = torch.log(mag_fake.clamp(min=1e-5))
            mag_loss += F.l1_loss(log_mag_fake, log_mag_real)
        
        n = len(self.fft_sizes)
        return sc_loss / n, mag_loss / n


class MultiResolutionMelLoss(nn.Module):
    """
    Multi-Resolution Mel L1 Loss (Paper Appendix B.1).
    
    Paper configuration:
    - FFT sizes: [1024, 2048, 4096] @ 44.1kHz
    - Mel bands: [64, 128, 128] - DIFFERENT per resolution!
    - Hop sizes: FFT/4
    - Window: Hann, size = FFT size
    
    For 22kHz we scale FFT sizes by 0.5:
    - FFT sizes: [512, 1024, 2048]
    - Mel bands: [64, 128, 128]
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],  # Scaled for 22kHz
        sample_rate: int = 22050,
        n_mels: Optional[List[int]] = None  # Paper: [64, 128, 128]
    ):
        super().__init__()
        
        # Paper default: different n_mels per resolution
        if n_mels is None:
            n_mels = [64, 128, 128]
        
        self.stft = MultiResolutionSTFT(
            fft_sizes=fft_sizes,
            sample_rate=sample_rate,
            n_mels=n_mels
        )
    
    def forward(
        self,
        real: torch.Tensor,
        generated: torch.Tensor
    ) -> torch.Tensor:
        """
        Обчислює multi-resolution mel L1 loss.
        
        Args:
            real: Real audio [B, T]
            generated: Generated audio [B, T]
            
        Returns:
            loss: Scalar tensor
        """
        # Вирівнюємо довжини
        min_len = min(real.size(-1), generated.size(-1))
        real = real[..., :min_len]
        generated = generated[..., :min_len]
        
        real_mels = self.stft(real)
        gen_mels = self.stft(generated)
        
        loss = 0.0
        for real_mel, gen_mel in zip(real_mels, gen_mels):
            # Вирівнюємо по часу
            min_t = min(real_mel.size(-1), gen_mel.size(-1))
            loss += F.l1_loss(
                gen_mel[..., :min_t],
                real_mel[..., :min_t]
            )
        
        return loss / len(real_mels)


class GANLoss(nn.Module):
    """
    GAN Loss для adversarial training.
    
    Підтримує:
    - LSGAN (least squares)
    - Hinge loss
    - Vanilla GAN
    """
    
    def __init__(self, loss_type: str = "lsgan"):
        super().__init__()
        
        self.loss_type = loss_type
        
    def discriminator_loss(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Discriminator loss.
        
        Paper (Appendix B.1, Eq. 5):
        L_adv(D;G) = E[(D(G(x)) + 1)² + (D(x) - 1)²]
        
        This uses -1/+1 labels (real=+1, fake=-1), NOT 0/1!
        
        Args:
            real_outputs: List of discriminator outputs for real samples
            fake_outputs: List of discriminator outputs for fake samples
            
        Returns:
            loss: Scalar tensor
        """
        loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            if self.loss_type == "lsgan":
                # Paper Eq. 5: (D(x) - 1)² + (D(G(x)) + 1)²
                # real → +1, fake → -1
                loss += torch.mean((real_out - 1) ** 2) + torch.mean((fake_out + 1) ** 2)
            elif self.loss_type == "hinge":
                loss += torch.mean(F.relu(1 - real_out)) + torch.mean(F.relu(1 + fake_out))
            else:  # vanilla
                loss += F.binary_cross_entropy_with_logits(
                    real_out, torch.ones_like(real_out)
                ) + F.binary_cross_entropy_with_logits(
                    fake_out, torch.zeros_like(fake_out)
                )
        
        return loss / len(real_outputs)
    
    def generator_loss(
        self,
        fake_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Generator adversarial loss.
        
        Args:
            fake_outputs: List of discriminator outputs for generated samples
            
        Returns:
            loss: Scalar tensor
        """
        loss = 0.0
        
        for fake_out in fake_outputs:
            if self.loss_type == "lsgan":
                loss += torch.mean((fake_out - 1) ** 2)
            elif self.loss_type == "hinge":
                loss += -torch.mean(fake_out)
            else:  # vanilla
                loss += F.binary_cross_entropy_with_logits(
                    fake_out, torch.ones_like(fake_out)
                )
        
        return loss / len(fake_outputs)


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss.
    
    L1 loss на intermediate features discriminator'а.
    Стабілізує GAN training.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Обчислює feature matching loss.
        
        Args:
            real_features: List of feature lists from discriminators (real)
            fake_features: List of feature lists from discriminators (fake)
            
        Returns:
            loss: Scalar tensor
        """
        loss = 0.0
        num_features = 0
        
        for real_feat_list, fake_feat_list in zip(real_features, fake_features):
            for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
                loss += F.l1_loss(fake_feat, real_feat.detach())
                num_features += 1
        
        return loss / max(num_features, 1)


class AutoencoderLoss(nn.Module):
    """
    Loss для Speech Autoencoder (точно як в SupertonicTTS paper).
    
    Paper formula (Appendix B.1):
        L_G = λ_recon × L_recon + λ_adv × L_adv + λ_fm × L_fm
    
    Where:
        L_recon: Multi-resolution spectral L1 loss in mel domain
        L_adv: LSGAN adversarial loss  
        L_fm: Feature matching L1 loss
    
    Paper weights:
        λ_recon = 45
        λ_adv = 1
        λ_fm = 0.1
    
    Paper config (for 44.1kHz):
        FFT sizes: [1024, 2048, 4096]
        Mel bands: [64, 128, 128]
        
    For 22kHz we scale FFT sizes by 0.5.
    """
    
    def __init__(
        self,
        lambda_recon: float = 45.0,   # Paper: λ_recon = 45
        lambda_adv: float = 1.0,      # Paper: λ_adv = 1
        lambda_fm: float = 0.1,       # Paper: λ_fm = 0.1
        fft_sizes: List[int] = [512, 1024, 2048],  # Scaled for 22kHz
        n_mels: List[int] = [64, 128, 128],        # Paper: different per resolution
        sample_rate: int = 22050,
        gan_type: str = "lsgan",
        # Legacy params (kept for config compat)
        lambda_mel: float = None,     # Alias for lambda_recon
        lambda_wave: float = None,    # Not used in paper
        lambda_sc: float = None,
        lambda_mag: float = None,
    ):
        super().__init__()
        
        # Handle legacy param names
        if lambda_mel is not None:
            lambda_recon = lambda_mel
        
        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        
        # Mel Loss (paper's reconstruction objective)
        self.mel_loss = MultiResolutionMelLoss(
            fft_sizes=fft_sizes,
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        self.gan_loss = GANLoss(loss_type=gan_type)
        self.fm_loss = FeatureMatchingLoss()
    
    def generator_loss(
        self,
        real_audio: torch.Tensor,
        generated_audio: torch.Tensor,
        disc_fake_outputs: List[torch.Tensor],
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]],
        use_adv: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generator loss (Paper Appendix B.1).
        
        L_G = λ_recon × L_recon + λ_adv × L_adv + λ_fm × L_fm
        
        Args:
            real_audio: Real audio [B, T]
            generated_audio: Generated audio [B, T]
            disc_fake_outputs: Discriminator outputs for generated audio
            real_features: Discriminator features for real audio
            fake_features: Discriminator features for generated audio
            use_adv: Whether to use adversarial loss (False during warmup)
            
        Returns:
            Dict with loss components and total loss
        """
        # Match lengths
        min_len = min(real_audio.size(-1), generated_audio.size(-1))
        real_audio_crop = real_audio[..., :min_len]
        generated_audio_crop = generated_audio[..., :min_len]
        
        # L_recon: Multi-resolution mel L1 loss (paper's main objective)
        l_recon = self.mel_loss(real_audio_crop, generated_audio_crop)
        
        # Adversarial + Feature Matching (can be disabled during warmup)
        if use_adv and len(disc_fake_outputs) > 0:
            # L_adv: LSGAN loss
            l_adv = self.gan_loss.generator_loss(disc_fake_outputs)
            # L_fm: Feature matching L1
            l_fm = self.fm_loss(real_features, fake_features)
        else:
            l_adv = torch.tensor(0.0, device=real_audio.device)
            l_fm = torch.tensor(0.0, device=real_audio.device)
        
        # Paper formula: L_G = λ_recon × L_recon + λ_adv × L_adv + λ_fm × L_fm
        total = (
            self.lambda_recon * l_recon +
            self.lambda_adv * l_adv +
            self.lambda_fm * l_fm
        )
        
        return {
            "total": total,
            "reconstruction": l_recon,
            "adversarial": l_adv,
            "feature_matching": l_fm
        }
    
    def discriminator_loss(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Discriminator loss.
        
        Args:
            real_outputs: Discriminator outputs for real audio
            fake_outputs: Discriminator outputs for generated audio
            
        Returns:
            Dict with loss
        """
        loss = self.gan_loss.discriminator_loss(real_outputs, fake_outputs)
        
        return {"total": loss, "discriminator": loss}


# ============================================================================
# Unit tests
# ============================================================================

def _test_losses():
    """Тест loss functions (paper configuration)."""
    print("Testing Autoencoder Losses (SupertonicTTS paper config)...")
    
    batch_size = 4
    audio_len = 22050  # 1 second at 22kHz
    
    # Test multi-resolution STFT with paper config
    stft = MultiResolutionSTFT(
        fft_sizes=[512, 1024, 2048],  # Scaled for 22kHz
        sample_rate=22050,
        n_mels=[64, 128, 128]  # Paper: different per resolution
    )
    
    audio = torch.randn(batch_size, audio_len)
    mels = stft(audio)
    
    assert len(mels) == 3
    print(f"  Multi-resolution STFT shapes: {[m.shape for m in mels]}")
    print(f"    - FFT 512:  {mels[0].shape} (64 mels) ✓")
    print(f"    - FFT 1024: {mels[1].shape} (128 mels) ✓")
    print(f"    - FFT 2048: {mels[2].shape} (128 mels) ✓")
    
    # Test mel loss
    mel_loss = MultiResolutionMelLoss(
        fft_sizes=[512, 1024, 2048],
        sample_rate=22050,
        n_mels=[64, 128, 128]
    )
    real = torch.randn(batch_size, audio_len)
    fake = torch.randn(batch_size, audio_len)
    
    loss = mel_loss(real, fake)
    assert loss.dim() == 0
    print(f"  Mel L1 loss: {loss.item():.4f} ✓")
    
    # Test GAN loss (LSGAN as in paper)
    gan_loss = GANLoss(loss_type="lsgan")
    real_out = [torch.randn(batch_size, 100) for _ in range(5)]
    fake_out = [torch.randn(batch_size, 100) for _ in range(5)]
    
    d_loss = gan_loss.discriminator_loss(real_out, fake_out)
    g_loss = gan_loss.generator_loss(fake_out)
    
    print(f"  Discriminator loss: {d_loss.item():.4f} ✓")
    print(f"  Generator adv loss: {g_loss.item():.4f} ✓")
    
    # Test feature matching loss
    fm_loss = FeatureMatchingLoss()
    real_features = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(5)]
    fake_features = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(5)]
    
    fm = fm_loss(real_features, fake_features)
    print(f"  Feature matching loss: {fm.item():.4f} ✓")
    
    # Test full autoencoder loss with EXACT paper config
    ae_loss = AutoencoderLoss(
        lambda_recon=45.0,  # Paper: λ_recon = 45
        lambda_adv=1.0,     # Paper: λ_adv = 1
        lambda_fm=0.1,      # Paper: λ_fm = 0.1
        fft_sizes=[512, 1024, 2048],
        n_mels=[64, 128, 128],
        sample_rate=22050
    )
    
    gen_losses = ae_loss.generator_loss(
        real_audio=real,
        generated_audio=fake,
        disc_fake_outputs=fake_out,
        real_features=real_features,
        fake_features=fake_features
    )
    
    print(f"\n  Paper loss formula: L_G = 45×L_recon + 1×L_adv + 0.1×L_fm")
    print(f"  Total generator loss: {gen_losses['total'].item():.4f}")
    print(f"    - L_recon (mel):    {gen_losses['reconstruction'].item():.4f} × 45 = {gen_losses['reconstruction'].item() * 45:.4f}")
    print(f"    - L_adv:            {gen_losses['adversarial'].item():.4f} × 1  = {gen_losses['adversarial'].item():.4f}")
    print(f"    - L_fm:             {gen_losses['feature_matching'].item():.4f} × 0.1 = {gen_losses['feature_matching'].item() * 0.1:.4f}")
    
    disc_losses = ae_loss.discriminator_loss(real_out, fake_out)
    print(f"  Discriminator loss: {disc_losses['total'].item():.4f}")
    
    print("\nAll Autoencoder loss tests passed! ✓\n")


if __name__ == "__main__":
    _test_losses()
