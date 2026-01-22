"""
Flow Matching Loss для Text-to-Latent module

Conditional Flow Matching (CFM) loss:
    L_FM = E[||m · (v_θ(z_t, z_ref, c, t) - (z_1 - (1-σ_min)z_0))||_1]

Де:
- z_0: Noise sample ~ N(0, I)
- z_1: Target latents
- z_t: Interpolated latents = (1 - (1-σ_min)t) * z_0 + t * z_1
- v_θ: Predicted velocity field
- m: Reference mask (для reference masking)
- σ_min = 1e-8

Classifier-Free Guidance (CFG):
- p_uncond = 0.05 (5% unconditional training)
- При unconditional: conditional inputs замінюються на learnable parameters

Референс: Supertonic v2 paper (2509.11084), Matcha-TTS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


def create_reference_mask(
    shape: Tuple[int, ...],
    mask_ratio_min: float = 0.3,
    mask_ratio_max: float = 0.7,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Створює маску для reference masking в flow-matching loss.
    
    Reference masking запобігає information leakage:
    модель не може просто копіювати reference.
    
    Args:
        shape: Shape of the mask (B, C, T) або (B, T)
        mask_ratio_min: Minimum ratio to mask
        mask_ratio_max: Maximum ratio to mask
        device: Target device
        
    Returns:
        mask: Binary mask tensor (1 = masked/train, 0 = visible/reference)
    """
    if len(shape) == 3:
        batch_size, channels, seq_len = shape
        mask_shape = (batch_size, 1, seq_len)  # Broadcast over channels
    else:
        batch_size, seq_len = shape
        mask_shape = (batch_size, seq_len)
    
    mask = torch.zeros(mask_shape, device=device)
    
    for b in range(batch_size):
        # Random mask ratio
        ratio = torch.rand(1).item() * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask_len = int(seq_len * ratio)
        
        # Random start position
        if seq_len > mask_len:
            start = torch.randint(0, seq_len - mask_len, (1,)).item()
        else:
            start = 0
            mask_len = seq_len
        
        # Apply mask
        if len(shape) == 3:
            mask[b, :, start:start + mask_len] = 1.0
        else:
            mask[b, start:start + mask_len] = 1.0
    
    return mask


def interpolate_latents(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 1e-8
) -> torch.Tensor:
    """
    Інтерполює між noise (z0) та target (z1).
    
    z_t = (1 - (1-σ_min)t) * z_0 + t * z_1
    
    Args:
        z0: Noise samples [B, C, T]
        z1: Target latents [B, C, T]
        t: Timesteps [B, 1, 1] або [B]
        sigma_min: Minimum sigma value
        
    Returns:
        z_t: Interpolated latents [B, C, T]
    """
    # Ensure t has correct shape
    while t.dim() < z0.dim():
        t = t.unsqueeze(-1)
    
    # Interpolation: z_t = (1 - (1-σ)t) * z_0 + t * z_1
    coef_z0 = 1 - (1 - sigma_min) * t
    coef_z1 = t
    
    z_t = coef_z0 * z0 + coef_z1 * z1
    
    return z_t


def compute_target_velocity(
    z0: torch.Tensor,
    z1: torch.Tensor,
    sigma_min: float = 1e-8
) -> torch.Tensor:
    """
    Обчислює target velocity (straight path).
    
    v_target = z_1 - (1-σ_min) * z_0
    
    Args:
        z0: Noise samples [B, C, T]
        z1: Target latents [B, C, T]
        sigma_min: Minimum sigma value
        
    Returns:
        target_velocity: [B, C, T]
    """
    return z1 - (1 - sigma_min) * z0


def flow_matching_loss(
    model: nn.Module,
    z1: torch.Tensor,
    text_encoding: torch.Tensor,
    reference_encoding: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
    sigma_min: float = 1e-8,
    p_uncond: float = 0.05,
    mask_ratio_min: float = 0.3,
    mask_ratio_max: float = 0.7
) -> Dict[str, torch.Tensor]:
    """
    Flow-matching loss для Text-to-Latent training.
    
    Args:
        model: VectorFieldEstimator model
        z1: Target latents [B, C, T]
        text_encoding: Encoded text [B, L, D]
        reference_encoding: Reference vectors [B, 50, D]
        text_mask: Optional text mask [B, L]
        sigma_min: Minimum sigma
        p_uncond: Probability of unconditional training (CFG)
        mask_ratio_min: Minimum reference mask ratio
        mask_ratio_max: Maximum reference mask ratio
        
    Returns:
        Dict with loss components
    """
    batch_size = z1.shape[0]
    device = z1.device
    
    # Sample timestep t ~ U[0, 1]
    t = torch.rand(batch_size, device=device)
    
    # Sample noise z_0 ~ N(0, I)
    z0 = torch.randn_like(z1)
    
    # Interpolate: z_t = (1 - (1-σ)t) * z_0 + t * z_1
    z_t = interpolate_latents(z0, z1, t, sigma_min)
    
    # Target velocity (straight path)
    target_velocity = compute_target_velocity(z0, z1, sigma_min)
    
    # Reference masking (prevent information leakage)
    mask = create_reference_mask(
        z1.shape,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
        device=device
    )
    z_ref = (1 - mask) * z1  # Visible reference (unmasked parts)
    
    # Classifier-Free Guidance: sometimes train unconditionally
    uncond_mask = torch.rand(batch_size, device=device) < p_uncond
    
    # For unconditional samples, we null out the conditioning
    text_cond = text_encoding.clone()
    z_ref_cond = z_ref.clone()
    
    if uncond_mask.any():
        # Zero out conditioning for unconditional samples
        text_cond[uncond_mask] = 0
        z_ref_cond[uncond_mask] = 0  # Null reference for unconditional samples
    
    # Predict velocity
    # Note: VectorFieldEstimator uses z_ref (masked latents) for reference conditioning,
    # not the 50-vector reference_encoding.
    predicted_velocity = model(
        z_t=z_t,
        z_ref=z_ref_cond,
        text_encoding=text_cond,
        t=t,
        text_mask=text_mask
    )
    
    # L1 loss з masking (тільки на masked regions)
    loss_per_sample = (mask * (predicted_velocity - target_velocity).abs()).sum(dim=(1, 2))
    loss_per_sample = loss_per_sample / (mask.sum(dim=(1, 2)) + 1e-8)
    
    loss = loss_per_sample.mean()
    
    return {
        "total": loss,
        "flow_matching": loss,
        "mean_velocity_error": (predicted_velocity - target_velocity).abs().mean()
    }


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss module.
    
    Encapsulates flow-matching loss computation з configurable parameters.
    
    Args:
        sigma_min: Minimum sigma value (default 1e-8)
        p_uncond: Unconditional training probability for CFG (default 0.05)
        mask_ratio_min: Minimum reference mask ratio
        mask_ratio_max: Maximum reference mask ratio
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-8,
        p_uncond: float = 0.05,
        mask_ratio_min: float = 0.3,
        mask_ratio_max: float = 0.7
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
    
    def forward(
        self,
        model: nn.Module,
        z1: torch.Tensor,
        text_encoding: torch.Tensor,
        reference_encoding: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow-matching loss.
        
        Args:
            model: VectorFieldEstimator
            z1: Target compressed latents [B, 144, T]
            text_encoding: Encoded text [B, L, D]
            reference_encoding: Reference vectors [B, 50, D]
            text_mask: Optional text mask
            
        Returns:
            Loss dict
        """
        return flow_matching_loss(
            model=model,
            z1=z1,
            text_encoding=text_encoding,
            reference_encoding=reference_encoding,
            text_mask=text_mask,
            sigma_min=self.sigma_min,
            p_uncond=self.p_uncond,
            mask_ratio_min=self.mask_ratio_min,
            mask_ratio_max=self.mask_ratio_max
        )


class ODESolver:
    """
    Euler ODE Solver для inference.
    
    Генерує латенти з noise через Euler integration:
        z_{i+1} = z_i + v(z_i, t_i) * dt
    
    Args:
        nfe: Number of function evaluations (default 32)
        cfg_scale: Classifier-free guidance scale (default 3.0)
    """
    
    def __init__(
        self,
        nfe: int = 32,
        cfg_scale: float = 3.0
    ):
        self.nfe = nfe
        self.cfg_scale = cfg_scale
    
    @torch.no_grad()
    def solve(
        self,
        model: nn.Module,
        z_shape: Tuple[int, ...],
        text_encoding: torch.Tensor,
        reference_encoding: torch.Tensor,
        z_ref: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Solve ODE to generate latents.
        
        Args:
            model: VectorFieldEstimator
            z_shape: Shape of latents to generate (B, 144, T)
            text_encoding: Encoded text [B, L, D]
            reference_encoding: Reference vectors [B, 50, D]
            z_ref: Optional reference latents for conditioning
            text_mask: Optional text mask
            device: Target device
            
        Returns:
            Generated latents [B, 144, T]
        """
        device = device or text_encoding.device
        
        # Initialize from noise
        z = torch.randn(z_shape, device=device)
        
        if z_ref is None:
            z_ref = torch.zeros_like(z)
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / self.nfe
        
        for step in range(self.nfe):
            t = torch.full((z_shape[0],), step * dt, device=device)
            
            # Conditional velocity
            velocity_cond = model(
                z_t=z,
                z_ref=z_ref,
                text_encoding=text_encoding,
                t=t,
                text_mask=text_mask
            )
            
            # Unconditional velocity (for CFG)
            if self.cfg_scale > 1.0:
                # Null conditioning
                text_null = torch.zeros_like(text_encoding)
                z_ref_null = torch.zeros_like(z_ref)
                
                velocity_uncond = model(
                    z_t=z,
                    z_ref=z_ref_null,
                    text_encoding=text_null,
                    t=t,
                    text_mask=text_mask
                )
                
                # CFG: v = v_uncond + scale * (v_cond - v_uncond)
                velocity = velocity_uncond + self.cfg_scale * (velocity_cond - velocity_uncond)
            else:
                velocity = velocity_cond
            
            # Euler step
            z = z + velocity * dt
        
        return z


# ============================================================================
# Temporal Compression utilities
# ============================================================================

def compress_latents(
    latents: torch.Tensor,
    compression_factor: int = 6
) -> torch.Tensor:
    """
    Temporal compression: стек Kc фреймів в один вектор.
    
    (C=24, T) → (C×Kc=144, T/Kc)
    
    Args:
        latents: [B, C, T] де C=24
        compression_factor: Kc (default 6)
        
    Returns:
        compressed: [B, C*Kc, T//Kc]
    """
    batch_size, channels, seq_len = latents.shape
    
    # Pad to multiple of compression_factor
    if seq_len % compression_factor != 0:
        pad_len = compression_factor - (seq_len % compression_factor)
        latents = F.pad(latents, (0, pad_len))
        seq_len = latents.shape[-1]
    
    # Reshape: [B, C, T] → [B, C, T/Kc, Kc] → [B, C*Kc, T/Kc]
    latents = latents.view(batch_size, channels, seq_len // compression_factor, compression_factor)
    latents = latents.permute(0, 1, 3, 2)  # [B, C, Kc, T/Kc]
    latents = latents.reshape(batch_size, channels * compression_factor, seq_len // compression_factor)
    
    return latents


def decompress_latents(
    compressed: torch.Tensor,
    compression_factor: int = 6
) -> torch.Tensor:
    """
    Temporal decompression: reverse of compress_latents.
    
    (C×Kc=144, T/Kc) → (C=24, T)
    
    Args:
        compressed: [B, C*Kc, T_compressed]
        compression_factor: Kc (default 6)
        
    Returns:
        latents: [B, C, T]
    """
    batch_size, compressed_channels, compressed_len = compressed.shape
    channels = compressed_channels // compression_factor
    seq_len = compressed_len * compression_factor
    
    # Reshape: [B, C*Kc, T/Kc] → [B, C, Kc, T/Kc] → [B, C, T]
    compressed = compressed.view(batch_size, channels, compression_factor, compressed_len)
    compressed = compressed.permute(0, 1, 3, 2)  # [B, C, T/Kc, Kc]
    latents = compressed.reshape(batch_size, channels, seq_len)
    
    return latents


# ============================================================================
# Unit tests
# ============================================================================

def _test_flow_matching():
    """Тест flow matching loss."""
    print("Testing Flow Matching Loss...")
    
    batch_size = 4
    latent_dim = 144
    seq_len = 50
    text_len = 100
    hidden_dim = 128
    
    # Test interpolation
    z0 = torch.randn(batch_size, latent_dim, seq_len)
    z1 = torch.randn(batch_size, latent_dim, seq_len)
    t = torch.rand(batch_size)
    
    z_t = interpolate_latents(z0, z1, t, sigma_min=1e-8)
    assert z_t.shape == z0.shape
    print(f"  Interpolation: {z0.shape} → {z_t.shape} ✓")
    
    # Test target velocity
    target_v = compute_target_velocity(z0, z1, sigma_min=1e-8)
    assert target_v.shape == z0.shape
    print(f"  Target velocity: {target_v.shape} ✓")
    
    # Test reference masking
    mask = create_reference_mask((batch_size, latent_dim, seq_len))
    assert mask.shape == (batch_size, 1, seq_len)
    print(f"  Reference mask: {mask.shape}, ratio={mask.mean().item():.2f} ✓")
    
    # Test compression/decompression
    latents_24 = torch.randn(batch_size, 24, 300)
    compressed = compress_latents(latents_24, compression_factor=6)
    assert compressed.shape == (batch_size, 144, 50)
    print(f"  Compression: {latents_24.shape} → {compressed.shape} ✓")
    
    decompressed = decompress_latents(compressed, compression_factor=6)
    assert decompressed.shape == latents_24.shape
    print(f"  Decompression: {compressed.shape} → {decompressed.shape} ✓")
    
    # Test ODE solver (mock model)
    class MockVF(nn.Module):
        def forward(self, z_t, z_ref, text_encoding, t, text_mask=None):
            return torch.randn_like(z_t)
    
    mock_model = MockVF()
    solver = ODESolver(nfe=8, cfg_scale=3.0)
    
    text_enc = torch.randn(batch_size, text_len, hidden_dim)
    ref_enc = torch.randn(batch_size, 50, hidden_dim)
    
    generated = solver.solve(
        model=mock_model,
        z_shape=(batch_size, latent_dim, seq_len),
        text_encoding=text_enc,
        reference_encoding=ref_enc
    )
    
    assert generated.shape == (batch_size, latent_dim, seq_len)
    print(f"  ODE solver: generated shape {generated.shape} ✓")
    
    print("All Flow Matching tests passed! ✓\n")


if __name__ == "__main__":
    _test_flow_matching()
