"""
Duration Loss - L1 loss для Duration Predictor

Простий L1 loss для utterance-level duration prediction:
    L_dur = L1(d_predicted, d_groundtruth)

Duration вимірюється в секундах або кількості фреймів.

Референс: Supertonic v2 paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def duration_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L1 loss для duration prediction.
    
    Args:
        predicted: Predicted duration [B] або [B, 1]
        target: Ground-truth duration [B] або [B, 1]
        reduction: "mean", "sum", або "none"
        
    Returns:
        loss: L1 loss tensor
    """
    # Flatten
    predicted = predicted.view(-1)
    target = target.view(-1)
    
    return F.l1_loss(predicted, target, reduction=reduction)


class DurationLoss(nn.Module):
    """
    Duration Loss module.
    
    Обчислює L1 loss + optional percentage error.
    
    Args:
        reduction: "mean", "sum", або "none"
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute duration loss.
        
        Args:
            predicted: Predicted duration [B] або [B, 1]
            target: Ground-truth duration [B] або [B, 1]
            
        Returns:
            Dict with loss components
        """
        # Flatten
        predicted = predicted.view(-1)
        target = target.view(-1)
        
        # L1 loss
        l1 = duration_loss(predicted, target, reduction=self.reduction)
        
        # Percentage error (for logging)
        with torch.no_grad():
            percent_error = ((predicted - target).abs() / (target + 1e-8)).mean() * 100
        
        return {
            "total": l1,
            "l1": l1,
            "percent_error": percent_error
        }


class DurationLossWithMask(nn.Module):
    """
    Duration Loss з mask support.
    
    Для batch training де деякі samples можуть бути padded.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked duration loss.
        
        Args:
            predicted: [B] або [B, 1]
            target: [B] або [B, 1]
            mask: Optional valid sample mask [B]
            
        Returns:
            Dict with loss components
        """
        predicted = predicted.view(-1)
        target = target.view(-1)
        
        if mask is None:
            mask = torch.ones_like(predicted)
        else:
            mask = mask.view(-1).float()
        
        # Masked L1
        l1_per_sample = (predicted - target).abs()
        l1 = (l1_per_sample * mask).sum() / (mask.sum() + 1e-8)
        
        # Masked percentage error
        with torch.no_grad():
            percent_per_sample = l1_per_sample / (target + 1e-8) * 100
            percent_error = (percent_per_sample * mask).sum() / (mask.sum() + 1e-8)
        
        return {
            "total": l1,
            "l1": l1,
            "percent_error": percent_error
        }


# ============================================================================
# Utilities
# ============================================================================

def duration_to_frames(
    duration_seconds: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512,
    compression_factor: int = 6
) -> torch.Tensor:
    """
    Конвертує duration з секунд в кількість compressed frames.
    
    Args:
        duration_seconds: Duration in seconds [B]
        sample_rate: Audio sample rate
        hop_length: STFT hop length
        compression_factor: Temporal compression factor (Kc)
        
    Returns:
        num_frames: Number of compressed frames [B]
    """
    # Samples → mel frames → compressed frames
    num_samples = duration_seconds * sample_rate
    num_mel_frames = num_samples / hop_length
    num_compressed_frames = num_mel_frames / compression_factor
    
    return num_compressed_frames


def frames_to_duration(
    num_frames: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512,
    compression_factor: int = 6
) -> torch.Tensor:
    """
    Конвертує кількість compressed frames в секунди.
    
    Args:
        num_frames: Number of compressed frames [B]
        sample_rate: Audio sample rate
        hop_length: STFT hop length
        compression_factor: Temporal compression factor (Kc)
        
    Returns:
        duration_seconds: Duration in seconds [B]
    """
    num_mel_frames = num_frames * compression_factor
    num_samples = num_mel_frames * hop_length
    duration_seconds = num_samples / sample_rate
    
    return duration_seconds


# ============================================================================
# Unit tests
# ============================================================================

def _test_duration_loss():
    """Тест duration loss."""
    print("Testing Duration Loss...")
    
    batch_size = 8
    
    # Test basic loss
    predicted = torch.tensor([2.5, 3.0, 1.5, 4.0, 2.0, 3.5, 1.8, 2.2])
    target = torch.tensor([2.3, 3.2, 1.4, 4.1, 2.1, 3.3, 1.9, 2.0])
    
    loss = duration_loss(predicted, target)
    print(f"  Basic L1 loss: {loss.item():.4f} ✓")
    
    # Test module
    loss_fn = DurationLoss()
    losses = loss_fn(predicted, target)
    
    print(f"  Module L1 loss: {losses['l1'].item():.4f}")
    print(f"  Percent error: {losses['percent_error'].item():.2f}%")
    
    # Test masked loss
    masked_loss_fn = DurationLossWithMask()
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1])  # Ignore samples 4, 5
    
    masked_losses = masked_loss_fn(predicted, target, mask)
    print(f"  Masked L1 loss: {masked_losses['l1'].item():.4f}")
    
    # Test duration conversions
    duration_sec = torch.tensor([2.0, 3.0, 1.5])
    frames = duration_to_frames(duration_sec)
    print(f"  Duration {duration_sec.tolist()} sec → {frames.tolist()} frames")
    
    back_to_sec = frames_to_duration(frames)
    print(f"  Frames {frames.tolist()} → {back_to_sec.tolist()} sec")
    
    print("All Duration Loss tests passed! ✓\n")


if __name__ == "__main__":
    _test_duration_loss()
