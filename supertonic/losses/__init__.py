"""
Loss functions для Supertonic v2 TTS

Модулі:
- autoencoder_loss: Multi-resolution reconstruction + GAN losses
- flow_matching_loss: Conditional Flow Matching loss
- duration_loss: L1 loss для duration prediction
"""

from supertonic.losses.autoencoder_loss import (
    AutoencoderLoss,
    MultiResolutionMelLoss,
    GANLoss,
    FeatureMatchingLoss
)

from supertonic.losses.flow_matching_loss import (
    FlowMatchingLoss,
    flow_matching_loss,
    create_reference_mask
)

from supertonic.losses.duration_loss import (
    DurationLoss,
    duration_loss
)

__all__ = [
    "AutoencoderLoss",
    "MultiResolutionMelLoss",
    "GANLoss",
    "FeatureMatchingLoss",
    "FlowMatchingLoss",
    "flow_matching_loss",
    "create_reference_mask",
    "DurationLoss",
    "duration_loss"
]
