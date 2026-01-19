"""
Duration Predictor - передбачення загальної тривалості utterance

Ключова інновація Supertonic v2: передбачає UTTERANCE-LEVEL duration,
а не per-phoneme durations як у традиційних TTS системах.

Архітектура (~0.5M параметрів):
- Text Encoder: ConvNeXt blocks + attention → utterance embedding
- Reference Encoder: ConvNeXt blocks + attention → reference embedding
- Concatenate + MLP → scalar duration (в секундах або фреймах)

Training:
- L1 loss на ground-truth duration
- Швидке тренування: ~3000 iterations

Референс: Supertonic v2 paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from einops import rearrange

from supertonic.models.convnext import ConvNeXtBlock, ConvNeXtStack
from supertonic.models.attention import MultiHeadAttention


class DurationTextEncoder(nn.Module):
    """
    Text Encoder для Duration Predictor.

    Простіша версія ніж TextToLatent encoder:
    - ConvNeXt blocks
    - Self-attention для global context
    - Mean pooling → utterance embedding

    Args:
        vocab_size: Character vocabulary size
        embed_dim: Character embedding dimension
        hidden_dim: Hidden dimension
        num_convnext_blocks: Number of ConvNeXt blocks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_convnext_blocks: int = 4,
        kernel_size: int = 7,
        num_heads: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # ConvNeXt blocks
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_convnext_blocks,
            intermediate_dim=hidden_dim * 4,
            kernel_size=kernel_size,
            causal=False
        )

        # Self-attention для global context
        self.attention = MultiHeadAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            use_rope=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text to utterance embedding.

        Args:
            text: Character indices [B, L]
            text_mask: Optional mask [B, L]

        Returns:
            utterance_emb: [B, D]
        """
        # Embedding
        x = self.char_embed(text)  # [B, L, embed_dim]
        x = self.input_proj(x)  # [B, L, hidden_dim]

        # ConvNeXt
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.convnext(x)
        x = x.transpose(1, 2)  # [B, L, D]

        # Self-attention
        x = x + self.attention(self.norm(x), mask=text_mask)

        # Mean pooling (з урахуванням маски)
        if text_mask is not None:
            x = x * text_mask.unsqueeze(-1).float()
            utterance_emb = x.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).float()
        else:
            utterance_emb = x.mean(dim=1)

        return utterance_emb


class DurationReferenceEncoder(nn.Module):
    """
    Reference Encoder для Duration Predictor.

    Кодує reference audio latent → reference embedding.

    Args:
        input_dim: Latent dimension (144 compressed)
        hidden_dim: Hidden dimension
        num_convnext_blocks: Number of ConvNeXt blocks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        input_dim: int = 144,
        hidden_dim: int = 256,
        num_convnext_blocks: int = 4,
        kernel_size: int = 7,
        num_heads: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ConvNeXt blocks
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_convnext_blocks,
            intermediate_dim=hidden_dim * 4,
            kernel_size=kernel_size,
            causal=False
        )

        # Self-attention
        self.attention = MultiHeadAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            use_rope=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ref_latent: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode reference latent to embedding.

        Args:
            ref_latent: Reference latent [B, C, T]
            ref_mask: Optional mask [B, T]

        Returns:
            ref_emb: [B, D]
        """
        # Input projection
        x = ref_latent.transpose(1, 2)  # [B, T, C]
        x = self.input_proj(x)  # [B, T, D]

        # ConvNeXt
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.convnext(x)
        x = x.transpose(1, 2)  # [B, T, D]

        # Self-attention
        x = x + self.attention(self.norm(x), mask=ref_mask)

        # Mean pooling
        if ref_mask is not None:
            x = x * ref_mask.unsqueeze(-1).float()
            ref_emb = x.sum(dim=1) / ref_mask.sum(dim=1, keepdim=True).float()
        else:
            ref_emb = x.mean(dim=1)

        return ref_emb


class DurationPredictor(nn.Module):
    """
    Duration Predictor - передбачає utterance-level duration.

    Архітектура:
    1. Text Encoder → utterance embedding
    2. Reference Encoder → reference embedding
    3. Concatenate → MLP → scalar duration

    Ключова особливість: передбачає ЗАГАЛЬНУ тривалість,
    не per-phoneme durations. Це спрощує архітектуру та
    покращує prosody transfer від reference.

    Args:
        vocab_size: Character vocabulary size
        latent_dim: Reference latent dimension (144)
        hidden_dim: Hidden dimension for encoders
        output_unit: "frames" або "seconds"
        hop_length: Hop length для конвертації фрейми↔секунди
        sample_rate: Sample rate
        temporal_compression: Kc factor
    """

    def __init__(
        self,
        vocab_size: int = 512,
        latent_dim: int = 144,
        hidden_dim: int = 256,
        num_convnext_blocks: int = 4,
        kernel_size: int = 7,
        num_heads: int = 4,
        output_unit: str = "frames",  # "frames" або "seconds"
        hop_length: int = 512,
        sample_rate: int = 44100,
        temporal_compression: int = 6
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_unit = output_unit
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.temporal_compression = temporal_compression

        # Frame duration in seconds
        self.frame_duration = hop_length / sample_rate * temporal_compression

        # Text encoder
        self.text_encoder = DurationTextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_convnext_blocks=num_convnext_blocks,
            kernel_size=kernel_size,
            num_heads=num_heads
        )

        # Reference encoder
        self.reference_encoder = DurationReferenceEncoder(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_convnext_blocks=num_convnext_blocks,
            kernel_size=kernel_size,
            num_heads=num_heads
        )

        # Duration MLP
        # Input: text_emb (D) + ref_emb (D) = 2D
        self.duration_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive duration
        )

        # Optional: speech rate predictor (relative to reference)
        self.rate_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], scaled to [0.5, 2.0]
        )

    def forward(
        self,
        text: torch.Tensor,
        ref_latent: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        target_duration: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text: Character indices [B, L]
            ref_latent: Reference latent [B, C, T]
            text_mask: Optional text mask [B, L]
            ref_mask: Optional reference mask [B, T]
            target_duration: Ground-truth duration [B] (for training)

        Returns:
            Dict with 'duration', 'rate', and optionally 'loss'
        """
        # Encode text and reference
        text_emb = self.text_encoder(text, text_mask)  # [B, D]
        ref_emb = self.reference_encoder(ref_latent, ref_mask)  # [B, D]

        # Concatenate
        combined = torch.cat([text_emb, ref_emb], dim=-1)  # [B, 2D]

        # Predict duration
        duration = self.duration_mlp(combined).squeeze(-1)  # [B]

        # Predict speech rate (optional)
        rate_raw = self.rate_predictor(combined).squeeze(-1)  # [B] in [0, 1]
        rate = 0.5 + rate_raw * 1.5  # Scale to [0.5, 2.0]

        output = {
            'duration': duration,
            'rate': rate,
            'text_embedding': text_emb,
            'reference_embedding': ref_emb
        }

        # Compute loss if target provided
        if target_duration is not None:
            loss = F.l1_loss(duration, target_duration)
            output['loss'] = loss

        return output

    def predict(
        self,
        text: torch.Tensor,
        ref_latent: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        rate_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict duration (inference mode).

        Args:
            text: Character indices [B, L]
            ref_latent: Reference latent [B, C, T]
            text_mask: Optional text mask
            ref_mask: Optional reference mask
            rate_scale: Manual rate adjustment (0.5-2.0)

        Returns:
            duration_frames: Predicted duration in frames [B]
            duration_seconds: Predicted duration in seconds [B]
        """
        output = self.forward(text, ref_latent, text_mask, ref_mask)

        # Apply rate scale
        duration = output['duration'] * rate_scale

        if self.output_unit == "frames":
            duration_frames = duration
            duration_seconds = duration * self.frame_duration
        else:
            duration_seconds = duration
            duration_frames = duration / self.frame_duration

        return duration_frames.long(), duration_seconds

    def frames_to_seconds(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert frames to seconds."""
        return frames.float() * self.frame_duration

    def seconds_to_frames(self, seconds: torch.Tensor) -> torch.Tensor:
        """Convert seconds to frames."""
        return (seconds / self.frame_duration).long()


class DurationPredictorLoss(nn.Module):
    """
    Loss function для Duration Predictor.

    Combines:
    - L1 loss на duration
    - Optional: rate consistency loss

    Args:
        duration_weight: Weight for duration loss
        rate_weight: Weight for rate consistency loss
    """

    def __init__(
        self,
        duration_weight: float = 1.0,
        rate_weight: float = 0.1
    ):
        super().__init__()

        self.duration_weight = duration_weight
        self.rate_weight = rate_weight

    def forward(
        self,
        pred_duration: torch.Tensor,
        target_duration: torch.Tensor,
        pred_rate: Optional[torch.Tensor] = None,
        target_rate: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            pred_duration: Predicted duration [B]
            target_duration: Ground-truth duration [B]
            pred_rate: Predicted rate [B] (optional)
            target_rate: Target rate [B] (optional)

        Returns:
            Dict with 'loss', 'duration_loss', 'rate_loss'
        """
        # Duration loss (L1)
        duration_loss = F.l1_loss(pred_duration, target_duration)

        total_loss = self.duration_weight * duration_loss

        output = {
            'duration_loss': duration_loss,
        }

        # Rate loss (optional)
        if pred_rate is not None and target_rate is not None:
            rate_loss = F.l1_loss(pred_rate, target_rate)
            total_loss = total_loss + self.rate_weight * rate_loss
            output['rate_loss'] = rate_loss

        output['loss'] = total_loss
        return output


# ============================================================================
# Unit tests
# ============================================================================

def _test_duration_text_encoder():
    """Тест DurationTextEncoder."""
    print("Testing DurationTextEncoder...")

    encoder = DurationTextEncoder(
        vocab_size=512,
        hidden_dim=256,
        num_convnext_blocks=4
    )

    batch_size = 2
    text_len = 50
    text = torch.randint(0, 512, (batch_size, text_len))

    utterance_emb = encoder(text)
    assert utterance_emb.shape == (batch_size, 256)
    print(f"  Text: {text.shape} -> Utterance emb: {utterance_emb.shape} ✓")

    # With mask
    mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    mask[0, 40:] = False
    utterance_emb_masked = encoder(text, mask)
    assert utterance_emb_masked.shape == (batch_size, 256)
    print(f"  With mask: {utterance_emb_masked.shape} ✓")

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("DurationTextEncoder tests passed! ✓\n")


def _test_duration_reference_encoder():
    """Тест DurationReferenceEncoder."""
    print("Testing DurationReferenceEncoder...")

    encoder = DurationReferenceEncoder(
        input_dim=144,
        hidden_dim=256,
        num_convnext_blocks=4
    )

    batch_size = 2
    t_ref = 100
    ref_latent = torch.randn(batch_size, 144, t_ref)

    ref_emb = encoder(ref_latent)
    assert ref_emb.shape == (batch_size, 256)
    print(f"  Ref latent: {ref_latent.shape} -> Ref emb: {ref_emb.shape} ✓")

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("DurationReferenceEncoder tests passed! ✓\n")


def _test_duration_predictor():
    """Тест повного DurationPredictor."""
    print("Testing DurationPredictor...")

    predictor = DurationPredictor(
        vocab_size=512,
        latent_dim=144,
        hidden_dim=256,
        num_convnext_blocks=4,
        output_unit="frames",
        hop_length=512,
        sample_rate=44100,
        temporal_compression=6
    )

    batch_size = 2
    text_len = 50
    t_ref = 100

    text = torch.randint(0, 512, (batch_size, text_len))
    ref_latent = torch.randn(batch_size, 144, t_ref)
    target_duration = torch.tensor([150.0, 200.0])  # frames

    # Training forward
    output = predictor(text, ref_latent, target_duration=target_duration)
    print(f"  Predicted duration: {output['duration']}")
    print(f"  Predicted rate: {output['rate']}")
    print(f"  Loss: {output['loss'].item():.6f}")

    # Inference
    duration_frames, duration_seconds = predictor.predict(text, ref_latent)
    print(f"  Inference - frames: {duration_frames}, seconds: {duration_seconds}")

    # Parameter count
    total_params = sum(p.numel() for p in predictor.parameters())
    text_params = sum(p.numel() for p in predictor.text_encoder.parameters())
    ref_params = sum(p.numel() for p in predictor.reference_encoder.parameters())

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Text encoder: {text_params:,}")
    print(f"  Reference encoder: {ref_params:,}")

    # Check it's around 0.5M as expected
    assert total_params < 1_000_000, f"Too many params: {total_params}"
    print(f"  ✓ Parameter count is reasonable (~0.5M)")

    print("\nDurationPredictor tests passed! ✓\n")


if __name__ == "__main__":
    _test_duration_text_encoder()
    _test_duration_reference_encoder()
    _test_duration_predictor()
    print("All Duration Predictor tests passed! ✓")
