"""
Attention модулі для Supertonic v2

Включає:
- MultiHeadAttention з RoPE (для self-attention)
- SelfAttentionBlock (ConvNeXt-style wrapper)
- CrossAttentionBlock з LARoPE (для text-speech alignment)

Референс: Supertonic v2 paper - Text Encoder використовує 4 self-attention блоки
з RoPE, Reference/Text Encoder використовують cross-attention для conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange

from supertonic.models.larope import LARoPE, apply_larope, LARoPECrossAttention


class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Position Embedding (RoPE) для self-attention.

    Використовується в Text Encoder для self-attention між токенами.

    Args:
        dim: Розмірність per head (має бути парним)
        base: База для частотних смуг (default 10000)
        max_seq_len: Максимальна довжина для кешування
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seq_len: int = 8192
    ):
        super().__init__()

        assert dim % 2 == 0, f"Dimension must be even, got {dim}"

        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache sin/cos values
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Побудова кешу sin/cos значень."""
        positions = torch.arange(seq_len)
        angles = positions.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0)

        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Застосовує RoPE до input tensor.

        Args:
            x: Input tensor [B, H, T, D] або [B, T, D]
            seq_len: Optional sequence length (for offset)

        Returns:
            Rotated tensor тієї ж форми
        """
        if x.dim() == 3:
            batch_size, t, dim = x.shape
            x = x.unsqueeze(1)
            squeeze = True
        else:
            batch_size, num_heads, t, dim = x.shape
            squeeze = False

        # Отримуємо sin/cos з кешу
        if t > self.max_seq_len:
            self._build_cache(t)

        cos = self.cos_cache[:t].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
        sin = self.sin_cache[:t].unsqueeze(0).unsqueeze(0)

        # Apply rotation
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

        if squeeze:
            x_rotated = x_rotated.squeeze(1)

        return x_rotated


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention з RoPE.

    Використовується в Text Encoder (4 блоки, 4 heads, 512 filters).

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Attention dropout
        use_rope: Whether to use Rotary Position Embedding
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        bias: bool = True
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        inner_dim = self.num_heads * self.head_dim

        # QKV projection
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)

        # RoPE
        if use_rope:
            self.rope = RotaryEmbedding(dim=self.head_dim)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask [B, T] або [B, T, T]
            causal: Whether to use causal attention

        Returns:
            Output tensor [B, T, D]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b l (three h d) -> three b h l d',
                        three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply custom mask
        if mask is not None:
            if mask.dim() == 2:
                # [B, T] -> [B, 1, 1, T]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [B, T, T] -> [B, 1, T, T]
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h l d -> b l (h d)')

        return self.to_out(out)


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention блок з pre-norm та residual connection.

    Структура:
    1. LayerNorm
    2. MultiHeadAttention з RoPE
    3. Residual connection

    Використовується в Text Encoder: 4 блоки після ConvNeXt стеку.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_rope: bool = True
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask

        Returns:
            Output tensor [B, T, D]
        """
        # Pre-norm + attention + residual
        return x + self.attn(self.norm(x), mask=mask)


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention блок з LARoPE для text-speech conditioning.

    Структура:
    1. LayerNorm (for query)
    2. LARoPECrossAttention
    3. Residual connection

    Використовується в:
    - Reference Encoder: 2 cross-attention layers
    - Text Encoder: 2 cross-attention layers з reference vectors

    Args:
        dim: Model dimension
        context_dim: Context (key/value) dimension (default: same as dim)
        num_heads: Number of attention heads
        gamma: LARoPE gamma parameter
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 4,
        gamma: float = 10.0,
        dropout: float = 0.0
    ):
        super().__init__()

        context_dim = context_dim or dim

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(context_dim)

        self.cross_attn = LARoPECrossAttention(
            dim=dim,
            num_heads=num_heads,
            gamma=gamma,
            dropout=dropout
        )

        # Optional: projection if context_dim != dim
        if context_dim != dim:
            self.context_proj = nn.Linear(context_dim, dim)
        else:
            self.context_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,           # Query [B, Lq, D]
        context: torch.Tensor,     # Key/Value [B, Lk, D_context]
        x_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Query tensor [B, Lq, D]
            context: Context tensor [B, Lk, D_context]
            x_mask: Optional mask for queries
            context_mask: Optional mask for context

        Returns:
            Output tensor [B, Lq, D]
        """
        # Normalize
        x_norm = self.norm_q(x)
        context_norm = self.norm_kv(context)
        context_proj = self.context_proj(context_norm)

        # Cross-attention з LARoPE + residual
        out = self.cross_attn(x_norm, context_proj, x_mask, context_mask)
        return x + out


class FeedForward(nn.Module):
    """
    Feed-Forward Network для Transformer-style блоків.

    Структура: Linear → GELU → Dropout → Linear → Dropout

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (default: dim * 4)
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Transformer блок: Self-Attention + FFN.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        ff_mult: FFN hidden dim multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rope: bool = True
    ):
        super().__init__()

        self.attn_block = SelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope
        )

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(
            dim=dim,
            hidden_dim=dim * ff_mult,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.attn_block(x, mask)
        x = x + self.ff(self.ff_norm(x))
        return x


# ============================================================================
# Unit tests
# ============================================================================

def _test_rotary_embedding():
    """Тест RoPE."""
    print("Testing RotaryEmbedding...")

    batch_size = 2
    num_heads = 4
    seq_len = 100
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim)

    # Test 3D input
    x_3d = torch.randn(batch_size, seq_len, head_dim)
    y_3d = rope(x_3d)
    assert y_3d.shape == x_3d.shape
    print(f"  3D: {x_3d.shape} -> {y_3d.shape} ✓")

    # Test 4D input
    x_4d = torch.randn(batch_size, num_heads, seq_len, head_dim)
    y_4d = rope(x_4d)
    assert y_4d.shape == x_4d.shape
    print(f"  4D: {x_4d.shape} -> {y_4d.shape} ✓")

    print("RotaryEmbedding tests passed! ✓\n")


def _test_multi_head_attention():
    """Тест MultiHeadAttention."""
    print("Testing MultiHeadAttention...")

    batch_size = 2
    seq_len = 100
    dim = 512
    num_heads = 4

    attn = MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        use_rope=True
    )

    x = torch.randn(batch_size, seq_len, dim)
    out = attn(x)
    assert out.shape == x.shape
    print(f"  Basic: {x.shape} -> {out.shape} ✓")

    # With causal mask
    out_causal = attn(x, causal=True)
    assert out_causal.shape == x.shape
    print(f"  Causal: {out_causal.shape} ✓")

    # With padding mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 80:] = False
    out_masked = attn(x, mask=mask)
    assert out_masked.shape == x.shape
    print(f"  Masked: {out_masked.shape} ✓")

    num_params = sum(p.numel() for p in attn.parameters())
    print(f"  Parameters: {num_params:,}")

    print("MultiHeadAttention tests passed! ✓\n")


def _test_self_attention_block():
    """Тест SelfAttentionBlock."""
    print("Testing SelfAttentionBlock...")

    batch_size = 2
    seq_len = 100
    dim = 512

    block = SelfAttentionBlock(dim=dim, num_heads=4)

    x = torch.randn(batch_size, seq_len, dim)
    out = block(x)
    assert out.shape == x.shape
    print(f"  {x.shape} -> {out.shape} ✓")

    print("SelfAttentionBlock tests passed! ✓\n")


def _test_cross_attention_block():
    """Тест CrossAttentionBlock."""
    print("Testing CrossAttentionBlock...")

    batch_size = 2
    dim = 512

    # Speech features (query)
    lq = 100
    x = torch.randn(batch_size, lq, dim)

    # Text embeddings (context)
    lk = 20
    context = torch.randn(batch_size, lk, dim)

    block = CrossAttentionBlock(
        dim=dim,
        num_heads=4,
        gamma=10.0
    )

    out = block(x, context)
    assert out.shape == x.shape
    print(f"  Query: {x.shape}, Context: {context.shape} -> {out.shape} ✓")

    num_params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters: {num_params:,}")

    print("CrossAttentionBlock tests passed! ✓\n")


def _test_transformer_block():
    """Тест TransformerBlock."""
    print("Testing TransformerBlock...")

    batch_size = 2
    seq_len = 100
    dim = 512

    block = TransformerBlock(
        dim=dim,
        num_heads=4,
        ff_mult=4
    )

    x = torch.randn(batch_size, seq_len, dim)
    out = block(x)
    assert out.shape == x.shape
    print(f"  {x.shape} -> {out.shape} ✓")

    num_params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters: {num_params:,}")

    print("TransformerBlock tests passed! ✓\n")


if __name__ == "__main__":
    _test_rotary_embedding()
    _test_multi_head_attention()
    _test_self_attention_block()
    _test_cross_attention_block()
    _test_transformer_block()
    print("All attention tests passed! ✓")
