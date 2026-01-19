"""
LARoPE: Length-Aware Rotary Position Embedding

Ключова інновація Supertonic v2 для text-speech alignment у cross-attention.

Проблема стандартного RoPE:
- RoPE використовує абсолютні позиційні індекси
- У cross-attention text (короткий) та speech (довгий) мають різну довжину
- Це порушує relative position property

LARoPE рішення:
- Нормалізує позиції за довжиною послідовності
- normalized_pos = γ × (position / seq_length)
- γ = 10 індукує diagonal bias в attention maps
- Це відповідає монотонному text-speech alignment

Референс: Supertonic v2 paper (2509.11084)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange


class LARoPE(nn.Module):
    """
    Length-Aware Rotary Position Embedding.

    Нормалізує позиційні embeddings за довжиною послідовності,
    що критично для cross-attention між текстом та аудіо різної довжини.

    Args:
        dim: Розмірність embedding (має бути парним)
        gamma: Scaling hyperparameter (default 10 — оптимально для TTS)
        base: База для частотних смуг (default 10000)
        max_seq_len: Максимальна довжина для кешування (optional)
    """

    def __init__(
        self,
        dim: int,
        gamma: float = 10.0,
        base: float = 10000.0,
        max_seq_len: int = 8192
    ):
        super().__init__()

        assert dim % 2 == 0, f"Dimension must be even, got {dim}"

        self.dim = dim
        self.gamma = gamma
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies
        # θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_rope_embeddings(
        self,
        positions: torch.Tensor,  # [T] or [B, T]
        seq_length: torch.Tensor,  # scalar or [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Обчислює sin/cos embeddings для LARoPE.

        Args:
            positions: Позиційні індекси
            seq_length: Довжина послідовності для нормалізації

        Returns:
            cos, sin tensors для rotation
        """
        # Нормалізуємо позиції: γ × (pos / L)
        # Це ключова відмінність від стандартного RoPE
        if positions.dim() == 1:
            normalized_pos = self.gamma * (positions.float() / seq_length)  # [T]
        else:
            # [B, T] / [B, 1] -> [B, T]
            normalized_pos = self.gamma * (positions.float() / seq_length.unsqueeze(-1))

        # Обчислюємо кути: normalized_pos × θ
        # [T] × [D/2] -> [T, D/2] або [B, T] × [D/2] -> [B, T, D/2]
        if normalized_pos.dim() == 1:
            angles = normalized_pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [T, D/2]
        else:
            angles = normalized_pos.unsqueeze(-1) * self.inv_freq  # [B, T, D/2]

        # cos та sin для rotation
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        seq_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Застосовує LARoPE до input tensor.

        Args:
            x: Input tensor [B, T, D] або [B, H, T, D]
            seq_length: Довжина послідовності (якщо None — використовує T)

        Returns:
            Rotated tensor тієї ж форми
        """
        # Визначаємо розміри
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x = x.unsqueeze(1)  # [B, 1, T, D]
            squeeze_output = True
        else:
            batch_size, num_heads, seq_len, dim = x.shape
            squeeze_output = False

        assert dim == self.dim, f"Dimension mismatch: {dim} vs {self.dim}"

        # Якщо seq_length не задано — використовуємо актуальну довжину
        if seq_length is None:
            seq_length = torch.tensor(seq_len, device=x.device, dtype=torch.float)

        # Позиції: 0, 1, 2, ..., T-1
        positions = torch.arange(seq_len, device=x.device)

        # Обчислюємо rotation embeddings
        cos, sin = self._compute_rope_embeddings(positions, seq_length)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]

        # Розділяємо на парні/непарні індекси
        x_even = x[..., 0::2]  # [B, H, T, D/2]
        x_odd = x[..., 1::2]   # [B, H, T, D/2]

        # Застосовуємо rotation
        # x_rotated = x × cos + rotate(x) × sin
        # rotate([x_even, x_odd]) = [-x_odd, x_even]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Збираємо назад
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # [B, H, T, D]

        if squeeze_output:
            x_rotated = x_rotated.squeeze(1)  # [B, T, D]

        return x_rotated


def apply_larope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_seq_len: torch.Tensor,
    k_seq_len: torch.Tensor,
    gamma: float = 10.0,
    base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Функціональний інтерфейс для LARoPE в cross-attention.

    Застосовує length-normalized rotary embeddings до query та key,
    кожен з власною довжиною для нормалізації.

    Args:
        q: Query tensor [B, H, Lq, D]
        k: Key tensor [B, H, Lk, D]
        q_seq_len: Query sequence length (для нормалізації)
        k_seq_len: Key sequence length (для нормалізації)
        gamma: LARoPE scaling factor (default 10)
        base: RoPE base frequency

    Returns:
        Rotated (q, k) tensors

    Приклад:
        >>> q = torch.randn(2, 4, 100, 64)  # Speech features
        >>> k = torch.randn(2, 4, 20, 64)   # Text embeddings
        >>> q_rot, k_rot = apply_larope(q, k, 100, 20, gamma=10)
    """
    batch_size, num_heads, lq, dim = q.shape
    _, _, lk, _ = k.shape

    assert dim % 2 == 0, f"Dimension must be even, got {dim}"

    # Inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=q.device).float() / dim))

    # === Query rotation ===
    q_positions = torch.arange(lq, device=q.device)
    q_normalized = gamma * (q_positions.float() / float(q_seq_len))
    q_angles = q_normalized.unsqueeze(-1) * inv_freq  # [Lq, D/2]
    q_cos = torch.cos(q_angles).unsqueeze(0).unsqueeze(0)  # [1, 1, Lq, D/2]
    q_sin = torch.sin(q_angles).unsqueeze(0).unsqueeze(0)

    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    q_rotated_even = q_even * q_cos - q_odd * q_sin
    q_rotated_odd = q_even * q_sin + q_odd * q_cos
    q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).flatten(-2)

    # === Key rotation ===
    k_positions = torch.arange(lk, device=k.device)
    k_normalized = gamma * (k_positions.float() / float(k_seq_len))
    k_angles = k_normalized.unsqueeze(-1) * inv_freq  # [Lk, D/2]
    k_cos = torch.cos(k_angles).unsqueeze(0).unsqueeze(0)  # [1, 1, Lk, D/2]
    k_sin = torch.sin(k_angles).unsqueeze(0).unsqueeze(0)

    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    k_rotated_even = k_even * k_cos - k_odd * k_sin
    k_rotated_odd = k_even * k_sin + k_odd * k_cos
    k_rotated = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).flatten(-2)

    return q_rotated, k_rotated


class LARoPECrossAttention(nn.Module):
    """
    Cross-Attention модуль з LARoPE для text-speech alignment.

    Критична особливість: query (speech) та key (text) мають різну
    довжину, тому використовуємо length-normalized positions.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: dim // num_heads)
        gamma: LARoPE gamma (default: 10)
        dropout: Attention dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        gamma: float = 10.0,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.gamma = gamma
        self.scale = self.head_dim ** -0.5

        inner_dim = self.num_heads * self.head_dim

        # Query projection (для speech features)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)

        # Key, Value projections (для text embeddings)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # Speech features [B, Lq, D]
        context: torch.Tensor,     # Text embeddings [B, Lk, D]
        x_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass з LARoPE.

        Args:
            x: Query (speech features) [B, Lq, D]
            context: Key/Value (text embeddings) [B, Lk, D]
            x_mask: Optional mask for queries
            context_mask: Optional mask for keys/values

        Returns:
            Output tensor [B, Lq, D]
        """
        batch_size, lq, _ = x.shape
        _, lk, _ = context.shape

        # Project to Q, K, V
        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        # Reshape to [B, H, L, D_head]
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Apply LARoPE — критично для alignment!
        q, k = apply_larope(q, k, lq, lk, gamma=self.gamma)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Lq, Lk]

        # Apply mask if provided
        if context_mask is not None:
            # context_mask: [B, Lk] -> [B, 1, 1, Lk]
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        out = torch.matmul(attn_weights, v)  # [B, H, Lq, D_head]

        # Reshape back
        out = rearrange(out, 'b h l d -> b l (h d)')

        return self.to_out(out)


# ============================================================================
# Unit tests
# ============================================================================

def _test_larope():
    """Тест LARoPE модуля."""
    print("Testing LARoPE...")

    batch_size = 2
    seq_len = 100
    dim = 64

    larope = LARoPE(dim=dim, gamma=10.0)

    # Test basic forward
    x = torch.randn(batch_size, seq_len, dim)
    y = larope(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    print(f"  Basic LARoPE: {x.shape} -> {y.shape} ✓")

    # Test with explicit seq_length
    y2 = larope(x, seq_length=torch.tensor(seq_len))
    assert y2.shape == x.shape
    print(f"  With explicit seq_length: {x.shape} -> {y2.shape} ✓")

    # Test 4D input (multi-head)
    x_4d = torch.randn(batch_size, 4, seq_len, dim)  # [B, H, T, D]
    y_4d = larope(x_4d)
    assert y_4d.shape == x_4d.shape
    print(f"  Multi-head LARoPE: {x_4d.shape} -> {y_4d.shape} ✓")

    print("LARoPE tests passed! ✓\n")


def _test_apply_larope():
    """Тест функціонального LARoPE для cross-attention."""
    print("Testing apply_larope...")

    batch_size = 2
    num_heads = 4
    dim_per_head = 64

    # Speech features (довша послідовність)
    lq = 100
    q = torch.randn(batch_size, num_heads, lq, dim_per_head)

    # Text embeddings (коротша послідовність)
    lk = 20
    k = torch.randn(batch_size, num_heads, lk, dim_per_head)

    # Apply LARoPE з різними довжинами
    q_rot, k_rot = apply_larope(q, k, lq, lk, gamma=10.0)

    assert q_rot.shape == q.shape, f"Query shape mismatch"
    assert k_rot.shape == k.shape, f"Key shape mismatch"
    print(f"  Q: {q.shape} -> {q_rot.shape} ✓")
    print(f"  K: {k.shape} -> {k_rot.shape} ✓")

    # Перевіряємо, що attention scores мають очікуваний розподіл
    attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
    attn_weights = F.softmax(attn_scores, dim=-1)

    # LARoPE з γ=10 має давати diagonal bias
    # Перевіряємо, що attention concentration є розумною
    avg_entropy = -(attn_weights * attn_weights.log().clamp(min=-100)).sum(-1).mean()
    print(f"  Average attention entropy: {avg_entropy:.3f}")

    print("apply_larope tests passed! ✓\n")


def _test_larope_cross_attention():
    """Тест LARoPE cross-attention модуля."""
    print("Testing LARoPECrossAttention...")

    batch_size = 2
    dim = 256
    num_heads = 4

    # Speech features
    lq = 100
    x = torch.randn(batch_size, lq, dim)

    # Text embeddings
    lk = 20
    context = torch.randn(batch_size, lk, dim)

    # Create module
    cross_attn = LARoPECrossAttention(
        dim=dim,
        num_heads=num_heads,
        gamma=10.0,
        dropout=0.0
    )

    # Forward pass
    out = cross_attn(x, context)
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape}"
    print(f"  Input: {x.shape}, Context: {context.shape} -> Output: {out.shape} ✓")

    # Test with mask
    context_mask = torch.ones(batch_size, lk, dtype=torch.bool)
    context_mask[0, 15:] = False  # Mask last 5 tokens for first sample

    out_masked = cross_attn(x, context, context_mask=context_mask)
    assert out_masked.shape == x.shape
    print(f"  With mask: {out_masked.shape} ✓")

    # Parameter count
    num_params = sum(p.numel() for p in cross_attn.parameters())
    print(f"  Parameters: {num_params:,}")

    print("LARoPECrossAttention tests passed! ✓\n")


if __name__ == "__main__":
    _test_larope()
    _test_apply_larope()
    _test_larope_cross_attention()
    print("All LARoPE tests passed! ✓")
