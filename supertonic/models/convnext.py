"""
ConvNeXt Block - базовий будівельний блок для всіх модулів Supertonic v2

ConvNeXt — модернізована CNN архітектура з transformer-подібними елементами:
- Depthwise separable convolution
- Inverted bottleneck (expand 4x)
- Layer scale для стабілізації глибоких мереж
- GELU activation

Референс: "A ConvNet for the 2020s" (Liu et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from einops import rearrange


class LayerNorm1d(nn.Module):
    """
    Layer Normalization для 1D послідовностей.
    Підтримує формат channels-first [B, C, T] та channels-last [B, T, C].
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_first"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [B, C, T] -> normalize over C
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
        else:
            raise ValueError(f"Unknown data_format: {self.data_format}")


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block — основний будівельний блок Supertonic v2.

    Архітектура:
    1. Depthwise convolution (groups=dim) — локальні патерни
    2. LayerNorm
    3. Pointwise conv 1 (expand 4x) — inverted bottleneck
    4. GELU activation
    5. Pointwise conv 2 (contract back)
    6. Layer scale (γ ≈ 1e-6 init для глибоких мереж)
    7. Residual connection

    Args:
        dim: Кількість каналів (вхід = вихід)
        intermediate_dim: Розмір прихованого шару (default: dim * 4)
        kernel_size: Розмір ядра depthwise conv (default: 7)
        dilation: Dilation factor для dilated convolutions
        layer_scale_init: Початкове значення layer scale
        causal: Чи використовувати causal convolution (для streaming)
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: Optional[int] = None,
        kernel_size: int = 7,
        dilation: int = 1,
        layer_scale_init: float = 1e-6,
        causal: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        intermediate_dim = intermediate_dim or dim * 4

        # Causal padding: весь padding зліва
        effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
        if causal:
            self.padding = (effective_kernel - 1, 0)  # Left padding only
        else:
            self.padding = ((effective_kernel - 1) // 2, (effective_kernel - 1) // 2)

        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Depthwise convolution (groups=dim для spatial mixing)
        # Не включаємо padding тут — робимо вручну для causal
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=0,  # Manual padding
            groups=dim,
            dilation=dilation
        )

        # LayerNorm у channels-last форматі
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Inverted bottleneck: expand → GELU → contract
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        # Layer scale — критично для глибоких мереж
        # Ініціалізуємо малим значенням для стабільності
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C, T] (same shape)
        """
        residual = x

        # Manual padding для causal conv
        x = F.pad(x, self.padding)

        # Depthwise convolution
        x = self.dwconv(x)

        # [B, C, T] -> [B, T, C] для LayerNorm та Linear
        x = x.transpose(1, 2)
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        x = self.gamma * x
        x = self.dropout(x)

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Residual connection
        return residual + x


class ConvNeXtStack(nn.Module):
    """
    Стек ConvNeXt блоків з опціональними dilations.

    Використовується в:
    - Latent Encoder: 10 blocks, dilations=[1]*10
    - Latent Decoder: 10 blocks, dilations=[1,2,4,1,2,4,1,1,1,1]
    - Text Encoder: 6 blocks
    - Reference Encoder: 6 blocks
    - VF Estimator: 8 blocks з dilations

    Args:
        dim: Кількість каналів
        num_blocks: Кількість ConvNeXt блоків
        intermediate_dim: Розмір прихованого шару
        kernel_size: Розмір ядра
        dilations: Список dilation factors для кожного блоку
        causal: Використовувати causal convolutions
        dropout: Dropout rate
        gradient_checkpointing: Use gradient checkpointing to save memory
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        intermediate_dim: Optional[int] = None,
        kernel_size: int = 7,
        dilations: Optional[List[int]] = None,
        causal: bool = False,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-6,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.gradient_checkpointing = gradient_checkpointing

        # Якщо dilations не задано — всі 1
        if dilations is None:
            dilations = [1] * num_blocks
        else:
            assert len(dilations) == num_blocks, \
                f"Dilations length ({len(dilations)}) must match num_blocks ({num_blocks})"

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                kernel_size=kernel_size,
                dilation=dilations[i],
                layer_scale_init=layer_scale_init,
                causal=causal,
                dropout=dropout
            )
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через всі блоки.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C, T]
        """
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class ConvNeXtDownsample(nn.Module):
    """
    Downsampling модуль на основі ConvNeXt.
    Зменшує temporal resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2
    ):
        super().__init__()
        self.norm = LayerNorm1d(in_channels, data_format="channels_first")
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=factor,
            stride=factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class ConvNeXtUpsample(nn.Module):
    """
    Upsampling модуль на основі ConvNeXt.
    Збільшує temporal resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2
    ):
        super().__init__()
        self.norm = LayerNorm1d(in_channels, data_format="channels_first")
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=factor,
            stride=factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


# ============================================================================
# Unit tests
# ============================================================================

def _test_convnext_block():
    """Тест ConvNeXt блоку."""
    print("Testing ConvNeXtBlock...")

    batch_size = 4
    channels = 512
    seq_len = 100

    # Standard block
    block = ConvNeXtBlock(
        dim=channels,
        intermediate_dim=2048,
        kernel_size=7,
        dilation=1,
        causal=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    y = block(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  Standard block: input {x.shape} -> output {y.shape} ✓")

    # Causal block
    block_causal = ConvNeXtBlock(
        dim=channels,
        kernel_size=7,
        dilation=1,
        causal=True
    )

    y_causal = block_causal(x)
    assert y_causal.shape == x.shape, f"Causal shape mismatch"
    print(f"  Causal block: input {x.shape} -> output {y_causal.shape} ✓")

    # Dilated block
    block_dilated = ConvNeXtBlock(
        dim=channels,
        kernel_size=7,
        dilation=4,
        causal=False
    )

    y_dilated = block_dilated(x)
    assert y_dilated.shape == x.shape, f"Dilated shape mismatch"
    print(f"  Dilated block (d=4): input {x.shape} -> output {y_dilated.shape} ✓")

    # Parameter count
    num_params = sum(p.numel() for p in block.parameters())
    print(f"  Parameters per block: {num_params:,}")

    print("ConvNeXtBlock tests passed! ✓\n")


def _test_convnext_stack():
    """Тест стеку ConvNeXt блоків."""
    print("Testing ConvNeXtStack...")

    batch_size = 4
    channels = 512
    seq_len = 100

    # Encoder-style stack (no dilation)
    encoder_stack = ConvNeXtStack(
        dim=channels,
        num_blocks=10,
        intermediate_dim=2048,
        kernel_size=7,
        dilations=None,
        causal=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    y = encoder_stack(x)

    assert y.shape == x.shape
    print(f"  Encoder stack (10 blocks): {x.shape} -> {y.shape} ✓")

    # Decoder-style stack (with dilations)
    decoder_dilations = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
    decoder_stack = ConvNeXtStack(
        dim=channels,
        num_blocks=10,
        intermediate_dim=2048,
        kernel_size=7,
        dilations=decoder_dilations,
        causal=True  # Decoder is causal
    )

    y_decoder = decoder_stack(x)
    assert y_decoder.shape == x.shape
    print(f"  Decoder stack (10 blocks, dilated, causal): {x.shape} -> {y_decoder.shape} ✓")

    # Parameter count
    encoder_params = sum(p.numel() for p in encoder_stack.parameters())
    decoder_params = sum(p.numel() for p in decoder_stack.parameters())
    print(f"  Encoder params: {encoder_params:,}")
    print(f"  Decoder params: {decoder_params:,}")

    print("ConvNeXtStack tests passed! ✓\n")


if __name__ == "__main__":
    _test_convnext_block()
    _test_convnext_stack()
    print("All ConvNeXt tests passed! ✓")
