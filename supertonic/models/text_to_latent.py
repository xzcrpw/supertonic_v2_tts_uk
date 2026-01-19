"""
Text-to-Latent Module - генерація латентів з тексту через conditional flow matching

Архітектура (~19M параметрів):
1. Reference Encoder: Linear(144→128) → 6 ConvNeXt → 2 cross-attn → 50 vectors
2. Text Encoder: char_emb(128) → 6 ConvNeXt → 4 self-attn(RoPE) → 2 cross-attn
3. Vector Field (VF) Estimator: ConvNeXt + dilations + time + cross-attn(LARoPE)

Flow-matching:
- CFM з σ_min=1e-8
- Classifier-Free Guidance (p_uncond=0.05)
- LARoPE (γ=10) для text-speech alignment

Temporal compression:
- Input: compressed latents (144-dim, T/6 frames)
- Output: velocity field для ODE solving

Референс: Supertonic v2 paper (2509.11084)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict
from einops import rearrange

from supertonic.models.convnext import ConvNeXtBlock, ConvNeXtStack
from supertonic.models.attention import (
    MultiHeadAttention,
    SelfAttentionBlock,
    CrossAttentionBlock,
    TransformerBlock
)
from supertonic.models.larope import LARoPE, apply_larope, LARoPECrossAttention


class CharacterEmbedding(nn.Module):
    """
    Character-level embedding для multilingual text.

    NO G2P REQUIRED — модель сама навчається pronunciation!

    Підтримує:
    - Latin (en, es, pt, fr)
    - Cyrillic (uk, ru)
    - Korean Hangul (ko)
    - CJK characters
    - Punctuation, numbers, special symbols

    Args:
        vocab_size: Розмір словника (default 512 для extended multilingual)
        embed_dim: Розмірність embedding
        padding_idx: Index for padding token
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embed_dim: int = 128,
        padding_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # Character embedding table
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx
        )

        # Optional: language embedding (4-dim як у paper)
        self.lang_embedding = nn.Embedding(10, 4)  # 10 languages max

        # Projection if lang embedding used
        self.proj = nn.Linear(embed_dim + 4, embed_dim)

    def forward(
        self,
        chars: torch.Tensor,
        lang_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed characters.

        Args:
            chars: Character indices [B, L]
            lang_id: Optional language ID [B]

        Returns:
            embeddings: [B, L, D]
        """
        x = self.embedding(chars)  # [B, L, D]

        if lang_id is not None:
            lang_emb = self.lang_embedding(lang_id)  # [B, 4]
            lang_emb = lang_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, L, 4]
            x = torch.cat([x, lang_emb], dim=-1)  # [B, L, D+4]
            x = self.proj(x)  # [B, L, D]

        return x


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder - кодує reference audio для speaker conditioning.

    Архітектура:
    1. Linear(144 → 128)
    2. 6 ConvNeXt blocks (kernel=5, intermediate=512)
    3. 2 cross-attention layers
    4. Output: 50 fixed-size reference vectors

    Input: Compressed latents від Speech Autoencoder (144-dim)
    Output: 50 reference vectors (128-dim each)
    """

    def __init__(
        self,
        input_dim: int = 144,
        hidden_dim: int = 128,
        num_convnext_blocks: int = 6,
        num_cross_attn_layers: int = 2,
        num_output_vectors: int = 50,
        kernel_size: int = 5,
        intermediate_mult: int = 4,
        num_heads: int = 4,
        gamma: float = 10.0
    ):
        super().__init__()

        self.num_output_vectors = num_output_vectors
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ConvNeXt blocks
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_convnext_blocks,
            intermediate_dim=hidden_dim * intermediate_mult,
            kernel_size=kernel_size,
            dilations=None,
            causal=False
        )

        # Learnable query vectors (50 fixed vectors)
        self.query_vectors = nn.Parameter(
            torch.randn(1, num_output_vectors, hidden_dim) * 0.02
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                gamma=gamma
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ref_latent: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode reference audio latents.

        Args:
            ref_latent: Reference latents [B, C=144, T]
            ref_mask: Optional mask [B, T]

        Returns:
            ref_vectors: [B, 50, D=128]
        """
        batch_size = ref_latent.shape[0]

        # Input projection [B, T, 144] → [B, T, 128]
        x = ref_latent.transpose(1, 2)  # [B, T, C]
        x = self.input_proj(x)

        # ConvNeXt processing
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.convnext(x)
        x = x.transpose(1, 2)  # [B, T, C]

        # Cross-attention: query vectors attend to encoded features
        queries = self.query_vectors.expand(batch_size, -1, -1)  # [B, 50, D]

        for cross_attn in self.cross_attn_layers:
            queries = cross_attn(queries, x, context_mask=ref_mask)

        # Output normalization
        ref_vectors = self.output_norm(queries)

        return ref_vectors


class TextEncoder(nn.Module):
    """
    Text Encoder - кодує текст з reference conditioning.

    Архітектура:
    1. Character embedding → 128-dim
    2. 6 ConvNeXt blocks (kernel=5)
    3. 4 self-attention blocks (512 filters, 4 heads, RoPE)
    4. 2 cross-attention layers з reference vectors (LARoPE)

    Args:
        vocab_size: Character vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension after expansion
        num_convnext_blocks: Number of ConvNeXt blocks
        num_self_attn_blocks: Number of self-attention blocks
        num_cross_attn_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        kernel_size: ConvNeXt kernel size
        gamma: LARoPE gamma parameter
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_convnext_blocks: int = 6,
        num_self_attn_blocks: int = 4,
        num_cross_attn_layers: int = 2,
        num_heads: int = 4,
        kernel_size: int = 5,
        gamma: float = 10.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Character embedding
        self.char_embed = CharacterEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        # Projection to hidden dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # ConvNeXt blocks
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_convnext_blocks,
            intermediate_dim=hidden_dim * 4,
            kernel_size=kernel_size,
            dilations=None,
            causal=False
        )

        # Self-attention blocks з RoPE
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                use_rope=True
            )
            for _ in range(num_self_attn_blocks)
        ])

        # Cross-attention з reference vectors (LARoPE)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                context_dim=128,  # Reference vectors dim
                num_heads=num_heads,
                gamma=gamma
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        ref_vectors: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        lang_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text with reference conditioning.

        Args:
            text: Character indices [B, L]
            ref_vectors: Reference vectors [B, 50, 128]
            text_mask: Optional mask [B, L]
            lang_id: Optional language ID [B]

        Returns:
            text_encoding: [B, L, D=512]
        """
        # Character embedding
        x = self.char_embed(text, lang_id)  # [B, L, 128]

        # Project to hidden dim
        x = self.input_proj(x)  # [B, L, 512]

        # ConvNeXt processing
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.convnext(x)
        x = x.transpose(1, 2)  # [B, L, D]

        # Self-attention blocks
        for self_attn in self.self_attn_blocks:
            x = self_attn(x, mask=text_mask)

        # Cross-attention з reference vectors
        for cross_attn in self.cross_attn_layers:
            x = cross_attn(x, ref_vectors)

        # Output normalization
        x = self.output_norm(x)

        return x


class TimeEmbedding(nn.Module):
    """
    Time embedding для flow-matching.

    Sinusoidal embedding + MLP, схоже на diffusion models.

    Args:
        dim: Output dimension
        max_period: Maximum period for sinusoidal embedding
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()

        self.dim = dim
        self.max_period = max_period

        # MLP для time embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timestep.

        Args:
            t: Timestep [B] in range [0, 1]

        Returns:
            embedding: [B, D]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half_dim, device=t.device) / half_dim
        )

        # [B] × [D/2] → [B, D/2]
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)

        # Sinusoidal embedding
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # MLP
        emb = self.mlp(emb)

        return emb


class VectorFieldEstimator(nn.Module):
    """
    Vector Field (VF) Estimator для conditional flow matching.

    Передбачає velocity field v(z_t, z_ref, text, t) для ODE solving.

    Архітектура:
    - ConvNeXt blocks з dilations
    - Time conditioning через global embedding addition
    - Text/reference conditioning через cross-attention з LARoPE

    Args:
        latent_dim: Input latent dimension (144 compressed)
        hidden_dim: Hidden dimension
        text_dim: Text encoding dimension
        num_blocks: Number of ConvNeXt blocks
        kernel_size: ConvNeXt kernel size
        dilations: Dilation factors
        num_heads: Number of attention heads
        gamma: LARoPE gamma
    """

    def __init__(
        self,
        latent_dim: int = 144,
        hidden_dim: int = 512,
        text_dim: int = 512,
        num_blocks: int = 8,
        kernel_size: int = 7,
        dilations: Optional[List[int]] = None,
        num_heads: int = 4,
        gamma: float = 10.0
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 1, 2, 4, 8]

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)

        # Time embedding
        self.time_embed = TimeEmbedding(hidden_dim)

        # ConvNeXt blocks з cross-attention
        self.blocks = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            self.blocks.append(
                VFBlock(
                    dim=hidden_dim,
                    text_dim=text_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    num_heads=num_heads,
                    gamma=gamma
                )
            )

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,           # Noisy latent [B, 144, T]
        z_ref: torch.Tensor,         # Reference latent (masked) [B, 144, T]
        text_encoding: torch.Tensor, # Text encoding [B, L, 512]
        t: torch.Tensor,             # Timestep [B]
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate velocity field.

        Args:
            z_t: Noisy latent at time t
            z_ref: Reference latent (with masking)
            text_encoding: Encoded text
            t: Timestep in [0, 1]
            text_mask: Optional text mask

        Returns:
            velocity: Predicted velocity [B, 144, T]
        """
        batch_size = z_t.shape[0]

        # Concatenate z_t and z_ref for context
        # x = torch.cat([z_t, z_ref], dim=1)  # [B, 288, T]
        x = z_t + z_ref  # Additive conditioning (simpler)

        # Input projection
        x = self.input_proj(x)  # [B, hidden, T]

        # Time embedding (global)
        t_emb = self.time_embed(t)  # [B, hidden]

        # Process through blocks
        x = x.transpose(1, 2)  # [B, T, hidden]
        for block in self.blocks:
            x = block(x, text_encoding, t_emb, text_mask)

        # Output projection
        x = self.output_norm(x)
        velocity = self.output_proj(x)  # [B, T, 144]
        velocity = velocity.transpose(1, 2)  # [B, 144, T]

        return velocity


class VFBlock(nn.Module):
    """
    Vector Field Block - ConvNeXt + time conditioning + cross-attention.

    Args:
        dim: Hidden dimension
        text_dim: Text encoding dimension
        kernel_size: ConvNeXt kernel size
        dilation: Dilation factor
        num_heads: Number of attention heads
        gamma: LARoPE gamma
    """

    def __init__(
        self,
        dim: int,
        text_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        num_heads: int = 4,
        gamma: float = 10.0
    ):
        super().__init__()

        # ConvNeXt block
        self.convnext = ConvNeXtBlock(
            dim=dim,
            intermediate_dim=dim * 4,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=False
        )

        # Time conditioning (scale + shift)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2)
        )

        # Cross-attention з text (LARoPE)
        self.cross_attn = CrossAttentionBlock(
            dim=dim,
            context_dim=text_dim,
            num_heads=num_heads,
            gamma=gamma
        )

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,             # [B, T, D]
        text_encoding: torch.Tensor, # [B, L, D_text]
        t_emb: torch.Tensor,         # [B, D]
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # ConvNeXt (transpose for conv)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.convnext(x)
        x = x.transpose(1, 2)  # [B, T, D]

        # Time conditioning (AdaLN style)
        t_proj = self.time_proj(t_emb)  # [B, D*2]
        scale, shift = t_proj.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # [B, 1, D]
        shift = shift.unsqueeze(1)

        x = self.norm(x) * (1 + scale) + shift

        # Cross-attention з text
        x = self.cross_attn(x, text_encoding, context_mask=text_mask)

        return x


class TextToLatent(nn.Module):
    """
    Text-to-Latent Module - головний модуль для генерації латентів з тексту.

    Архітектура (~19M параметрів):
    - Reference Encoder → 50 reference vectors
    - Text Encoder → text encoding
    - Vector Field Estimator → velocity for flow-matching

    Flow-matching training:
    - z_t = (1 - (1-σ)t)·z_0 + t·z_1
    - target = z_1 - (1-σ)·z_0
    - loss = |m · (v_θ - target)|₁

    Inference:
    - ODE solving з Euler method
    - CFG для покращення якості

    Args:
        latent_dim: Latent dimension (24 uncompressed, 144 compressed)
        vocab_size: Character vocabulary size
        hidden_dim: Hidden dimension for encoders
        vf_hidden_dim: Hidden dimension for VF estimator
        sigma_min: Minimum sigma for flow-matching
        p_uncond: Probability of unconditional training (CFG)
        cfg_scale: Classifier-free guidance scale
    """

    def __init__(
        self,
        latent_dim: int = 144,  # Compressed: 24 * 6
        vocab_size: int = 512,
        text_embed_dim: int = 128,
        text_hidden_dim: int = 512,
        ref_hidden_dim: int = 128,
        vf_hidden_dim: int = 512,
        num_ref_vectors: int = 50,
        sigma_min: float = 1e-8,
        p_uncond: float = 0.05,
        cfg_scale: float = 3.0,
        gamma: float = 10.0
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond
        self.cfg_scale = cfg_scale

        # Reference Encoder
        self.reference_encoder = ReferenceEncoder(
            input_dim=latent_dim,
            hidden_dim=ref_hidden_dim,
            num_convnext_blocks=6,
            num_cross_attn_layers=2,
            num_output_vectors=num_ref_vectors,
            gamma=gamma
        )

        # Text Encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            num_convnext_blocks=6,
            num_self_attn_blocks=4,
            num_cross_attn_layers=2,
            gamma=gamma
        )

        # Vector Field Estimator
        self.vector_field = VectorFieldEstimator(
            latent_dim=latent_dim,
            hidden_dim=vf_hidden_dim,
            text_dim=text_hidden_dim,
            num_blocks=8,
            gamma=gamma
        )

        # Learnable unconditional embeddings (для CFG)
        self.uncond_ref = nn.Parameter(torch.randn(1, num_ref_vectors, ref_hidden_dim) * 0.02)
        self.uncond_text = nn.Parameter(torch.randn(1, 1, text_hidden_dim) * 0.02)

    def encode_reference(
        self,
        ref_latent: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode reference audio."""
        return self.reference_encoder(ref_latent, ref_mask)

    def encode_text(
        self,
        text: torch.Tensor,
        ref_vectors: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        lang_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode text with reference conditioning."""
        return self.text_encoder(text, ref_vectors, text_mask, lang_id)

    def estimate_velocity(
        self,
        z_t: torch.Tensor,
        z_ref: torch.Tensor,
        text_encoding: torch.Tensor,
        t: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Estimate velocity field."""
        return self.vector_field(z_t, z_ref, text_encoding, t, text_mask)

    def forward(
        self,
        z_1: torch.Tensor,           # Target latent [B, 144, T]
        text: torch.Tensor,          # Text [B, L]
        ref_latent: torch.Tensor,    # Reference latent [B, 144, T_ref]
        text_mask: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        lang_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Implements flow-matching loss computation:
        1. Sample t ~ U[0,1]
        2. Sample z_0 ~ N(0,I)
        3. Interpolate z_t
        4. Compute target velocity
        5. Estimate velocity
        6. Compute masked L1 loss

        Args:
            z_1: Target latent (from autoencoder)
            text: Text tokens
            ref_latent: Reference audio latent
            text_mask: Optional text mask
            ref_mask: Optional reference mask
            lang_id: Optional language ID

        Returns:
            Dict with 'loss', 'velocity_pred', 'velocity_target'
        """
        batch_size = z_1.shape[0]
        device = z_1.device

        # === Encode reference and text ===
        ref_vectors = self.encode_reference(ref_latent, ref_mask)
        text_encoding = self.encode_text(text, ref_vectors, text_mask, lang_id)

        # === CFG: occasionally use unconditional embeddings ===
        if self.training and self.p_uncond > 0:
            uncond_mask = torch.rand(batch_size, device=device) < self.p_uncond
            if uncond_mask.any():
                # Replace with unconditional embeddings
                ref_vectors = torch.where(
                    uncond_mask.view(-1, 1, 1),
                    self.uncond_ref.expand(batch_size, -1, -1),
                    ref_vectors
                )
                # Для text encoding потрібно перекодувати
                # (спрощення: просто замінюємо на uncond_text)
                uncond_text_expanded = self.uncond_text.expand(
                    batch_size, text_encoding.size(1), -1
                )
                text_encoding = torch.where(
                    uncond_mask.view(-1, 1, 1),
                    uncond_text_expanded,
                    text_encoding
                )

        # === Sample timestep t ~ U[0, 1] ===
        t = torch.rand(batch_size, device=device)

        # === Sample noise z_0 ~ N(0, I) ===
        z_0 = torch.randn_like(z_1)

        # === Interpolate: z_t = (1 - (1-σ)t)·z_0 + t·z_1 ===
        sigma = self.sigma_min
        z_t = (1 - (1 - sigma) * t.view(-1, 1, 1)) * z_0 + t.view(-1, 1, 1) * z_1

        # === Target velocity: z_1 - (1-σ)·z_0 ===
        velocity_target = z_1 - (1 - sigma) * z_0

        # === Create reference mask (random crop for anti-leakage) ===
        mask = self._create_reference_mask(z_1.shape, device)
        z_ref = (1 - mask) * z_1  # Keep unmasked parts of target

        # === Estimate velocity ===
        velocity_pred = self.estimate_velocity(z_t, z_ref, text_encoding, t, text_mask)

        # === Compute masked L1 loss ===
        loss = (mask * (velocity_pred - velocity_target).abs()).mean()

        return {
            'loss': loss,
            'velocity_pred': velocity_pred,
            'velocity_target': velocity_target,
            't': t
        }

    def _create_reference_mask(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        mask_ratio_range: Tuple[float, float] = (0.3, 0.7)
    ) -> torch.Tensor:
        """
        Create reference mask for training.

        Random contiguous region is masked (ones), rest is unmasked (zeros).
        This prevents information leakage from reference.
        """
        batch_size, c, t = shape

        masks = []
        for _ in range(batch_size):
            # Random mask ratio
            ratio = torch.rand(1).item() * (mask_ratio_range[1] - mask_ratio_range[0])
            ratio += mask_ratio_range[0]

            mask_len = int(t * ratio)
            start = torch.randint(0, t - mask_len + 1, (1,)).item()

            mask = torch.zeros(c, t, device=device)
            mask[:, start:start + mask_len] = 1.0
            masks.append(mask)

        return torch.stack(masks)

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,
        ref_latent: torch.Tensor,
        num_frames: int,
        nfe: int = 32,
        cfg_scale: Optional[float] = None,
        text_mask: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        lang_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate latents using Euler ODE solver.

        Args:
            text: Text tokens [B, L]
            ref_latent: Reference latent [B, 144, T_ref]
            num_frames: Number of output frames
            nfe: Number of function evaluations (Euler steps)
            cfg_scale: CFG scale (None = use default)
            text_mask: Optional text mask
            ref_mask: Optional reference mask
            lang_id: Optional language ID

        Returns:
            z: Generated latents [B, 144, num_frames]
        """
        batch_size = text.shape[0]
        device = text.device
        cfg_scale = cfg_scale or self.cfg_scale

        # Encode reference and text (conditional)
        ref_vectors = self.encode_reference(ref_latent, ref_mask)
        text_encoding = self.encode_text(text, ref_vectors, text_mask, lang_id)

        # Unconditional embeddings
        ref_vectors_uncond = self.uncond_ref.expand(batch_size, -1, -1)
        text_encoding_uncond = self.uncond_text.expand(batch_size, text_encoding.size(1), -1)

        # Initialize from noise
        z = torch.randn(batch_size, self.latent_dim, num_frames, device=device)

        # Reference latent (no masking at inference)
        z_ref = torch.zeros_like(z)

        # Euler integration from t=0 to t=1
        dt = 1.0 / nfe
        for step in range(nfe):
            t = torch.full((batch_size,), step * dt, device=device)

            # Conditional velocity
            v_cond = self.estimate_velocity(z, z_ref, text_encoding, t, text_mask)

            # Unconditional velocity
            v_uncond = self.estimate_velocity(z, z_ref, text_encoding_uncond, t, text_mask)

            # CFG combination
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond)

            # Euler step
            z = z + velocity * dt

        return z


# ============================================================================
# Unit tests
# ============================================================================

def _test_character_embedding():
    """Тест CharacterEmbedding."""
    print("Testing CharacterEmbedding...")

    embed = CharacterEmbedding(vocab_size=512, embed_dim=128)

    batch_size = 2
    seq_len = 50
    chars = torch.randint(0, 512, (batch_size, seq_len))

    out = embed(chars)
    assert out.shape == (batch_size, seq_len, 128)
    print(f"  {chars.shape} -> {out.shape} ✓")

    # With language ID
    lang_id = torch.tensor([0, 5])  # English, Ukrainian
    out_lang = embed(chars, lang_id)
    assert out_lang.shape == (batch_size, seq_len, 128)
    print(f"  With lang_id: {out_lang.shape} ✓")

    print("CharacterEmbedding tests passed! ✓\n")


def _test_reference_encoder():
    """Тест ReferenceEncoder."""
    print("Testing ReferenceEncoder...")

    encoder = ReferenceEncoder(
        input_dim=144,
        hidden_dim=128,
        num_output_vectors=50
    )

    batch_size = 2
    t_ref = 100
    ref_latent = torch.randn(batch_size, 144, t_ref)

    ref_vectors = encoder(ref_latent)
    assert ref_vectors.shape == (batch_size, 50, 128)
    print(f"  {ref_latent.shape} -> {ref_vectors.shape} ✓")

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("ReferenceEncoder tests passed! ✓\n")


def _test_text_encoder():
    """Тест TextEncoder."""
    print("Testing TextEncoder...")

    encoder = TextEncoder(
        vocab_size=512,
        embed_dim=128,
        hidden_dim=512
    )

    batch_size = 2
    text_len = 50
    text = torch.randint(0, 512, (batch_size, text_len))
    ref_vectors = torch.randn(batch_size, 50, 128)

    text_encoding = encoder(text, ref_vectors)
    assert text_encoding.shape == (batch_size, text_len, 512)
    print(f"  Text: {text.shape}, Ref: {ref_vectors.shape} -> {text_encoding.shape} ✓")

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {num_params:,}")

    print("TextEncoder tests passed! ✓\n")


def _test_vector_field_estimator():
    """Тест VectorFieldEstimator."""
    print("Testing VectorFieldEstimator...")

    vf = VectorFieldEstimator(
        latent_dim=144,
        hidden_dim=512,
        text_dim=512,
        num_blocks=8
    )

    batch_size = 2
    t_latent = 50
    text_len = 30

    z_t = torch.randn(batch_size, 144, t_latent)
    z_ref = torch.randn(batch_size, 144, t_latent)
    text_encoding = torch.randn(batch_size, text_len, 512)
    t = torch.rand(batch_size)

    velocity = vf(z_t, z_ref, text_encoding, t)
    assert velocity.shape == z_t.shape
    print(f"  z_t: {z_t.shape}, text: {text_encoding.shape} -> velocity: {velocity.shape} ✓")

    num_params = sum(p.numel() for p in vf.parameters())
    print(f"  Parameters: {num_params:,}")

    print("VectorFieldEstimator tests passed! ✓\n")


def _test_text_to_latent():
    """Тест повного TextToLatent модуля."""
    print("Testing TextToLatent...")

    model = TextToLatent(
        latent_dim=144,
        vocab_size=512,
        text_embed_dim=128,
        text_hidden_dim=512,
        vf_hidden_dim=512
    )

    batch_size = 2
    text_len = 50
    t_latent = 60
    t_ref = 100

    text = torch.randint(0, 512, (batch_size, text_len))
    z_1 = torch.randn(batch_size, 144, t_latent)
    ref_latent = torch.randn(batch_size, 144, t_ref)

    # Training forward
    output = model(z_1, text, ref_latent)
    print(f"  Loss: {output['loss'].item():.6f}")
    print(f"  Velocity pred: {output['velocity_pred'].shape}")
    print(f"  Velocity target: {output['velocity_target'].shape}")

    # Generation
    model.eval()
    generated = model.generate(
        text=text,
        ref_latent=ref_latent,
        num_frames=t_latent,
        nfe=8  # Quick test
    )
    print(f"  Generated: {generated.shape} ✓")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    ref_params = sum(p.numel() for p in model.reference_encoder.parameters())
    text_params = sum(p.numel() for p in model.text_encoder.parameters())
    vf_params = sum(p.numel() for p in model.vector_field.parameters())

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Reference encoder: {ref_params:,}")
    print(f"  Text encoder: {text_params:,}")
    print(f"  Vector field: {vf_params:,}")

    print("\nTextToLatent tests passed! ✓\n")


if __name__ == "__main__":
    _test_character_embedding()
    _test_reference_encoder()
    _test_text_encoder()
    _test_vector_field_estimator()
    _test_text_to_latent()
    print("All Text-to-Latent tests passed! ✓")
