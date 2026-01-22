"""
Transfer Learning: Адаптація існуючого encoder під Vocos

Використовує натренований encoder (24-dim, 228 mel) + додає projection layer 24→100
для compatibility з Vocos decoder.

Економія: ~40k ітерацій (тільки projection треба тренувати)
"""

import torch
import torch.nn as nn
from typing import Optional

from supertonic.models.speech_autoencoder import LatentEncoder


class EncoderToVocosAdapter(nn.Module):
    """
    Adapter для існуючого encoder → Vocos compatibility.
    
    Архітектура:
    Mel(228) → Pretrained Encoder → Latent(24) → Projection → Features(100) → Vocos
    
    Тренуємо тільки:
    - Mel resampler: 228 → 100 (ConvNeXt)
    - Projection: 24 → 100 (Linear)
    
    Frozen:
    - Encoder (вже натренований)
    - Vocos decoder (pretrained)
    
    Args:
        pretrained_encoder: Натренований encoder з checkpoint
        freeze_encoder: Чи freeze encoder (рекомендую True)
    """
    
    def __init__(
        self,
        pretrained_encoder: LatentEncoder,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        self.encoder = pretrained_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Pretrained encoder frozen")
        
        # Mel resampler: 228 mel bands → 100 (для Vocos compatibility)
        # Використовуємо ConvNeXt для adaptive resampling
        from supertonic.models.convnext import ConvNeXtBlock
        
        self.mel_adapter = nn.Sequential(
            nn.Conv1d(228, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            ConvNeXtBlock(dim=128, intermediate_dim=512, kernel_size=7, dilation=1),
            ConvNeXtBlock(dim=128, intermediate_dim=512, kernel_size=7, dilation=2),
            nn.Conv1d(128, 100, kernel_size=1)
        )
        
        # Latent projection: 24-dim → 100-dim (для Vocos decoder)
        self.latent_projection = nn.Sequential(
            nn.Linear(24, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 100),
            nn.LayerNorm(100)
        )
        
        # Learnable blending weight
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, mel_228: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_228: Mel-spectrogram [B, 228, T] на 24kHz
            
        Returns:
            features_100: [B, 100, T] для Vocos
        """
        # Path 1: Adapt mel directly
        mel_100 = self.mel_adapter(mel_228)  # [B, 100, T]
        
        # Path 2: Encode + project latents (with gradient control)
        if self.encoder.training:
            # Encoder frozen but still pass gradients through projection
            with torch.no_grad():
                latent_24 = self.encoder(mel_228)  # [B, 24, T]
        else:
            latent_24 = self.encoder(mel_228)
        
        latent_24 = latent_24.transpose(1, 2)  # [B, T, 24]
        features_100_proj = self.latent_projection(latent_24)  # [B, T, 100]
        features_100_proj = features_100_proj.transpose(1, 2)  # [B, 100, T]
        
        # Blend both paths
        alpha = torch.sigmoid(self.blend_weight)
        features = alpha * mel_100 + (1 - alpha) * features_100_proj
        
        return features


def load_pretrained_encoder(checkpoint_path: str, device: torch.device):
    """
    Завантажити encoder з checkpoint автоенкодера.
    
    Args:
        checkpoint_path: Шлях до checkpoint_80000.pt
        device: Device
        
    Returns:
        LatentEncoder
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Створити encoder з тими ж параметрами
    encoder = LatentEncoder(
        input_dim=228,
        hidden_dim=512,
        output_dim=24,
        num_blocks=10
    )
    
    # Завантажити weights
    encoder.load_state_dict(checkpoint["encoder"])
    
    print(f"✓ Loaded pretrained encoder from iteration {checkpoint.get('iteration', 'unknown')}")
    
    return encoder


def create_transfer_learning_adapter(
    pretrained_checkpoint: str,
    vocos_model: str = "charactr/vocos-mel-24khz",
    freeze_encoder: bool = True,
    freeze_vocos: bool = True,
    device: torch.device = torch.device("cuda")
):
    """
    Factory для створення transfer learning adapter.
    
    Args:
        pretrained_checkpoint: Шлях до checkpoint_80000.pt
        vocos_model: Pretrained Vocos model
        freeze_encoder: Freeze натренований encoder
        freeze_vocos: Freeze Vocos decoder
        device: Device
        
    Returns:
        Повний autoencoder: encoder + adapter + Vocos
    """
    from vocos import Vocos
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(pretrained_checkpoint, device)
    
    # Create adapter
    adapter = EncoderToVocosAdapter(
        pretrained_encoder=encoder,
        freeze_encoder=freeze_encoder
    )
    
    # Load Vocos
    print(f"Loading Vocos: {vocos_model}")
    vocos = Vocos.from_pretrained(vocos_model)
    
    if freeze_vocos:
        for param in vocos.parameters():
            param.requires_grad = False
        print("✓ Vocos decoder frozen")
    
    # Wrap в єдиний module
    class TransferLearningAutoencoder(nn.Module):
        def __init__(self, adapter, vocos):
            super().__init__()
            self.adapter = adapter
            self.vocos = vocos
            
            # Create mel transform once
            import torchaudio.transforms as T
            self.mel_transform = T.MelSpectrogram(
                sample_rate=24000,
                n_fft=2048,
                hop_length=512,
                n_mels=228,
                f_min=0,
                f_max=12000
            )
        
        def encode(self, audio: torch.Tensor) -> torch.Tensor:
            """Audio → Mel(228) → Features(100)"""
            # Extract 228-band mel
            mel_228 = self._extract_mel_228(audio)
            # Adapt to 100-band features
            features = self.adapter(mel_228)
            return features
        
        def decode(self, features: torch.Tensor) -> torch.Tensor:
            """Features(100) → Audio через Vocos"""
            return self.vocos.decode(features)
        
        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            """Reconstruction"""
            features = self.encode(audio)
            return self.decode(features)
        
        def _extract_mel_228(self, audio: torch.Tensor) -> torch.Tensor:
            """Extract 228-band mel як у оригінальному encoder."""
            if self.mel_transform.device != audio.device:
                self.mel_transform = self.mel_transform.to(audio.device)
            
            mel = self.mel_transform(audio)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            return mel
    
    model = TransferLearningAutoencoder(adapter, vocos).to(device)
    
    # Print trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return model
