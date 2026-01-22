"""
Vocos Wrapper для інтеграції з Supertonic v2.

Використовує pretrained Vocos decoder замість власного недотренованого decoder.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from vocos import Vocos
    VOCOS_AVAILABLE = True
except ImportError:
    VOCOS_AVAILABLE = False
    print("WARNING: vocos package not installed. Run: pip install vocos")


class VocosAdapter(nn.Module):
    """
    Adapter для використання Vocos з нашим encoder.
    
    Архітектура:
    Audio (44.1kHz) → Resample (24kHz) → Mel(100) → Encoder → Latent(24) 
                                                                    ↓
                                                    Vocos Decoder → Audio (24kHz)
    
    Args:
        encoder: Наш латентний encoder
        vocos_model_name: Pretrained Vocos model
        freeze_vocos: Чи freeze Vocos decoder
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        vocos_model_name: str = "charactr/vocos-mel-24khz",
        freeze_vocos: bool = True
    ):
        super().__init__()
        
        if not VOCOS_AVAILABLE:
            raise ImportError("vocos not installed. Run: pip install vocos")
        
        self.encoder = encoder
        
        # Load pretrained Vocos
        print(f"Loading pretrained Vocos: {vocos_model_name}")
        self.vocos = Vocos.from_pretrained(vocos_model_name)
        
        if freeze_vocos:
            # Freeze Vocos - тренуємо тільки encoder
            for param in self.vocos.parameters():
                param.requires_grad = False
            print("✓ Vocos decoder frozen")
        
        # Vocos очікує 100 mel bands на 24kHz
        self.target_sample_rate = 24000
        self.n_mels = 100
        
    def encode(self, audio: torch.Tensor, return_mel: bool = False):
        """
        Encode audio to latents через Vocos mel extractor.
        
        Args:
            audio: Waveform [B, T] на 24kHz
            return_mel: Якщо True, повертає також mel-spectrogram
            
        Returns:
            latent: [B, latent_dim, T_frames]
            mel (optional): [B, 100, T_frames]
        """
        # Extract mel-spectrogram з Vocos feature extractor
        mel = self.vocos.feature_extractor(audio)  # [B, 100, T]
        
        # Encode mel → latent
        latent = self.encoder(mel)  # [B, latent_dim, T]
        
        if return_mel:
            return latent, mel
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latents → audio через Vocos.
        
        ПРОБЛЕМА: Vocos очікує mel-spec [B, 100, T], а не латенти!
        Треба навчити encoder виводити 100-dim features як у Vocos.
        
        Args:
            latent: [B, latent_dim, T]
            
        Returns:
            audio: [B, T_audio]
        """
        # Якщо encoder виводить не 100 dim - треба проекція
        if latent.shape[1] != 100:
            raise ValueError(
                f"Encoder outputs {latent.shape[1]} dims, but Vocos expects 100 mel bands. "
                f"Train encoder with output_dim=100 or add projection layer."
            )
        
        # Decode через Vocos
        audio = self.vocos.decode(latent)
        
        return audio
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction: audio → latent → audio
        
        Args:
            audio: [B, T] на 24kHz
            
        Returns:
            reconstructed: [B, T] на 24kHz
        """
        latent = self.encode(audio)
        reconstructed = self.decode(latent)
        return reconstructed


class VocosLatentEncoder(nn.Module):
    """
    Encoder який виводить 100-dim features для Vocos.
    
    Замість 24-dim латентів, виводимо 100-dim (як mel-spectrogram) 
    щоб можна було передати в Vocos decoder.
    
    Args:
        input_dim: Розмірність вхідних mel (100 для Vocos)
        hidden_dim: Hidden dimension
        output_dim: Вихідна розмірність (100 для Vocos compatibility)
        num_blocks: Кількість ConvNeXt blocks
    """
    
    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 512,
        output_dim: int = 100,
        num_blocks: int = 6
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # ConvNeXt blocks
        from supertonic.models.convnext import ConvNeXtStack
        self.convnext = ConvNeXtStack(
            dim=hidden_dim,
            num_blocks=num_blocks,
            intermediate_dim=hidden_dim * 4,
            kernel_size=7,
            dilations=[1, 2, 4, 1, 2, 4][:num_blocks],
            causal=False
        )
        
        # Output projection to 100 dims (Vocos-compatible)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 100, T]
            
        Returns:
            features: [B, 100, T] - Vocos-compatible
        """
        # Input projection
        x = self.input_conv(mel)
        x = self.input_norm(x)
        
        # ConvNeXt processing
        x = self.convnext(x)  # [B, hidden, T]
        
        # Output projection
        x = x.transpose(1, 2)  # [B, T, hidden]
        x = self.output_linear(x)  # [B, T, 100]
        x = self.output_norm(x)
        x = x.transpose(1, 2)  # [B, 100, T]
        
        return x


def create_vocos_autoencoder(
    encoder_hidden_dim: int = 512,
    encoder_blocks: int = 6,
    vocos_model: str = "charactr/vocos-mel-24khz",
    freeze_vocos: bool = True
):
    """
    Factory function для створення encoder + Vocos decoder.
    
    Returns:
        VocosAdapter з encoder + pretrained Vocos
    """
    encoder = VocosLatentEncoder(
        input_dim=100,
        hidden_dim=encoder_hidden_dim,
        output_dim=100,
        num_blocks=encoder_blocks
    )
    
    adapter = VocosAdapter(
        encoder=encoder,
        vocos_model_name=vocos_model,
        freeze_vocos=freeze_vocos
    )
    
    return adapter
