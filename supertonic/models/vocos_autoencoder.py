"""
Vocos-based Speech Autoencoder.

Uses pretrained Vocos as both encoder and decoder.
No training needed for autoencoder stage!

Encoder: audio → 100-band mel (Vocos feature_extractor)
Decoder: 100-band mel → audio (Vocos decode)
"""

import torch
import torch.nn as nn
from vocos import Vocos


class VocosSpeechAutoencoder(nn.Module):
    """
    Speech Autoencoder using pretrained Vocos.
    
    No training needed - Vocos already provides perfect reconstruction.
    This is a drop-in replacement for SpeechAutoencoder in the TTS pipeline.
    """
    
    def __init__(
        self,
        vocos_model: str = "charactr/vocos-mel-24khz",
        sample_rate: int = 24000,
        latent_dim: int = 100,  # Vocos mel bands
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        
        # Load pretrained Vocos
        print(f"Loading Vocos: {vocos_model}")
        self.vocos = Vocos.from_pretrained(vocos_model)
        
        # Freeze all Vocos parameters
        for param in self.vocos.parameters():
            param.requires_grad = False
        
        print(f"✓ Vocos loaded (frozen)")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Latent dim: {latent_dim} (mel bands)")
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent representation (mel spectrogram).
        
        Args:
            audio: [B, T] waveform at 24kHz
            
        Returns:
            latent: [B, 100, T'] mel spectrogram
        """
        with torch.no_grad():
            mel = self.vocos.feature_extractor(audio)
        return mel
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio.
        
        Args:
            latent: [B, 100, T'] mel spectrogram
            
        Returns:
            audio: [B, T] waveform at 24kHz
        """
        with torch.no_grad():
            audio = self.vocos.decode(latent)
        return audio
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio through Vocos.
        
        Args:
            audio: [B, T] input waveform
            
        Returns:
            audio: [B, T] reconstructed waveform
        """
        latent = self.encode(audio)
        return self.decode(latent)
    
    @property
    def hop_length(self) -> int:
        """Hop length for mel spectrogram."""
        return 256  # Vocos default
    
    @property
    def n_mels(self) -> int:
        """Number of mel bands."""
        return 100
    
    def get_latent_length(self, audio_length: int) -> int:
        """Calculate latent sequence length from audio length."""
        return audio_length // self.hop_length + 1


def create_vocos_autoencoder(device: torch.device = None) -> VocosSpeechAutoencoder:
    """Create Vocos-based autoencoder."""
    model = VocosSpeechAutoencoder()
    
    if device is not None:
        model = model.to(device)
    
    return model


# Compatibility wrapper for existing code
class VocosAutoencoderWrapper(nn.Module):
    """
    Wrapper to make VocosSpeechAutoencoder compatible with existing training code.
    
    Provides same interface as original SpeechAutoencoder:
    - encode() returns [B, latent_dim, T]
    - decode() takes [B, latent_dim, T]
    - compressed_dim property
    """
    
    def __init__(self, vocos_model: str = "charactr/vocos-mel-24khz"):
        super().__init__()
        self.autoencoder = VocosSpeechAutoencoder(vocos_model=vocos_model)
        
        # Properties for compatibility
        self.latent_dim = 100
        self.compressed_dim = 100  # Same as latent_dim for Vocos
        self.sample_rate = 24000
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to mel features."""
        return self.autoencoder.encode(audio)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode mel features to audio."""
        return self.autoencoder.decode(features)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio."""
        return self.autoencoder(audio)
    
    @property
    def vocos(self):
        return self.autoencoder.vocos
