"""
Supertonic v2 - Модулі моделей

Три ключові компоненти:
1. SpeechAutoencoder - кодування аудіо в латентний простір
2. TextToLatent - генерація латентів з тексту через flow matching
3. DurationPredictor - передбачення тривалості utterance
"""

from supertonic.models.speech_autoencoder import SpeechAutoencoder
from supertonic.models.text_to_latent import TextToLatent
from supertonic.models.duration_predictor import DurationPredictor
from supertonic.models.convnext import ConvNeXtBlock, ConvNeXtStack
from supertonic.models.attention import (
    MultiHeadAttention,
    SelfAttentionBlock,
    CrossAttentionBlock,
)
from supertonic.models.larope import LARoPE, apply_larope

__all__ = [
    "SpeechAutoencoder",
    "TextToLatent",
    "DurationPredictor",
    "ConvNeXtBlock",
    "ConvNeXtStack",
    "MultiHeadAttention",
    "SelfAttentionBlock",
    "CrossAttentionBlock",
    "LARoPE",
    "apply_larope",
]
