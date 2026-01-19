"""
Supertonic v2 TTS - Реімплементація для української мови

Ультраефективна мультимовна система text-to-speech з 66M параметрів,
що досягає 167× швидше реального часу на споживчому обладнанні.

Архітектура:
- Speech Autoencoder (~47M): Vocos-based, 24-dim latent space
- Text-to-Latent (~19M): Conditional flow matching з LARoPE
- Duration Predictor (~0.5M): Utterance-level prediction

Підтримувані мови: English, Korean, Spanish, Portuguese, French, Ukrainian
"""

__version__ = "2.0.0-uk"
__author__ = "Reimplementation for Ukrainian TTS"

from supertonic.models import (
    SpeechAutoencoder,
    TextToLatent,
    DurationPredictor,
)

__all__ = [
    "SpeechAutoencoder",
    "TextToLatent",
    "DurationPredictor",
]
