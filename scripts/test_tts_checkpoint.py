#!/usr/bin/env python3
"""
Test Text-to-Latent checkpoint - synthesize speech from text.

Usage:
    python scripts/test_tts_checkpoint.py \
        --tts-checkpoint checkpoints/tts/checkpoint_10000.pt \
        --autoencoder-checkpoint checkpoints/autoencoder/checkpoint_80000.pt \
        --text "Привіт, це тестове повідомлення українською мовою." \
        --output test_output.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from supertonic.models.text_to_latent import TextToLatent
from supertonic.models.speech_autoencoder import SpeechAutoencoder, LatentEncoder, LatentDecoder
from supertonic.losses.flow_matching_loss import ODESolver, decompress_latents
from supertonic.data.text_processor import TextProcessor


def load_autoencoder(checkpoint_path: str, config, device: torch.device):
    """Load pretrained autoencoder."""
    # Build encoder
    encoder = LatentEncoder(
        input_dim=config.autoencoder.encoder.input_dim,
        hidden_dim=config.autoencoder.encoder.hidden_dim,
        output_dim=config.autoencoder.encoder.output_dim,
        num_blocks=config.autoencoder.encoder.num_blocks
    )
    
    # Build decoder (Vocos)
    decoder = LatentDecoder(
        latent_dim=config.autoencoder.decoder.latent_dim,
        hidden_dim=config.autoencoder.decoder.hidden_dim,
        num_layers=config.autoencoder.decoder.num_layers,
        n_fft=config.autoencoder.decoder.n_fft,
        hop_length=config.autoencoder.decoder.hop_length,
        sample_rate=config.audio.sample_rate
    )
    
    # Build full autoencoder
    autoencoder = SpeechAutoencoder(encoder, decoder)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "autoencoder" in checkpoint:
        autoencoder.load_state_dict(checkpoint["autoencoder"])
    elif "model" in checkpoint:
        autoencoder.load_state_dict(checkpoint["model"])
    else:
        autoencoder.load_state_dict(checkpoint)
    
    autoencoder.to(device)
    autoencoder.eval()
    
    return autoencoder


def load_tts(checkpoint_path: str, config, device: torch.device):
    """Load Text-to-Latent model."""
    text_to_latent = TextToLatent(
        vocab_size=config.text_to_latent.vocab_size,
        text_dim=config.text_to_latent.text_dim,
        ref_dim=config.text_to_latent.ref_dim,
        latent_dim=config.text_to_latent.latent_dim,
        num_languages=config.text_to_latent.num_languages,
        text_encoder_layers=config.text_to_latent.text_encoder_layers,
        ref_encoder_layers=config.text_to_latent.ref_encoder_layers,
        vf_hidden_dim=config.text_to_latent.vf_hidden_dim,
        vf_num_blocks=config.text_to_latent.vf_num_blocks,
        gamma=config.larope.gamma
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint:
        text_to_latent.load_state_dict(checkpoint["model"])
    else:
        text_to_latent.load_state_dict(checkpoint)
    
    text_to_latent.to(device)
    text_to_latent.eval()
    
    return text_to_latent


@torch.no_grad()
def synthesize(
    text: str,
    text_to_latent: TextToLatent,
    autoencoder: SpeechAutoencoder,
    text_processor: TextProcessor,
    reference_audio: torch.Tensor = None,
    device: torch.device = torch.device("cuda"),
    num_steps: int = 32,
    cfg_scale: float = 3.0,
    duration_seconds: float = None
) -> torch.Tensor:
    """
    Synthesize speech from text.
    
    Args:
        text: Input text
        text_to_latent: TTS model
        autoencoder: Autoencoder for decoding
        text_processor: Text tokenizer
        reference_audio: Optional reference audio for voice cloning
        device: Device
        num_steps: ODE solver steps
        cfg_scale: CFG scale
        duration_seconds: Target duration (if None, estimate from text)
        
    Returns:
        Synthesized audio waveform
    """
    # Tokenize text
    text_ids = text_processor.encode(text)
    text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Estimate duration if not provided
    if duration_seconds is None:
        # ~0.1 sec per character for Ukrainian
        duration_seconds = max(1.0, len(text) * 0.08)
    
    # Calculate latent sequence length
    # 44100 Hz, hop=512, compression=6
    num_frames = int(duration_seconds * 44100 / 512)
    compressed_len = num_frames // 6
    
    print(f"Text: {text}")
    print(f"Estimated duration: {duration_seconds:.2f}s ({compressed_len} latent frames)")
    
    # Get reference encoding
    if reference_audio is not None:
        # Encode reference audio
        mel = autoencoder.encoder.compute_mel(reference_audio)
        ref_latents = autoencoder.encoder(mel)
        
        # Compress reference
        from supertonic.losses.flow_matching_loss import compress_latents
        ref_compressed = compress_latents(ref_latents, compression_factor=6)
        
        # Encode reference
        reference_encoding = text_to_latent.encode_reference(ref_compressed)
        z_ref = ref_compressed[:, :, :50]  # First 50 frames as reference
    else:
        # Use zeros for unconditional generation
        reference_encoding = torch.zeros(1, 50, 512, device=device)
        z_ref = torch.zeros(1, 144, compressed_len, device=device)
    
    # Encode text with reference
    text_encoding = text_to_latent.encode_text(
        text_ids,
        reference_encoding,
        text_mask=None,
        lang_id=torch.zeros(1, dtype=torch.long, device=device)  # Ukrainian
    )
    
    # ODE solve to generate latents
    solver = ODESolver(nfe=num_steps, cfg_scale=cfg_scale)
    
    generated_latents = solver.solve(
        model=text_to_latent.vector_field,
        z_shape=(1, 144, compressed_len),
        text_encoding=text_encoding,
        reference_encoding=reference_encoding,
        z_ref=z_ref,
        device=device
    )
    
    # Decompress latents
    latents = decompress_latents(generated_latents, compression_factor=6)
    
    # Decode to audio
    audio = autoencoder.decoder(latents)
    
    return audio.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Test TTS Checkpoint")
    parser.add_argument("--tts-checkpoint", type=str, required=True,
                        help="Path to TTS checkpoint")
    parser.add_argument("--autoencoder-checkpoint", type=str, required=True,
                        help="Path to autoencoder checkpoint")
    parser.add_argument("--config", type=str, default="config/rtx6000_optimal.yaml",
                        help="Config file")
    parser.add_argument("--text", type=str, 
                        default="Привіт! Це тестове повідомлення синтезу мови.",
                        help="Text to synthesize")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference audio for voice cloning")
    parser.add_argument("--output", type=str, default="test_tts_output.wav",
                        help="Output audio file")
    parser.add_argument("--num-steps", type=int, default=32,
                        help="ODE solver steps")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--duration", type=float, default=None,
                        help="Target duration in seconds")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Load models
    print(f"Loading autoencoder from {args.autoencoder_checkpoint}...")
    autoencoder = load_autoencoder(args.autoencoder_checkpoint, config, device)
    
    print(f"Loading TTS from {args.tts_checkpoint}...")
    text_to_latent = load_tts(args.tts_checkpoint, config, device)
    
    # Load text processor
    text_processor = TextProcessor(
        vocab_path=config.data.vocab_path if hasattr(config.data, 'vocab_path') else None
    )
    
    # Load reference audio if provided
    reference_audio = None
    if args.reference:
        reference_audio, sr = torchaudio.load(args.reference)
        if sr != config.audio.sample_rate:
            reference_audio = torchaudio.functional.resample(
                reference_audio, sr, config.audio.sample_rate
            )
        reference_audio = reference_audio.to(device)
        print(f"Loaded reference: {args.reference}")
    
    # Synthesize
    print("\nSynthesizing...")
    audio = synthesize(
        text=args.text,
        text_to_latent=text_to_latent,
        autoencoder=autoencoder,
        text_processor=text_processor,
        reference_audio=reference_audio,
        device=device,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        duration_seconds=args.duration
    )
    
    # Save output
    output_path = Path(args.output)
    torchaudio.save(str(output_path), audio.cpu(), config.audio.sample_rate)
    print(f"\nSaved to: {output_path}")
    print(f"Duration: {audio.shape[-1] / config.audio.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
