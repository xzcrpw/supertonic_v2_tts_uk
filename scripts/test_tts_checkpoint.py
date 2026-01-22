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
from supertonic.models.speech_autoencoder import SpeechAutoencoder
from supertonic.losses.flow_matching_loss import ODESolver, decompress_latents
from supertonic.data.tokenizer import CharacterTokenizer


def load_autoencoder(checkpoint_path: str, config, device: torch.device):
    """Load pretrained autoencoder."""
    # Build autoencoder with config params
    autoencoder = SpeechAutoencoder(
        sample_rate=config.audio.sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=config.autoencoder.encoder.input_dim,
        latent_dim=config.autoencoder.encoder.output_dim,
        hidden_dim=config.autoencoder.encoder.hidden_dim,
        num_encoder_blocks=config.autoencoder.encoder.num_blocks,
        num_decoder_blocks=config.autoencoder.decoder.num_blocks,
        decoder_dilations=list(config.autoencoder.decoder.dilations)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Checkpoint stores encoder/decoder separately
    if "encoder" in checkpoint:
        autoencoder.encoder.load_state_dict(checkpoint["encoder"])
        autoencoder.decoder.load_state_dict(checkpoint["decoder"])
        if "mpd" in checkpoint:
            autoencoder.mpd.load_state_dict(checkpoint["mpd"])
        if "mrd" in checkpoint:
            autoencoder.mrd.load_state_dict(checkpoint["mrd"])
        print(f"Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
    elif "autoencoder" in checkpoint:
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
    # Load checkpoint first to get actual vocab size
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get actual vocab size from checkpoint
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Extract vocab size from embedding weight shape
    embed_weight = state_dict.get("text_encoder.char_embed.embedding.weight")
    if embed_weight is not None:
        actual_vocab_size = embed_weight.shape[0]
        print(f"Detected vocab_size from checkpoint: {actual_vocab_size}")
    else:
        actual_vocab_size = config.text_to_latent.text_encoder.vocab_size
    
    # Use actual config structure
    ttl_cfg = config.text_to_latent
    
    text_to_latent = TextToLatent(
        latent_dim=ttl_cfg.reference_encoder.input_dim,  # 144 (compressed)
        vocab_size=actual_vocab_size,  # From checkpoint!
        text_embed_dim=ttl_cfg.text_encoder.embed_dim,
        text_hidden_dim=ttl_cfg.text_encoder.hidden_dim,
        ref_hidden_dim=ttl_cfg.reference_encoder.hidden_dim,
        vf_hidden_dim=ttl_cfg.vector_field.hidden_dim,
        num_ref_vectors=ttl_cfg.reference_encoder.num_output_vectors,
        sigma_min=config.flow_matching.sigma_min,
        p_uncond=config.flow_matching.p_uncond,
        cfg_scale=config.flow_matching.cfg_scale,
        gamma=config.larope.gamma
    )
    
    text_to_latent.load_state_dict(state_dict)
    print(f"Loaded TTS from iteration {checkpoint.get('iteration', 'unknown')}")
    
    text_to_latent.to(device)
    text_to_latent.eval()
    
    return text_to_latent


@torch.no_grad()
def synthesize(
    text: str,
    text_to_latent: TextToLatent,
    autoencoder: SpeechAutoencoder,
    tokenizer: CharacterTokenizer,
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
        tokenizer: Character tokenizer
        reference_audio: Optional reference audio for voice cloning
        device: Device
        num_steps: ODE solver steps
        cfg_scale: CFG scale
        duration_seconds: Target duration (if None, estimate from text)
        
    Returns:
        Synthesized audio waveform
    """
    # Tokenize text
    text_ids = tokenizer.encode(text, return_tensor=True)
    
    # Clamp token IDs to valid range (vocab_size from checkpoint)
    vocab_size = text_to_latent.text_encoder.char_embed.vocab_size
    text_ids = text_ids.clamp(0, vocab_size - 1)
    
    print(f"Text tokens: {text_ids.tolist()}, max_id={text_ids.max().item()}, vocab_size={vocab_size}")
    
    text_ids = text_ids.to(device).unsqueeze(0)
    
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
        mel = autoencoder.mel_spec(reference_audio)
        ref_latents = autoencoder.encoder(mel)
        
        # Compress reference
        from supertonic.losses.flow_matching_loss import compress_latents
        ref_compressed = compress_latents(ref_latents, compression_factor=6)
        
        # Encode reference
        reference_encoding = text_to_latent.encode_reference(ref_compressed)
        z_ref = ref_compressed[:, :, :compressed_len]  # Match output length
    else:
        # Use learnable unconditional embeddings
        reference_encoding = text_to_latent.uncond_ref.expand(1, -1, -1)
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
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(languages=["uk", "en"])
    
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
        tokenizer=tokenizer,
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
