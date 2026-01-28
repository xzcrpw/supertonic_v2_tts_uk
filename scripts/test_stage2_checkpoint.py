#!/usr/bin/env python3
"""
SUPERTONIC V2 - STAGE 2 (TEXT-TO-LATENT) CHECKPOINT TESTER
===========================================================
Інтерактивний скрипт для тестування Stage 2 чекпоінтів.

Без Duration Predictor (Stage 3) - використовуємо ground truth durations.

Можливості:
  - Synthesis з тексту використовуючи reference audio
  - Ground truth duration alignment (без Stage 3)
  - Порівняння з оригіналом
  - A/B тест різних чекпоінтів

Usage:
    python scripts/test_stage2_checkpoint.py
    python scripts/test_stage2_checkpoint.py --checkpoint outputs/text_to_latent/checkpoints/checkpoint_100000.pt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json
import random

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
from supertonic.models.text_to_latent import TextToLatent
from supertonic.losses.flow_matching_loss import compress_latents, decompress_latents
from supertonic.data.preprocessing import AudioProcessor
from supertonic.data.tokenizer import CharacterTokenizer

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def strip_ddp_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix from DDP state dict."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_autoencoder(checkpoint_path: str, config: dict, device: str = "cuda") -> Tuple:
    """Load autoencoder (encoder + decoder) from Stage 1 checkpoint."""
    print(f"\n📦 Loading autoencoder: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get encoder config
    ae_config = config.get("model", {}).get("autoencoder", config.get("autoencoder", {}))
    enc_config = ae_config.get("encoder", {})
    dec_config = ae_config.get("decoder", {})
    
    # Create encoder
    encoder = LatentEncoder(
        input_dim=enc_config.get("input_dim", 100),
        hidden_dim=enc_config.get("hidden_dim", 512),
        output_dim=enc_config.get("output_dim", 24),
        num_blocks=enc_config.get("num_blocks", 10),
        kernel_size=enc_config.get("kernel_size", 7),
    ).to(device)
    
    # Create decoder
    decoder = LatentDecoder(
        input_dim=dec_config.get("input_dim", 24),
        hidden_dim=dec_config.get("hidden_dim", 512),
        num_blocks=dec_config.get("num_blocks", 10),
        kernel_size=dec_config.get("kernel_size", 7),
        dilations=dec_config.get("dilations", [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]),
        causal=dec_config.get("causal", True),
        n_fft=config.get("audio", {}).get("n_fft", 1024),
        hop_length=config.get("audio", {}).get("hop_length", 256),
        use_hifigan=dec_config.get("use_hifigan", True),
        upsample_rates=dec_config.get("upsample_rates", [8, 8, 2, 2]),
        upsample_kernel_sizes=dec_config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
        upsample_initial_channel=dec_config.get("upsample_initial_channel", 512),
        resblock_kernel_sizes=dec_config.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilation_sizes=dec_config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
    ).to(device)
    
    # Load weights
    encoder.load_state_dict(strip_ddp_prefix(checkpoint["encoder"]))
    decoder.load_state_dict(strip_ddp_prefix(checkpoint["decoder"]))
    
    encoder.eval()
    decoder.eval()
    
    step = checkpoint.get("iteration", checkpoint.get("step", 0))
    print(f"   ✅ Autoencoder loaded (step {step:,})")
    
    return encoder, decoder


def load_text_to_latent(checkpoint_path: str, config: dict, device: str = "cuda") -> TextToLatent:
    """Load Text-to-Latent model from Stage 2 checkpoint."""
    print(f"\n📦 Loading Text-to-Latent: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Detect vocab_size from checkpoint weights
    state_dict = strip_ddp_prefix(checkpoint["model"])
    vocab_size = state_dict["text_encoder.char_embed.embedding.weight"].shape[0]
    print(f"   Detected vocab_size: {vocab_size}")
    
    # Get config
    ttl_config = config.get("model", {}).get("text_to_latent", config.get("text_to_latent", {}))
    ref_config = ttl_config.get("reference_encoder", {})
    text_config = ttl_config.get("text_encoder", {})
    vf_config = ttl_config.get("vector_field", {})
    
    # Create model with simplified interface
    model = TextToLatent(
        latent_dim=ref_config.get("input_dim", 144),
        vocab_size=vocab_size,  # Use detected vocab_size
        text_embed_dim=text_config.get("embed_dim", 128),
        text_hidden_dim=text_config.get("hidden_dim", 512),
        ref_hidden_dim=ref_config.get("hidden_dim", 128),
        vf_hidden_dim=vf_config.get("hidden_dim", 512),
        num_ref_vectors=ref_config.get("num_output_vectors", 50),
        sigma_min=config.get("flow_matching", {}).get("sigma_min", 1e-4),
        p_uncond=0.0,  # No dropout at inference
        cfg_scale=config.get("flow_matching", {}).get("cfg_scale", 3.0),
        gamma=config.get("larope", {}).get("gamma", 0.85),
    ).to(device)
    
    # Load weights
    model.load_state_dict(strip_ddp_prefix(checkpoint["model"]))
    model.eval()
    
    step = checkpoint.get("iteration", checkpoint.get("step", 0))
    print(f"   ✅ Text-to-Latent loaded (step {step:,})")
    
    return model


@torch.no_grad()
def synthesize_with_gt_duration(
    text: str,
    reference_audio: torch.Tensor,
    target_audio: torch.Tensor,  # For ground truth duration
    text_to_latent: TextToLatent,
    latent_encoder: LatentEncoder,
    latent_decoder: LatentDecoder,
    tokenizer: CharacterTokenizer,
    audio_processor: AudioProcessor,
    device: str = "cuda",
    nfe: int = 32,
    cfg_scale: float = 3.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Synthesize speech using ground truth duration (from target audio length).
    
    Returns:
        audio: Generated audio [T]
        info: Dict with timing and debug info
    """
    import time
    start_time = time.time()
    
    # 1. Tokenize text
    text_ids = tokenizer.encode(text)
    text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    
    # 2. Compute reference mel and encode (mel computation on CPU, then move to GPU)
    if reference_audio.dim() == 1:
        reference_audio = reference_audio.unsqueeze(0)
    ref_mel = audio_processor.compute_mel(reference_audio.cpu(), log_scale=True)  # [1, n_mels, T_ref]
    ref_mel = ref_mel.to(device)
    
    # Encode reference to latent
    ref_latent = latent_encoder(ref_mel)  # [1, 24, T_ref_latent]
    ref_compressed = compress_latents(ref_latent, compression_factor=6)  # [1, 144, T_ref_compressed]
    
    # 3. Get target duration from target audio
    if target_audio.dim() == 1:
        target_audio = target_audio.unsqueeze(0)
    target_mel = audio_processor.compute_mel(target_audio.cpu(), log_scale=True)
    target_mel = target_mel.to(device)
    target_latent = latent_encoder(target_mel)
    target_compressed = compress_latents(target_latent, compression_factor=6)
    target_length = target_compressed.shape[2]  # This is our target output length
    
    # 4. Use model's built-in generate() method
    generated_compressed = text_to_latent.generate(
        text=text_ids,
        ref_latent=ref_compressed,
        num_frames=target_length,
        nfe=nfe,
        cfg_scale=cfg_scale,
        text_mask=text_mask,
    )
    
    # 5. Decompress and decode
    generated_latent = decompress_latents(generated_compressed, compression_factor=6)  # [1, 24, T*6]
    generated_audio = latent_decoder(generated_latent)  # [1, T_audio]
    
    end_time = time.time()
    
    info = {
        "text_length": text_ids.shape[1],
        "ref_length": ref_compressed.shape[2],
        "target_latent_length": target_length,
        "generated_audio_length": generated_audio.shape[1],
        "inference_time": end_time - start_time,
        "nfe": nfe,
        "cfg_scale": cfg_scale,
    }
    
    return generated_audio.squeeze(0).cpu(), info


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """List available checkpoints, sorted by step."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    def get_step(path):
        name = path.stem
        parts = name.split("_")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0
    
    return sorted([str(cp) for cp in checkpoints], key=lambda x: get_step(Path(x)))


def discover_test_samples(manifest_path: str, num_samples: int = 10) -> List[dict]:
    """Get random samples from validation manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    # Filter samples with text
    samples = [s for s in samples if s.get("text")]
    
    # Random sample
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples


def plot_comparison(
    original_mel: torch.Tensor,
    generated_mel: torch.Tensor,
    save_path: str,
    title: str = ""
):
    """Plot mel spectrogram comparison."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Original
    im1 = axes[0].imshow(original_mel.numpy(), aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Original (Ground Truth)")
    axes[0].set_ylabel("Mel bin")
    plt.colorbar(im1, ax=axes[0])
    
    # Generated
    im2 = axes[1].imshow(generated_mel.numpy(), aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Generated (Stage 2)")
    axes[1].set_ylabel("Mel bin")
    axes[1].set_xlabel("Time")
    plt.colorbar(im2, ax=axes[1])
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   📊 Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Stage 2 (Text-to-Latent) checkpoint")
    parser.add_argument("--config", type=str, default="config/22khz_optimal.yaml")
    parser.add_argument("--autoencoder", type=str, default=None, help="Autoencoder checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Stage 2 checkpoint")
    parser.add_argument("--val-manifest", type=str, default="data/manifests_stage2/val.json")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2_test")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--nfe", type=int, default=32, help="ODE solver steps")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device
    
    print("\n" + "="*60)
    print("  🎤 SUPERTONIC V2 - Stage 2 Checkpoint Tester")
    print("="*60)
    
    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config)
    
    # Find autoencoder checkpoint
    if args.autoencoder:
        ae_path = args.autoencoder
    else:
        ae_paths = ["checkpoints/autoencoder/checkpoint_150000.pt",
                    "outputs/autoencoder_hifigan/checkpoints/autoencoder/checkpoint_150000.pt"]
        ae_path = None
        for p in ae_paths:
            if Path(p).exists():
                ae_path = p
                break
        if not ae_path:
            print("❌ No autoencoder checkpoint found!")
            return
    
    # Find Stage 2 checkpoint
    if args.checkpoint:
        s2_path = args.checkpoint
    else:
        s2_dirs = ["outputs/text_to_latent/checkpoints", "checkpoints/text_to_latent"]
        s2_path = None
        for d in s2_dirs:
            checkpoints = list_checkpoints(d)
            if checkpoints:
                s2_path = checkpoints[-1]  # Latest
                break
        if not s2_path:
            print("❌ No Stage 2 checkpoint found!")
            return
    
    # Load models
    latent_encoder, latent_decoder = load_autoencoder(ae_path, config, device)
    text_to_latent = load_text_to_latent(s2_path, config, device)
    
    # Create tokenizer and audio processor
    tokenizer = CharacterTokenizer(languages=config.get("languages", {}).get("supported", ["uk", "en"]))
    audio_processor = AudioProcessor(
        sample_rate=config["audio"]["sample_rate"],
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        n_mels=config["audio"]["n_mels"],
    )  # Stays on CPU, we'll move mel to GPU after
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test samples
    if Path(args.val_manifest).exists():
        samples = discover_test_samples(args.val_manifest, args.num_samples)
    else:
        print(f"❌ Validation manifest not found: {args.val_manifest}")
        return
    
    print(f"\n🔬 Testing {len(samples)} samples...")
    print(f"   NFE: {args.nfe}, CFG: {args.cfg_scale}")
    
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(samples)}")
        print(f"  Text: {sample['text'][:80]}...")
        print(f"  Speaker: {sample.get('speaker_id', 'unknown')}")
        
        # Load audio
        audio_path = Path(args.data_dir) / sample["audio_path"]
        if not audio_path.exists():
            print(f"  ⚠️ Audio not found: {audio_path}")
            continue
        
        target_audio, sr = torchaudio.load(audio_path)
        if sr != config["audio"]["sample_rate"]:
            target_audio = torchaudio.functional.resample(target_audio, sr, config["audio"]["sample_rate"])
        target_audio = target_audio.mean(dim=0)  # Mono
        
        # Use different reference audio (from same speaker if possible, or same audio)
        ref_audio = target_audio  # For now, use same audio as reference
        
        # Synthesize
        try:
            generated_audio, info = synthesize_with_gt_duration(
                text=sample["text"],
                reference_audio=ref_audio,
                target_audio=target_audio,
                text_to_latent=text_to_latent,
                latent_encoder=latent_encoder,
                latent_decoder=latent_decoder,
                tokenizer=tokenizer,
                audio_processor=audio_processor,
                device=device,
                nfe=args.nfe,
                cfg_scale=args.cfg_scale,
            )
            
            print(f"  ⏱️ Inference: {info['inference_time']:.2f}s")
            print(f"  📏 Generated: {generated_audio.shape[0] / config['audio']['sample_rate']:.2f}s")
            
            # Save audio
            sample_name = f"sample_{i+1:02d}"
            torchaudio.save(
                output_dir / f"{sample_name}_generated.wav",
                generated_audio.unsqueeze(0),
                config["audio"]["sample_rate"]
            )
            torchaudio.save(
                output_dir / f"{sample_name}_original.wav",
                target_audio.unsqueeze(0),
                config["audio"]["sample_rate"]
            )
            
            # Compute mel for comparison (on CPU)
            with torch.no_grad():
                orig_mel = audio_processor.compute_mel(target_audio.unsqueeze(0), log_scale=True)
                gen_mel = audio_processor.compute_mel(generated_audio.unsqueeze(0), log_scale=True)
                
                # Mel L1
                min_len = min(orig_mel.shape[2], gen_mel.shape[2])
                mel_l1 = F.l1_loss(orig_mel[:, :, :min_len], gen_mel[:, :, :min_len]).item()
            
            print(f"  📊 Mel L1: {mel_l1:.4f}")
            
            # Plot
            if MATPLOTLIB_AVAILABLE:
                plot_comparison(
                    orig_mel.squeeze(0),
                    gen_mel.squeeze(0),
                    str(output_dir / f"{sample_name}_comparison.png"),
                    title=f"Sample {i+1}: {sample['text'][:50]}..."
                )
            
            results.append({
                "sample": sample_name,
                "text": sample["text"],
                "speaker": sample.get("speaker_id", "unknown"),
                "mel_l1": mel_l1,
                "inference_time": info["inference_time"],
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("  📋 SUMMARY")
    print("="*60)
    
    if results:
        avg_mel_l1 = np.mean([r["mel_l1"] for r in results])
        avg_time = np.mean([r["inference_time"] for r in results])
        print(f"  Average Mel L1: {avg_mel_l1:.4f}")
        print(f"  Average inference time: {avg_time:.2f}s")
        print(f"  Output: {output_dir}")
        
        # Save results
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "checkpoint": s2_path,
                "autoencoder": ae_path,
                "nfe": args.nfe,
                "cfg_scale": args.cfg_scale,
                "avg_mel_l1": avg_mel_l1,
                "avg_inference_time": avg_time,
                "samples": results,
            }, f, indent=2, ensure_ascii=False)
    else:
        print("  No samples processed successfully")


if __name__ == "__main__":
    main()
