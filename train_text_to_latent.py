"""
Training Script для Text-to-Latent Module

Тренування flow-matching моделі для генерації латентів з тексту.

Ключові особливості:
- Context-Sharing Batch Expansion (B=64, Ke=4)
- Conditional Flow Matching (CFM) з σ_min=1e-8
- Classifier-Free Guidance (p_uncond=0.05)
- LARoPE (γ=10) для text-speech alignment
- Learning rate halving кожні 300k iterations

Конфігурація:
- Optimizer: AdamW, lr=5e-4
- Total iterations: 700k
- Effective batch size: 256 (64 × 4)

Hardware:
- Рекомендовано: 4× RTX 4090/5090
- Мінімум: 1× RTX 3090 з B=16, Ke=4

Референс: Supertonic v2 paper (2509.11084), Matcha-TTS
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.text_to_latent import (
    TextToLatent,
    ReferenceEncoder,
    TextEncoder,
    VectorFieldEstimator
)
from supertonic.models.speech_autoencoder import LatentEncoder
from supertonic.losses.flow_matching_loss import (
    FlowMatchingLoss,
    compress_latents,
    decompress_latents,
    ODESolver
)
from supertonic.data.dataset import TTSDataset, ContextSharingDataset
from supertonic.data.collate import tts_collate_fn
from supertonic.data.preprocessing import AudioProcessor
from supertonic.data.tokenizer import CharacterTokenizer


def setup_distributed():
    """Ініціалізує distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    path: Path,
    iteration: int,
    text_to_latent: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Dict
):
    """Зберігає checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "model": text_to_latent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at iteration {iteration}")


def load_checkpoint(
    path: Path,
    text_to_latent: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None
) -> int:
    """Завантажує checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    
    text_to_latent.load_state_dict(checkpoint["model"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    
    return checkpoint["iteration"]


def context_sharing_batch_expansion(
    batch: Dict[str, torch.Tensor],
    expansion_factor: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Context-Sharing Batch Expansion.
    
    Ключова інновація Supertonic v2:
    - Для кожного sample: один раз кодуємо text/reference
    - Потім застосовуємо Ke різних noise samples з різними timesteps
    
    Це зменшує memory usage та стабілізує alignment learning.
    
    Args:
        batch: Original batch
        expansion_factor: Ke - number of expansions
        
    Returns:
        Expanded batch
    """
    expanded = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Repeat each sample Ke times
            expanded[key] = value.repeat_interleave(expansion_factor, dim=0)
        else:
            expanded[key] = value
    
    # Add expansion indices for different noise/timesteps
    batch_size = batch["mel"].size(0)
    expansion_idx = torch.arange(expansion_factor).repeat(batch_size)
    expanded["expansion_idx"] = expansion_idx
    
    return expanded


def train_step(
    batch: Dict[str, torch.Tensor],
    text_to_latent: nn.Module,
    latent_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: FlowMatchingLoss,
    scaler: GradScaler,
    device: torch.device,
    expansion_factor: int = 4,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Один training step з context-sharing.
    """
    # Move to device
    mel = batch["mel"].to(device)
    text_ids = batch["text_ids"].to(device)
    text_mask = batch["text_mask"].to(device)
    reference_mel = batch["reference_mel"].to(device)
    reference_mask = batch["reference_mask"].to(device)
    lang_ids = batch.get("lang_ids", None)
    if lang_ids is not None:
        lang_ids = lang_ids.to(device)
    
    optimizer.zero_grad()
    
    with autocast(enabled=use_amp, dtype=torch.bfloat16):
        # 1. Encode target audio to latents (з pretrained autoencoder)
        with torch.no_grad():
            latent_encoder.eval()
            target_latent = latent_encoder(mel)  # [B, 24, T]
            
            # Temporal compression: [B, 24, T] → [B, 144, T/6]
            target_compressed = compress_latents(target_latent, compression_factor=6)
        
        # 2. Context-sharing batch expansion
        # Expand batch Ke times (same text/reference, different noise)
        batch_size = mel.size(0)
        expanded_size = batch_size * expansion_factor
        
        # Expand conditioning (shared across noise samples)
        text_ids_exp = text_ids.repeat_interleave(expansion_factor, dim=0)
        text_mask_exp = text_mask.repeat_interleave(expansion_factor, dim=0)
        reference_mel_exp = reference_mel.repeat_interleave(expansion_factor, dim=0)
        reference_mask_exp = reference_mask.repeat_interleave(expansion_factor, dim=0)
        target_exp = target_compressed.repeat_interleave(expansion_factor, dim=0)
        
        if lang_ids is not None:
            lang_ids_exp = lang_ids.repeat_interleave(expansion_factor, dim=0)
        else:
            lang_ids_exp = None
        
        # 3. Compress reference latents
        with torch.no_grad():
            ref_latent = latent_encoder(reference_mel_exp)
            ref_compressed = compress_latents(ref_latent, compression_factor=6)
        
        # 4. Encode reference FIRST (needed for text encoding)
        reference_encoding = text_to_latent.encode_reference(
            ref_compressed,
            reference_mask_exp
        )
        
        # 5. Encode text WITH reference conditioning
        text_encoding = text_to_latent.encode_text(
            text_ids_exp,
            reference_encoding,  # ref_vectors from reference encoder
            text_mask_exp,
            lang_id=lang_ids_exp
        )
        
        # 6. Flow-matching loss
        losses = loss_fn(
            model=text_to_latent.vector_field,
            z1=target_exp,
            text_encoding=text_encoding,
            reference_encoding=reference_encoding,
            text_mask=text_mask_exp
        )
        
        loss = losses["total"]
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return {
        "loss": losses["total"].item(),
        "flow_matching_loss": losses["flow_matching"].item(),
        "velocity_error": losses["mean_velocity_error"].item()
    }


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    iteration: int,
    base_lr: float,
    halve_interval: int
):
    """
    Learning rate halving кожні halve_interval iterations.
    """
    num_halvings = iteration // halve_interval
    lr = base_lr * (0.5 ** num_halvings)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return lr


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    text_to_latent: nn.Module,
    latent_encoder: nn.Module,
    loss_fn: FlowMatchingLoss,
    device: torch.device,
    max_samples: int = 50
) -> Dict[str, float]:
    """Validation loop."""
    text_to_latent.eval()
    latent_encoder.eval()
    
    total_loss = 0.0
    num_samples = 0
    
    for batch in dataloader:
        if num_samples >= max_samples:
            break
        
        mel = batch["mel"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        reference_mel = batch["reference_mel"].to(device)
        
        # Encode to latents
        target_latent = latent_encoder(mel)
        target_compressed = compress_latents(target_latent, compression_factor=6)
        
        ref_latent = latent_encoder(reference_mel)
        ref_compressed = compress_latents(ref_latent, compression_factor=6)
        
        # Encode conditioning - reference FIRST
        reference_encoding = text_to_latent.encode_reference(ref_compressed)
        text_encoding = text_to_latent.encode_text(text_ids, reference_encoding, text_mask)
        
        # Flow-matching loss
        losses = loss_fn(
            model=text_to_latent.vector_field,
            z1=target_compressed,
            text_encoding=text_encoding,
            reference_encoding=reference_encoding,
            text_mask=text_mask
        )
        
        total_loss += losses["total"].item() * mel.size(0)
        num_samples += mel.size(0)
    
    text_to_latent.train()
    
    return {
        "val_loss": total_loss / max(num_samples, 1)
    }


def main(args):
    """Main training function."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override with CLI args
    if args.batch_size:
        config.train_tts.batch_size = args.batch_size
    if args.lr:
        config.train_tts.learning_rate = args.lr
    
    # Logging
    if is_main and not args.no_wandb:
        wandb.init(
            project=config.logging.project,
            name=f"text_to_latent_{time.strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(config)
        )
    
    # Output directories
    checkpoint_dir = Path(config.output.checkpoint_dir) / "tts"
    sample_dir = Path(config.output.sample_dir) / "tts"
    
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio processor & tokenizer
    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels
    )
    
    tokenizer = CharacterTokenizer(
        languages=config.languages.supported
    )
    
    # Dataset
    train_dataset = TTSDataset(
        manifest_path=config.data.train_manifest,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        max_duration=config.data.max_audio_duration,
        min_duration=config.data.min_audio_duration,
        max_text_length=config.data.max_text_length
    )
    
    val_dataset = TTSDataset(
        manifest_path=config.data.val_manifest,
        audio_processor=audio_processor,
        tokenizer=tokenizer
    )
    
    # DataLoader (NO expansion here - we do it in train_step)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_tts.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=tts_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train_tts.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=tts_collate_fn
    )
    
    # Load pretrained latent encoder (frozen)
    latent_encoder = LatentEncoder(
        input_dim=config.autoencoder.encoder.input_dim,
        hidden_dim=config.autoencoder.encoder.hidden_dim,
        output_dim=config.autoencoder.encoder.output_dim,
        num_blocks=config.autoencoder.encoder.num_blocks
    ).to(device)
    
    if args.autoencoder_checkpoint:
        ae_ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
        latent_encoder.load_state_dict(ae_ckpt["encoder"])
        print(f"Loaded latent encoder from {args.autoencoder_checkpoint}")
    
    latent_encoder.eval()
    for param in latent_encoder.parameters():
        param.requires_grad = False
    
    # Text-to-Latent model
    text_to_latent = TextToLatent(
        latent_dim=144,  # Compressed: 24 * 6
        vocab_size=tokenizer.vocab_size,
        text_embed_dim=config.text_to_latent.text_encoder.embed_dim,
        text_hidden_dim=config.text_to_latent.text_encoder.hidden_dim,
        ref_hidden_dim=config.text_to_latent.reference_encoder.hidden_dim,
        vf_hidden_dim=config.text_to_latent.vector_field.hidden_dim,
        num_ref_vectors=config.text_to_latent.reference_encoder.num_output_vectors,
        gamma=config.larope.gamma,
        sigma_min=config.flow_matching.sigma_min,
        p_uncond=config.flow_matching.p_uncond,
        cfg_scale=config.flow_matching.cfg_scale
    ).to(device)
    
    # DDP wrapping
    if world_size > 1:
        text_to_latent = DDP(text_to_latent, device_ids=[local_rank])
    
    # Loss
    loss_fn = FlowMatchingLoss(
        sigma_min=config.flow_matching.sigma_min,
        p_uncond=config.flow_matching.p_uncond
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        text_to_latent.parameters(),
        lr=config.train_tts.learning_rate,
        betas=tuple(config.train_tts.optimizer.betas),
        weight_decay=config.train_tts.optimizer.weight_decay
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config.train_tts.amp.enabled)
    
    # Resume
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(
            Path(args.resume), text_to_latent, optimizer, scaler
        )
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    total_iterations = config.train_tts.total_iterations
    checkpoint_interval = config.train_tts.checkpoint_interval
    validation_interval = config.train_tts.validation_interval
    log_interval = config.logging.log_interval
    lr_halve_interval = config.train_tts.lr_halve_interval
    expansion_factor = config.train_tts.expansion_factor
    base_lr = config.train_tts.learning_rate
    
    iteration = start_iteration
    epoch = 0
    
    if is_main:
        pbar = tqdm(total=total_iterations, initial=start_iteration, desc="Training TTS")
    
    while iteration < total_iterations:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            # Adjust learning rate
            current_lr = adjust_learning_rate(
                optimizer, iteration, base_lr, lr_halve_interval
            )
            
            # Training step (з context-sharing всередині)
            losses = train_step(
                batch, text_to_latent, latent_encoder,
                optimizer, loss_fn, scaler, device,
                expansion_factor=expansion_factor,
                use_amp=config.train_tts.amp.enabled
            )
            
            iteration += 1
            
            # Logging
            if is_main and iteration % log_interval == 0:
                losses["learning_rate"] = current_lr
                
                if not args.no_wandb:
                    wandb.log(losses, step=iteration)
                
                pbar.set_postfix({
                    "loss": f"{losses['loss']:.4f}",
                    "lr": f"{current_lr:.2e}"
                })
                pbar.update(log_interval)
            
            # Validation
            if is_main and iteration % validation_interval == 0:
                val_metrics = validate(
                    val_loader, text_to_latent, latent_encoder,
                    loss_fn, device
                )
                
                if not args.no_wandb:
                    wandb.log(val_metrics, step=iteration)
                
                print(f"\nValidation at {iteration}: {val_metrics}")
            
            # Checkpoint
            if is_main and iteration % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_{iteration}.pt",
                    iteration, text_to_latent, optimizer, scaler,
                    OmegaConf.to_container(config)
                )
        
        epoch += 1
    
    # Final checkpoint
    if is_main:
        save_checkpoint(
            checkpoint_dir / "checkpoint_final.pt",
            iteration, text_to_latent, optimizer, scaler,
            OmegaConf.to_container(config)
        )
        
        pbar.close()
        
        if not args.no_wandb:
            wandb.finish()
    
    cleanup_distributed()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text-to-Latent Module")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--autoencoder-checkpoint", type=str, required=True,
                        help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    main(args)
