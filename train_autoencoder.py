"""
Training Script для Speech Autoencoder

Тренування Vocos-based autoencoder для кодування аудіо в латентний простір.

Конфігурація:
- Optimizer: AdamW, lr=2e-4
- Total iterations: 1.5M
- Batch size: 16
- Loss: λ_recon=45, λ_adv=1, λ_fm=0.1
- AMP: bfloat16

Hardware:
- Рекомендовано: 4× RTX 4090/5090 (24GB each)
- Мінімум: 1× RTX 3090 з gradient accumulation

Референс: Vocos, HiFi-GAN, Supertonic v2 paper
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.speech_autoencoder import (
    SpeechAutoencoder,
    LatentEncoder,
    LatentDecoder,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator
)
from supertonic.losses.autoencoder_loss import AutoencoderLoss
from supertonic.data.dataset import AutoencoderDataset
from supertonic.data.collate import autoencoder_collate_fn
from supertonic.data.preprocessing import AudioProcessor


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
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    path: Path,
    iteration: int,
    encoder: nn.Module,
    decoder: nn.Module,
    mpd: nn.Module,
    mrd: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Dict
):
    """Зберігає checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "mpd": mpd.state_dict(),
        "mrd": mrd.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at iteration {iteration}")


def load_checkpoint(
    path: Path,
    encoder: nn.Module,
    decoder: nn.Module,
    mpd: nn.Module,
    mrd: nn.Module,
    optimizer_g: Optional[torch.optim.Optimizer] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None
) -> int:
    """Завантажує checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    mpd.load_state_dict(checkpoint["mpd"])
    mrd.load_state_dict(checkpoint["mrd"])
    
    if optimizer_g is not None:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if optimizer_d is not None:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    
    return checkpoint["iteration"]


def train_step(
    batch: Dict[str, torch.Tensor],
    encoder: nn.Module,
    decoder: nn.Module,
    mpd: nn.Module,
    mrd: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss_fn: AutoencoderLoss,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Один training step.
    
    Returns:
        Dict з loss values
    """
    audio = batch["audio"].to(device)
    mel = batch["mel"].to(device)
    
    # ==================== Discriminator Step ====================
    optimizer_d.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    
    with autocast(enabled=use_amp, dtype=torch.bfloat16):
        # Encode → Decode (no grad for D step)
        with torch.no_grad():
            latent = encoder(mel)
            generated_audio = decoder(latent)
        
        # Match lengths
        min_len = min(audio.size(-1), generated_audio.size(-1))
        audio_trim = audio[..., :min_len]
        generated_trim = generated_audio[..., :min_len].detach()  # Ensure detached
        
        # Discriminator forward
        mpd_real_outputs, mpd_real_features = mpd(audio_trim)
        mpd_fake_outputs, _ = mpd(generated_trim)
        
        mrd_real_outputs, mrd_real_features = mrd(audio_trim)
        mrd_fake_outputs, _ = mrd(generated_trim)
        
        # Combine outputs
        real_outputs = mpd_real_outputs + mrd_real_outputs
        fake_outputs = mpd_fake_outputs + mrd_fake_outputs
        
        # Discriminator loss
        d_losses = loss_fn.discriminator_loss(real_outputs, fake_outputs)
        d_loss = d_losses["total"]
    
    scaler.scale(d_loss).backward()
    scaler.step(optimizer_d)
    
    # Clear D-step intermediates to save memory
    del mpd_fake_outputs, mrd_fake_outputs, fake_outputs, d_loss
    
    # ==================== Generator Step ====================
    optimizer_g.zero_grad(set_to_none=True)
    
    with autocast(enabled=use_amp, dtype=torch.bfloat16):
        # Encode → Decode (with gradients this time)
        latent = encoder(mel)
        generated_audio = decoder(latent)
        
        # Match lengths
        min_len = min(audio.size(-1), generated_audio.size(-1))
        audio_trim = audio[..., :min_len]
        generated_trim = generated_audio[..., :min_len]
        
        # Discriminator forward (for generator loss)
        mpd_fake_outputs, mpd_fake_features = mpd(generated_trim)
        mrd_fake_outputs, mrd_fake_features = mrd(generated_trim)
        
        # Combine outputs (list of tensors)
        fake_outputs = mpd_fake_outputs + mrd_fake_outputs
        
        # Features are List[List[Tensor]] - keep structure!
        # Detach real features from D step
        real_features_detached = [
            [f.detach() for f in feat_list] 
            for feat_list in (mpd_real_features + mrd_real_features)
        ]
        fake_features = mpd_fake_features + mrd_fake_features
        
        # Generator loss
        g_losses = loss_fn.generator_loss(
            real_audio=audio_trim,
            generated_audio=generated_trim,
            disc_fake_outputs=fake_outputs,
            real_features=real_features_detached,
            fake_features=fake_features
        )
        g_loss = g_losses["total"]
    
    scaler.scale(g_loss).backward()
    scaler.step(optimizer_g)
    scaler.update()
    
    # Clean up
    del audio, mel, latent, generated_audio, audio_trim, generated_trim
    torch.cuda.empty_cache()
    
    return {
        "d_loss": d_losses["total"].item(),
        "g_loss": g_losses["total"].item(),
        "recon_loss": g_losses["reconstruction"].item(),
        "adv_loss": g_losses["adversarial"].item(),
        "fm_loss": g_losses["feature_matching"].item()
    }


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    loss_fn: AutoencoderLoss,
    device: torch.device,
    max_samples: int = 100
) -> Dict[str, float]:
    """Validation loop."""
    encoder.eval()
    decoder.eval()
    
    total_recon_loss = 0.0
    num_samples = 0
    
    for batch in dataloader:
        if num_samples >= max_samples:
            break
        
        audio = batch["audio"].to(device)
        mel = batch["mel"].to(device)
        
        latent = encoder(mel)
        generated = decoder(latent)
        
        min_len = min(audio.size(-1), generated.size(-1))
        recon_loss = F.l1_loss(generated[..., :min_len], audio[..., :min_len])
        
        total_recon_loss += recon_loss.item() * audio.size(0)
        num_samples += audio.size(0)
    
    encoder.train()
    decoder.train()
    
    return {
        "val_recon_loss": total_recon_loss / max(num_samples, 1)
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
        config.train_autoencoder.batch_size = args.batch_size
    if args.lr:
        config.train_autoencoder.learning_rate = args.lr
    
    # Logging
    if is_main and not args.no_wandb:
        wandb.init(
            project=config.logging.project,
            name=f"autoencoder_{time.strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(config)
        )
    
    # Create output directories
    checkpoint_dir = Path(config.output.checkpoint_dir) / "autoencoder"
    sample_dir = Path(config.output.sample_dir) / "autoencoder"
    
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio processor
    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels
    )
    
    # Get segment_length from config (CRITICAL for OOM prevention!)
    segment_length = config.training.get("segment_length", 176400)  # default 4 sec at 44.1kHz
    if is_main:
        print(f"Using segment_length: {segment_length} samples ({segment_length/config.audio.sample_rate:.1f} sec)")
    
    # Dataset with segment cropping
    train_dataset = AutoencoderDataset(
        manifest_path=config.data.train_manifest,
        audio_processor=audio_processor,
        max_duration=config.data.max_audio_duration,
        min_duration=config.data.min_audio_duration,
        return_mel=True,
        segment_length=segment_length  # Random crop for memory efficiency
    )
    
    val_dataset = AutoencoderDataset(
        manifest_path=config.data.val_manifest,
        audio_processor=audio_processor,
        max_duration=config.data.max_audio_duration,
        min_duration=config.data.min_audio_duration,
        return_mel=True,
        segment_length=segment_length
    )
    
    # DataLoader
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Get num_workers from config (default 4 for compatibility)
    num_workers = config.training.autoencoder.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_autoencoder.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=autoencoder_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train_autoencoder.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=autoencoder_collate_fn
    )
    
    # Models
    gradient_checkpointing = config.optimization.get("gradient_checkpointing", False)
    if is_main:
        print(f"Gradient checkpointing: {gradient_checkpointing}")
    
    encoder = LatentEncoder(
        input_dim=config.autoencoder.encoder.input_dim,
        hidden_dim=config.autoencoder.encoder.hidden_dim,
        output_dim=config.autoencoder.encoder.output_dim,
        num_blocks=config.autoencoder.encoder.num_blocks,
        kernel_size=config.autoencoder.encoder.kernel_size,
        gradient_checkpointing=gradient_checkpointing
    ).to(device)
    
    decoder = LatentDecoder(
        input_dim=config.autoencoder.decoder.input_dim,
        hidden_dim=config.autoencoder.decoder.hidden_dim,
        num_blocks=config.autoencoder.decoder.num_blocks,
        kernel_size=config.autoencoder.decoder.kernel_size,
        dilations=config.autoencoder.decoder.dilations,
        causal=config.autoencoder.decoder.causal,
        gradient_checkpointing=gradient_checkpointing
    ).to(device)
    
    mpd = MultiPeriodDiscriminator(
        periods=config.autoencoder.discriminator.mpd_periods
    ).to(device)
    
    mrd = MultiResolutionDiscriminator(
        fft_sizes=config.autoencoder.discriminator.mrd_fft_sizes
    ).to(device)
    
    # DDP wrapping
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])
        mpd = DDP(mpd, device_ids=[local_rank])
        mrd = DDP(mrd, device_ids=[local_rank])
    
    # Loss
    loss_fn = AutoencoderLoss(
        lambda_recon=config.train_autoencoder.loss_weights.reconstruction,
        lambda_wave=config.train_autoencoder.loss_weights.get("waveform", 10.0),  # NEW
        lambda_adv=config.train_autoencoder.loss_weights.adversarial,
        lambda_fm=config.train_autoencoder.loss_weights.feature_matching
    )
    
    # Optimizers
    optimizer_g = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.train_autoencoder.learning_rate,
        betas=tuple(config.train_autoencoder.optimizer.betas),
        weight_decay=config.train_autoencoder.optimizer.weight_decay
    )
    
    optimizer_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=config.train_autoencoder.learning_rate,
        betas=tuple(config.train_autoencoder.optimizer.betas),
        weight_decay=config.train_autoencoder.optimizer.weight_decay
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config.train_autoencoder.amp.enabled)
    
    # Resume from checkpoint
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(
            Path(args.resume),
            encoder, decoder, mpd, mrd,
            optimizer_g, optimizer_d, scaler
        )
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    total_iterations = config.train_autoencoder.total_iterations
    checkpoint_interval = config.train_autoencoder.checkpoint_interval
    validation_interval = config.train_autoencoder.validation_interval
    log_interval = config.logging.log_interval
    
    iteration = start_iteration
    epoch = 0
    
    if is_main:
        pbar = tqdm(total=total_iterations, initial=start_iteration, desc="Training")
    
    while iteration < total_iterations:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            # Training step
            losses = train_step(
                batch, encoder, decoder, mpd, mrd,
                optimizer_g, optimizer_d, loss_fn, scaler,
                device, use_amp=config.train_autoencoder.amp.enabled
            )
            
            iteration += 1
            
            # Logging
            if is_main and iteration % log_interval == 0:
                if not args.no_wandb:
                    wandb.log(losses, step=iteration)
                
                pbar.set_postfix({
                    "g_loss": f"{losses['g_loss']:.4f}",
                    "d_loss": f"{losses['d_loss']:.4f}",
                    "recon": f"{losses['recon_loss']:.4f}"
                })
                pbar.update(log_interval)
            
            # Validation
            if is_main and iteration % validation_interval == 0:
                val_metrics = validate(val_loader, encoder, decoder, loss_fn, device)
                
                if not args.no_wandb:
                    wandb.log(val_metrics, step=iteration)
                
                print(f"\nValidation at {iteration}: {val_metrics}")
            
            # Checkpoint
            if is_main and iteration % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_{iteration}.pt",
                    iteration, encoder, decoder, mpd, mrd,
                    optimizer_g, optimizer_d, scaler,
                    OmegaConf.to_container(config)
                )
        
        epoch += 1
    
    # Final checkpoint
    if is_main:
        save_checkpoint(
            checkpoint_dir / "checkpoint_final.pt",
            iteration, encoder, decoder, mpd, mrd,
            optimizer_g, optimizer_d, scaler,
            OmegaConf.to_container(config)
        )
        
        pbar.close()
        
        if not args.no_wandb:
            wandb.finish()
    
    cleanup_distributed()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Autoencoder")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    
    args = parser.parse_args()
    main(args)
