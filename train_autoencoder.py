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
import gc
import psutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast  # Updated API (torch 2.0+)
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from omegaconf import OmegaConf
import wandb

# RAM cleanup threshold (in GB)
RAM_CLEANUP_THRESHOLD_GB = 300

def check_and_cleanup_ram(force: bool = False) -> bool:
    """Check RAM usage and cleanup if above threshold.
    
    Returns True if cleanup was performed.
    """
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    
    if force or used_gb > RAM_CLEANUP_THRESHOLD_GB:
        gc.collect()
        return True
    return False

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
from supertonic.utils.training_logger import TrainingLogger


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
    use_amp: bool = True,
    iteration: int = 0,
    disc_start_steps: int = 0  # NEW: Discriminator warmup
) -> Dict[str, float]:
    """
    Один training step.
    
    Args:
        iteration: Current training iteration
        disc_start_steps: Steps before discriminator starts (warmup)
    
    Returns:
        Dict з loss values
    """
    audio = batch["audio"].to(device)
    mel = batch["mel"].to(device)
    
    # Determine if discriminator should be active
    disc_active = iteration >= disc_start_steps
    
    # ==================== Discriminator Step ====================
    d_loss_val = 0.0
    mpd_real_features = []
    mrd_real_features = []
    
    if disc_active:
        optimizer_d.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            # Encode → Decode (no grad for D step)
            with torch.no_grad():
                latent = encoder(mel)
                generated_audio = decoder(latent)
            
            # Match lengths
            min_len = min(audio.size(-1), generated_audio.size(-1))
            audio_trim = audio[..., :min_len]
            generated_trim = generated_audio[..., :min_len].detach()
            
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
        d_loss_val = d_losses["total"].item()
        
        # Clear D-step intermediates
        del mpd_fake_outputs, mrd_fake_outputs, fake_outputs, d_loss
    
    # ==================== Generator Step ====================
    optimizer_g.zero_grad(set_to_none=True)
    
    with autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
        # Encode → Decode (with gradients this time)
        latent = encoder(mel)
        generated_audio = decoder(latent)
        
        # Match lengths
        min_len = min(audio.size(-1), generated_audio.size(-1))
        audio_trim = audio[..., :min_len]
        generated_trim = generated_audio[..., :min_len]
        
        # Discriminator forward only if disc is active
        fake_outputs = []
        fake_features = []
        real_features_detached = []
        
        if disc_active:
            mpd_fake_outputs, mpd_fake_features = mpd(generated_trim)
            mrd_fake_outputs, mrd_fake_features = mrd(generated_trim)
            
            fake_outputs = mpd_fake_outputs + mrd_fake_outputs
            fake_features = mpd_fake_features + mrd_fake_features
            
            real_features_detached = [
                [f.detach() for f in feat_list] 
                for feat_list in (mpd_real_features + mrd_real_features)
            ]
        
        # Generator loss (use_adv=False during warmup)
        g_losses = loss_fn.generator_loss(
            real_audio=audio_trim,
            generated_audio=generated_trim,
            disc_fake_outputs=fake_outputs,
            real_features=real_features_detached,
            fake_features=fake_features,
            use_adv=disc_active  # NEW: disable adversarial during warmup
        )
        g_loss = g_losses["total"]
    
    scaler.scale(g_loss).backward()
    scaler.step(optimizer_g)
    scaler.update()
    
    # Clean up ALL intermediates to prevent fragmentation
    del audio, mel, latent, generated_audio, audio_trim, generated_trim
    if disc_active:
        del mpd_real_features, mrd_real_features, real_features_detached
        del fake_outputs, fake_features
        if 'mpd_fake_outputs' in dir():
            del mpd_fake_outputs, mrd_fake_outputs, mpd_fake_features, mrd_fake_features
    
    return {
        "d_loss": d_loss_val,  # Fixed: use d_loss_val (0 during warmup)
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
    
    # Get data directory (CLI override or config)
    data_dir = args.data_dir if args.data_dir else config.paths.get("data_dir", "data")
    if is_main:
        print(f"Data directory: {data_dir}")
    
    # Override output directory if provided
    if args.output_dir:
        config.output.checkpoint_dir = f"{args.output_dir}/checkpoints"
        config.output.sample_dir = f"{args.output_dir}/samples"
        if is_main:
            print(f"Output directory: {args.output_dir}")
    
    # Dataset with segment cropping and RAM caching
    cache_audio = config.train_autoencoder.get("cache_audio", False)
    if is_main:
        print(f"Audio caching in RAM: {cache_audio}")
    
    train_dataset = AutoencoderDataset(
        manifest_path=config.data.train_manifest,
        audio_processor=audio_processor,
        max_duration=config.data.max_audio_duration,
        min_duration=config.data.min_audio_duration,
        return_mel=True,
        segment_length=segment_length,  # Random crop for memory efficiency
        data_dir=data_dir,
        cache_audio=cache_audio  # Cache all audio in RAM!
    )
    
    val_dataset = AutoencoderDataset(
        manifest_path=config.data.val_manifest,
        audio_processor=audio_processor,
        max_duration=config.data.max_audio_duration,
        min_duration=config.data.min_audio_duration,
        return_mel=True,
        segment_length=segment_length,
        data_dir=data_dir,
        cache_audio=cache_audio  # Cache all audio in RAM!
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
    
    # Get num_workers from config (try train_autoencoder first, then training.autoencoder)
    num_workers = config.train_autoencoder.get("num_workers", 
                    config.training.autoencoder.get("num_workers", 4))
    prefetch = config.train_autoencoder.get("prefetch_factor", 4)
    if is_main:
        print(f"DataLoader: {num_workers} workers, prefetch={prefetch}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_autoencoder.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=autoencoder_collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,  # Keep workers alive (use few workers to limit leak)
        prefetch_factor=prefetch if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train_autoencoder.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=autoencoder_collate_fn,
        persistent_workers=num_workers > 0,  # Keep workers alive
        prefetch_factor=prefetch if num_workers > 0 else None
    )
    
    # Models
    gradient_checkpointing = config.optimization.get("gradient_checkpointing", False)
    if is_main:
        print(f"Gradient checkpointing: {gradient_checkpointing}")
    
    # Get autoencoder config (support both config.autoencoder and config.model.autoencoder)
    ae_config = config.get("autoencoder", config.get("model", {}).get("autoencoder", {}))
    
    encoder = LatentEncoder(
        input_dim=ae_config.encoder.input_dim,
        hidden_dim=ae_config.encoder.hidden_dim,
        output_dim=ae_config.encoder.output_dim,
        num_blocks=ae_config.encoder.num_blocks,
        kernel_size=ae_config.encoder.kernel_size,
        gradient_checkpointing=gradient_checkpointing
    ).to(device)
    
    decoder = LatentDecoder(
        input_dim=ae_config.decoder.input_dim,
        hidden_dim=ae_config.decoder.hidden_dim,
        num_blocks=ae_config.decoder.num_blocks,
        kernel_size=ae_config.decoder.kernel_size,
        dilations=ae_config.decoder.dilations,
        n_fft=config.audio.n_fft,              # CRITICAL: use config values!
        hop_length=config.audio.hop_length,    # CRITICAL: use config values!
        causal=ae_config.decoder.causal,
        gradient_checkpointing=gradient_checkpointing
    ).to(device)
    
    if is_main:
        print(f"=" * 60)
        print(f"MODEL CONFIGURATION CHECK:")
        print(f"  Encoder input_dim (n_mels): {ae_config.encoder.input_dim}")
        print(f"  Decoder n_fft: {config.audio.n_fft}")
        print(f"  Decoder hop_length: {config.audio.hop_length}")
        print(f"  Audio sample_rate: {config.audio.sample_rate}")
        print(f"  MRD fft_sizes: {list(ae_config.discriminator.mrd_fft_sizes)}")
        print(f"=" * 60)
    
    mpd = MultiPeriodDiscriminator(
        periods=ae_config.discriminator.mpd_periods
    ).to(device)
    
    mrd = MultiResolutionDiscriminator(
        fft_sizes=ae_config.discriminator.mrd_fft_sizes
    ).to(device)
    
    # DDP wrapping
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])
        mpd = DDP(mpd, device_ids=[local_rank])
        mrd = DDP(mrd, device_ids=[local_rank])
    
    # Loss - simplified for WaveNeXt head (no SC/LogMag/Waveform needed)
    loss_weights = config.train_autoencoder.loss_weights
    loss_fn = AutoencoderLoss(
        lambda_mel=loss_weights.reconstruction,
        lambda_adv=loss_weights.adversarial,
        lambda_fm=loss_weights.feature_matching,
        fft_sizes=list(ae_config.discriminator.mrd_fft_sizes),
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels
    )
    
    if is_main:
        print(f"Loss config: λ_mel={loss_weights.reconstruction}, λ_adv={loss_weights.adversarial}, λ_fm={loss_weights.feature_matching}")
        print(f"Audio config: sample_rate={config.audio.sample_rate}, fft_sizes={list(ae_config.discriminator.mrd_fft_sizes)}, n_mels={config.audio.n_mels}")
    
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
    
    # Training loop
    total_iterations = config.train_autoencoder.total_iterations
    checkpoint_interval = config.train_autoencoder.checkpoint_interval
    validation_interval = config.train_autoencoder.validation_interval
    log_interval = config.train_autoencoder.get("log_interval", config.logging.log_interval)
    
    # Get discriminator warmup steps from config
    disc_start_steps = config.train_autoencoder.get("discriminator_start_steps", 2000)
    
    # Initialize beautiful logger
    logger = TrainingLogger(
        total_iterations=total_iterations,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        validation_interval=validation_interval,
        disc_start_steps=disc_start_steps,
        rank=rank,
        world_size=world_size
    )
    
    # Print config
    logger.print_config({
        'batch_size': config.train_autoencoder.batch_size,
        'learning_rate': config.train_autoencoder.learning_rate,
        'loss_weights': OmegaConf.to_container(config.train_autoencoder.loss_weights)
    })
    
    # Log resume
    if args.resume:
        logger.log_resume(start_iteration, args.resume)
    
    # Log GPU status at start
    logger.log_gpu_status()
    
    iteration = start_iteration
    epoch = 0
    
    while iteration < total_iterations:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            # Training step (with warmup support)
            losses = train_step(
                batch, encoder, decoder, mpd, mrd,
                optimizer_g, optimizer_d, loss_fn, scaler,
                device, 
                use_amp=config.train_autoencoder.amp.enabled,
                iteration=iteration,
                disc_start_steps=disc_start_steps
            )
            
            iteration += 1
            
            # Logging with beautiful logger
            logger.log_step(iteration, losses)
            
            # Wandb logging
            if is_main and iteration % log_interval == 0 and not args.no_wandb:
                wandb.log(losses, step=iteration)
            
            # Validation
            if is_main and iteration % validation_interval == 0:
                val_metrics = validate(val_loader, encoder, decoder, loss_fn, device)
                
                logger.log_validation(iteration, val_metrics)
                
                if not args.no_wandb:
                    wandb.log(val_metrics, step=iteration)
            
            # Checkpoint
            if is_main and iteration % checkpoint_interval == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
                save_checkpoint(
                    ckpt_path,
                    iteration, encoder, decoder, mpd, mrd,
                    optimizer_g, optimizer_d, scaler,
                    OmegaConf.to_container(config)
                )
                logger.log_checkpoint(iteration, str(ckpt_path))
            
            # RAM cleanup - check every 100 steps, cleanup if > 300 GB
            if iteration % 100 == 0:
                if check_and_cleanup_ram():
                    torch.cuda.empty_cache()
                    if is_main and iteration % 500 == 0:
                        mem = psutil.virtual_memory()
                        print(f"🧹 RAM cleanup: {mem.used / (1024**3):.1f} GB used")
        
        epoch += 1
    
    # Final checkpoint
    if is_main:
        ckpt_path = checkpoint_dir / "checkpoint_final.pt"
        save_checkpoint(
            ckpt_path,
            iteration, encoder, decoder, mpd, mrd,
            optimizer_g, optimizer_d, scaler,
            OmegaConf.to_container(config)
        )
        logger.log_checkpoint(iteration, str(ckpt_path))
        logger.log_training_complete(iteration)
        
        if not args.no_wandb:
            wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Autoencoder")
    parser.add_argument("--config", type=str, default="config/22khz_optimal.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--data_dir", "--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output_dir", "--output-dir", type=str, default=None,
                        help="Override output directory")
    
    args = parser.parse_args()
    main(args)
