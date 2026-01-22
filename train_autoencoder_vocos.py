#!/usr/bin/env python3
"""
Train Encoder з pretrained Vocos Decoder

Замість тренування encoder+decoder з нуля, тренуємо тільки encoder
з frozen pretrained Vocos decoder.

Переваги:
- Набагато швидше (~50k замість 200k ітерацій)
- Гарантована якість decoder
- Економія GPU годин

Usage:
    python train_autoencoder_vocos.py --config config/vocos_adapter.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchaudio

from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from supertonic.models.vocos_wrapper import create_vocos_autoencoder
from supertonic.data.dataset import TTSDataset
from supertonic.data.collate import tts_collate_fn


def mel_reconstruction_loss(pred_mel, target_mel):
    """L1 loss між predicted і target mel-spectrograms."""
    return F.l1_loss(pred_mel, target_mel)


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    max_samples: int = 50
):
    """Validation step."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        if num_batches >= max_samples:
            break
        
        audio = batch["audio"].to(device)
        
        # Resample to 24kHz for Vocos
        audio = torchaudio.functional.resample(
            audio, 44100, 24000
        )
        
        # Forward
        latent, mel = model.encode(audio, return_mel=True)
        recon_audio = model.decode(latent)
        
        # Compute mel loss
        with torch.no_grad():
            recon_mel = model.vocos.feature_extractor(recon_audio)
        
        loss = mel_reconstruction_loss(latent, mel)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    model.train()
    
    return {"val_loss": avg_loss}


def train_step(
    batch: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config,
    device: torch.device,
    use_amp: bool = True
):
    """Single training step."""
    optimizer.zero_grad()
    
    audio = batch["audio"].to(device)
    
    # Resample to 24kHz
    audio = torchaudio.functional.resample(audio, 44100, 24000)
    
    with autocast(enabled=use_amp, dtype=torch.bfloat16):
        # Encode
        latent, mel_target = model.encode(audio, return_mel=True)
        
        # Decode
        recon_audio = model.decode(latent)
        
        # Extract mel from reconstruction
        with torch.no_grad():
            mel_recon = model.vocos.feature_extractor(recon_audio)
        
        # Loss: encoder має виводити features близькі до mel-spec
        loss = mel_reconstruction_loss(latent, mel_target)
        
        # Додатковий L1 loss на реконструкцію
        recon_loss = F.l1_loss(recon_audio, audio)
        
        total_loss = loss + recon_loss * 0.1
    
    # Backward
    scaler.scale(total_loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return {
        "loss": total_loss.item(),
        "mel_loss": loss.item(),
        "recon_loss": recon_loss.item()
    }


def save_checkpoint(
    path: Path,
    iteration: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config
):
    """Save training checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "encoder": model.encoder.state_dict(),  # Тільки encoder
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config
    }
    
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")


def main(args):
    # Load config
    config = OmegaConf.load(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = Path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\n" + "="*60)
    print("Creating Vocos Adapter...")
    print("="*60)
    
    model = create_vocos_autoencoder(
        encoder_hidden_dim=config.encoder.hidden_dim,
        encoder_blocks=config.encoder.num_blocks,
        vocos_model=config.vocos.pretrained_model,
        freeze_vocos=config.vocos.freeze_decoder
    ).to(device)
    
    print(f"\nEncoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Vocos parameters: {sum(p.numel() for p in model.vocos.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer (тільки для encoder)
    optimizer = torch.optim.AdamW(
        model.encoder.parameters(),
        lr=config.train.learning_rate,
        betas=config.train.optimizer.betas,
        weight_decay=config.train.optimizer.weight_decay
    )
    
    # AMP Scaler
    scaler = GradScaler(enabled=config.train.amp.enabled)
    
    # Resume from checkpoint
    start_iteration = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.encoder.load_state_dict(checkpoint["encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_iteration = checkpoint["iteration"]
    
    # Dataset
    print("\n" + "="*60)
    print("Loading Dataset...")
    print("="*60)
    
    train_dataset = TTSDataset(
        audio_dir=config.data.train_dir,
        cache_dir=config.data.cache_dir,
        min_duration=config.data.min_audio_duration,
        max_duration=config.data.max_audio_duration,
        sample_rate=44100  # Буде resample в 24kHz під час тренування
    )
    
    val_dataset = TTSDataset(
        audio_dir=config.data.val_dir,
        cache_dir=config.data.cache_dir,
        min_duration=config.data.min_audio_duration,
        max_duration=config.data.max_audio_duration,
        sample_rate=44100
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=tts_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=tts_collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    model.train()
    
    iteration = start_iteration
    epoch = 0
    
    total_iterations = config.train.total_iterations
    log_interval = 10
    validation_interval = config.train.validation_interval
    checkpoint_interval = config.train.checkpoint_interval
    
    pbar = tqdm(total=total_iterations, initial=start_iteration, desc="Training Vocos Adapter")
    
    while iteration < total_iterations:
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            iteration += 1
            
            # Train step
            losses = train_step(
                batch, model, optimizer, scaler,
                config, device, use_amp=config.train.amp.enabled
            )
            
            # Logging
            if iteration % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{losses['loss']:.4f}",
                    "mel": f"{losses['mel_loss']:.4f}",
                    "recon": f"{losses['recon_loss']:.4f}"
                })
                pbar.update(log_interval)
            
            # Validation
            if iteration % validation_interval == 0:
                val_metrics = validate(val_loader, model, device)
                print(f"\nValidation at {iteration}: {val_metrics}")
            
            # Checkpoint
            if iteration % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_{iteration}.pt",
                    iteration, model, optimizer, scaler,
                    OmegaConf.to_container(config)
                )
        
        epoch += 1
    
    # Final checkpoint
    save_checkpoint(
        checkpoint_dir / "checkpoint_final.pt",
        iteration, model, optimizer, scaler,
        OmegaConf.to_container(config)
    )
    
    pbar.close()
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Encoder with Vocos")
    parser.add_argument("--config", type=str, default="config/vocos_adapter.yaml")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    main(args)
