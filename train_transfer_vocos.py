#!/usr/bin/env python3
"""
Transfer Learning: Адаптація encoder (80k) під Vocos

Використовує натренований encoder + projection layer.
Тренуємо тільки adapter (~15k ітерацій замість 50k).

Usage:
    python train_transfer_vocos.py --config config/transfer_vocos.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from supertonic.models.transfer_vocos import create_transfer_learning_adapter
from supertonic.data.dataset import AutoencoderDataset
from supertonic.data.preprocessing import AudioProcessor


def collate_fn(batch):
    """Collate function для variable-length audio with resampling."""
    # Resample all to 24kHz first
    audios_24k = []
    for item in batch:
        audio = item["audio"]
        # Resample 44.1kHz -> 24kHz
        audio = torchaudio.functional.resample(audio, 44100, 24000)
        audios_24k.append(audio)
    
    # Find max length
    max_len = max(a.shape[0] for a in audios_24k)
    
    # Pad all to max_len
    padded = []
    for audio in audios_24k:
        if len(audio) < max_len:
            audio = F.pad(audio, (0, max_len - len(audio)))
        padded.append(audio)
    
    return {"audio": torch.stack(padded, dim=0)}


def train_step(
    batch: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True
):
    """Training step."""
    optimizer.zero_grad()
    
    audio = batch["audio"].to(device)
    
    # Resample to 24kHz
    audio = torchaudio.functional.resample(audio, 44100, 24000)
    
    # Auto-detect autocast API
    try:
        ctx = autocast("cuda", enabled=use_amp, dtype=torch.bfloat16)
    except TypeError:
        ctx = autocast(enabled=use_amp, dtype=torch.bfloat16)
    
    with ctx:
        # Encode to features (this is what we're training)
        features = model.encode(audio)  # [B, 100, T]
        
        # Get target features from Vocos directly
        with torch.no_grad():
            target_features = model.vocos.feature_extractor(audio)  # [B, 100, T]
        
        # Match lengths
        min_len = min(features.shape[-1], target_features.shape[-1])
        features = features[..., :min_len]
        target_features = target_features[..., :min_len]
        
        # Feature matching loss (L1)
        feature_loss = F.l1_loss(features, target_features)
        
        # Mel spectral loss (additional)
        mel_loss = F.mse_loss(features, target_features)
        
        total_loss = feature_loss + mel_loss * 0.5
    
    scaler.scale(total_loss).backward()
    
    # Gradient clipping (тільки для trainable params)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 5.0
    )
    
    scaler.step(optimizer)
    scaler.update()
    
    return {
        "loss": total_loss.item(),
        "feature_loss": feature_loss.item(),
        "mel_loss": mel_loss.item()
    }


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    max_samples: int = 50
):
    """Validation."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        if num_batches >= max_samples:
            break
        
        audio = batch["audio"].to(device)
        audio = torchaudio.functional.resample(audio, 44100, 24000)
        
        recon_audio = model(audio)
        loss = F.l1_loss(recon_audio, audio)
        
        total_loss += loss.item()
        num_batches += 1
    
    model.train()
    
    return {"val_loss": total_loss / max(num_batches, 1)}


def save_checkpoint(
    path: Path,
    iteration: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config
):
    """Save checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "adapter": model.adapter.state_dict(),  # Тільки adapter
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config
    }
    
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")


def main(args):
    config = OmegaConf.load(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output dirs
    checkpoint_dir = Path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model з transfer learning
    print("\n" + "="*60)
    print("Creating Transfer Learning Model...")
    print("="*60)
    
    model = create_transfer_learning_adapter(
        pretrained_checkpoint=config.transfer.pretrained_checkpoint,
        vocos_model=config.vocos.pretrained_model,
        freeze_encoder=config.transfer.freeze_encoder,
        freeze_vocos=config.transfer.freeze_vocos,
        device=device
    )
    
    # Optimizer (тільки для trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        betas=config.train.optimizer.betas,
        weight_decay=config.train.optimizer.weight_decay
    )
    
    try:
        scaler = GradScaler("cuda", enabled=config.train.amp.enabled)
    except (TypeError, AttributeError):
        # Fallback for older PyTorch
        from torch.cuda.amp import GradScaler as OldGradScaler
        scaler = OldGradScaler(enabled=config.train.amp.enabled)
    
    # Resume
    start_iteration = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.adapter.load_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_iteration = checkpoint["iteration"]
    
    # Dataset
    print("\n" + "="*60)
    print("Loading Dataset...")
    print("="*60)
    
    audio_processor = AudioProcessor(
        sample_rate=44100,
        n_mels=228,
        hop_length=512,
        win_length=2048,
        n_fft=2048
    )
    
    train_dataset = AutoencoderDataset(
        manifest_path=config.data.train_manifest,
        audio_processor=audio_processor,
        min_duration=config.data.min_audio_duration,
        max_duration=config.data.max_audio_duration,
        segment_length=88200  # 2 sec at 44.1kHz (faster!)
    )
    
    val_dataset = AutoencoderDataset(
        manifest_path=config.data.val_manifest,
        audio_processor=audio_processor,
        min_duration=config.data.min_audio_duration,
        max_duration=config.data.max_audio_duration,
        segment_length=88200
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training
    print("\n" + "="*60)
    print("Starting Transfer Learning...")
    print("="*60 + "\n")
    
    model.train()
    
    iteration = start_iteration
    epoch = 0
    
    total_iterations = config.train.total_iterations
    log_interval = 10
    validation_interval = config.train.validation_interval
    checkpoint_interval = config.train.checkpoint_interval
    
    pbar = tqdm(total=total_iterations, initial=start_iteration, 
                desc="Transfer Learning")
    
    while iteration < total_iterations:
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            iteration += 1
            
            losses = train_step(
                batch, model, optimizer, scaler,
                device, use_amp=config.train.amp.enabled
            )
            
            if iteration % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{losses['loss']:.4f}",
                    "feat": f"{losses['feature_loss']:.4f}",
                    "mel": f"{losses['mel_loss']:.4f}"
                })
                pbar.update(log_interval)
            
            if iteration % validation_interval == 0:
                val_metrics = validate(val_loader, model, device)
                print(f"\nValidation at {iteration}: {val_metrics}")
            
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
    print("Transfer Learning completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning to Vocos")
    parser.add_argument("--config", type=str, default="config/transfer_vocos.yaml")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    main(args)
