#!/usr/bin/env python3
"""
Train Text-to-Latent with Vocos decoder.

This trains ONLY the Text-to-Latent module.
Vocos is used as a frozen decoder - no autoencoder training needed!

Architecture:
- Text → Text-to-Latent → 100-dim mel → Vocos → Audio

Usage:
    python train_vocos_tts.py --config config/vocos_tts.yaml
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

from supertonic.models.vocos_autoencoder import VocosAutoencoderWrapper
from supertonic.models.text_to_latent import TextToLatent
from supertonic.losses.flow_matching_loss import FlowMatchingLoss
from supertonic.data.dataset import TTSDataset
from supertonic.data.collate import tts_collate_fn
from supertonic.data.tokenizer import CharacterTokenizer


def train_step(
    batch: dict,
    model: TextToLatent,
    autoencoder: VocosAutoencoderWrapper,
    loss_fn: FlowMatchingLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True
):
    """Single training step."""
    optimizer.zero_grad()
    
    # Move to device
    audio = batch["audio"].to(device, non_blocking=True)
    text = batch["text"].to(device, non_blocking=True)
    text_lengths = batch["text_lengths"].to(device, non_blocking=True)
    reference_audio = batch["reference_audio"].to(device, non_blocking=True)
    
    # Resample to 24kHz
    audio = torchaudio.functional.resample(audio, 44100, 24000)
    reference_audio = torchaudio.functional.resample(reference_audio, 44100, 24000)
    
    try:
        ctx = autocast("cuda", enabled=use_amp, dtype=torch.bfloat16)
    except TypeError:
        ctx = autocast(enabled=use_amp, dtype=torch.bfloat16)
    
    with ctx:
        # Encode audio to mel using Vocos (frozen)
        with torch.no_grad():
            target_mel = autoencoder.encode(audio)  # [B, 100, T]
            reference_mel = autoencoder.encode(reference_audio)  # [B, 100, T_ref]
        
        # Text-to-Latent forward
        # Model predicts velocity field for flow matching
        loss_dict = loss_fn(
            model=model,
            z_target=target_mel,
            text_tokens=text,
            text_lengths=text_lengths,
            z_ref=reference_mel
        )
        
        total_loss = loss_dict["loss"]
    
    # Backward
    scaler.scale(total_loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    model: TextToLatent,
    autoencoder: VocosAutoencoderWrapper,
    loss_fn: FlowMatchingLoss,
    device: torch.device,
    max_batches: int = 50
):
    """Validation loop."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        if num_batches >= max_batches:
            break
        
        audio = batch["audio"].to(device)
        text = batch["text"].to(device)
        text_lengths = batch["text_lengths"].to(device)
        reference_audio = batch["reference_audio"].to(device)
        
        audio = torchaudio.functional.resample(audio, 44100, 24000)
        reference_audio = torchaudio.functional.resample(reference_audio, 44100, 24000)
        
        target_mel = autoencoder.encode(audio)
        reference_mel = autoencoder.encode(reference_audio)
        
        loss_dict = loss_fn(
            model=model,
            z_target=target_mel,
            text_tokens=text,
            text_lengths=text_lengths,
            z_ref=reference_mel
        )
        
        total_loss += loss_dict["loss"].item()
        num_batches += 1
    
    model.train()
    
    return {"val_loss": total_loss / max(num_batches, 1)}


def save_checkpoint(path, iteration, model, optimizer, scaler, config):
    """Save training checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": OmegaConf.to_container(config)
    }
    torch.save(checkpoint, path)
    print(f"✓ Saved: {path}")


def main(args):
    config = OmegaConf.load(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Output dirs
    ckpt_dir = Path(config.output.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Vocos Autoencoder (frozen) =====
    print("\n" + "="*60)
    print("Loading Vocos Autoencoder...")
    print("="*60)
    
    autoencoder = VocosAutoencoderWrapper(
        vocos_model=config.vocos.pretrained_model
    ).to(device)
    autoencoder.eval()
    
    # ===== Text-to-Latent Model =====
    print("\n" + "="*60)
    print("Creating Text-to-Latent Model...")
    print("="*60)
    
    model = TextToLatent(
        vocab_size=512,
        text_embed_dim=config.text_to_latent.text_embed_dim,
        hidden_dim=config.text_to_latent.hidden_dim,
        latent_dim=config.text_to_latent.latent_dim,  # 100 for Vocos
        num_layers=config.text_to_latent.num_layers,
        num_heads=config.text_to_latent.num_heads,
        dropout=config.text_to_latent.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ===== Loss Function =====
    loss_fn = FlowMatchingLoss(
        sigma_min=config.text_to_latent.flow_matching.sigma_min
    )
    
    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=tuple(config.train.optimizer.betas),
        weight_decay=config.train.optimizer.weight_decay
    )
    
    try:
        scaler = GradScaler("cuda", enabled=config.train.amp.enabled)
    except (TypeError, AttributeError):
        from torch.cuda.amp import GradScaler as OldGradScaler
        scaler = OldGradScaler(enabled=config.train.amp.enabled)
    
    # ===== Resume =====
    start_iteration = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iteration = ckpt["iteration"]
    
    # ===== Dataset =====
    print("\n" + "="*60)
    print("Loading Dataset...")
    print("="*60)
    
    from supertonic.data.preprocessing import AudioProcessor
    
    audio_processor = AudioProcessor(sample_rate=44100)
    tokenizer = CharacterTokenizer()
    
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
        tokenizer=tokenizer,
        max_duration=config.data.max_audio_duration,
        min_duration=config.data.min_audio_duration
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
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
    
    # ===== Training Loop =====
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    model.train()
    
    iteration = start_iteration
    total_iterations = config.train.total_iterations
    log_interval = 10
    validation_interval = config.train.validation_interval
    checkpoint_interval = config.train.checkpoint_interval
    
    pbar = tqdm(total=total_iterations, initial=start_iteration, desc="Training")
    
    epoch = 0
    while iteration < total_iterations:
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            iteration += 1
            
            losses = train_step(
                batch, model, autoencoder, loss_fn,
                optimizer, scaler, device,
                use_amp=config.train.amp.enabled
            )
            
            if iteration % log_interval == 0:
                pbar.set_postfix({"loss": f"{losses['loss']:.4f}"})
                pbar.update(log_interval)
            
            if iteration % validation_interval == 0:
                val_metrics = validate(val_loader, model, autoencoder, loss_fn, device)
                print(f"\nValidation at {iteration}: {val_metrics}")
            
            if iteration % checkpoint_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_{iteration}.pt",
                    iteration, model, optimizer, scaler, config
                )
        
        epoch += 1
    
    # Final checkpoint
    save_checkpoint(
        ckpt_dir / "checkpoint_final.pt",
        iteration, model, optimizer, scaler, config
    )
    
    pbar.close()
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vocos_tts.yaml")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    main(args)
