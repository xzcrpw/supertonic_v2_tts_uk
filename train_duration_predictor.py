"""
Training Script для Duration Predictor

Швидке тренування duration predictor (~3000 iterations).

Особливості:
- Utterance-level duration prediction (не per-phoneme!)
- L1 loss на ground-truth duration
- Швидке тренування: ~3000 iterations (кілька хвилин)

Конфігурація:
- Optimizer: AdamW, lr=5e-4
- Iterations: 3000
- Batch size: 64

Референс: Supertonic v2 paper
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.duration_predictor import DurationPredictor
from supertonic.models.speech_autoencoder import LatentEncoder
from supertonic.losses.duration_loss import DurationLoss
from supertonic.losses.flow_matching_loss import compress_latents
from supertonic.data.dataset import DurationDataset
from supertonic.data.collate import duration_collate_fn
from supertonic.data.preprocessing import AudioProcessor
from supertonic.data.tokenizer import CharacterTokenizer


def save_checkpoint(
    path: Path,
    iteration: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict
):
    """Зберігає checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at iteration {iteration}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """Завантажує checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]


def train_step(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    latent_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: DurationLoss,
    device: torch.device
) -> Dict[str, float]:
    """Один training step."""
    text_ids = batch["text_ids"].to(device)
    text_mask = batch["text_mask"].to(device)
    reference_mel = batch["reference_mel"].to(device)
    reference_mask = batch["reference_mask"].to(device)
    target_duration = batch["durations"].to(device)
    
    optimizer.zero_grad()
    
    # Encode reference to latent
    with torch.no_grad():
        ref_latent = latent_encoder(reference_mel)
        ref_compressed = compress_latents(ref_latent, compression_factor=6)
    
    # Predict duration
    predicted_duration = model(
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=ref_compressed,
        reference_mask=reference_mask
    )
    
    # Loss
    losses = loss_fn(predicted_duration, target_duration)
    loss = losses["total"]
    
    loss.backward()
    optimizer.step()
    
    return {
        "loss": losses["total"].item(),
        "l1_loss": losses["l1"].item(),
        "percent_error": losses["percent_error"].item()
    }


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    model: nn.Module,
    latent_encoder: nn.Module,
    loss_fn: DurationLoss,
    device: torch.device,
    max_samples: int = 200
) -> Dict[str, float]:
    """Validation loop."""
    model.eval()
    latent_encoder.eval()
    
    total_loss = 0.0
    total_percent_error = 0.0
    num_samples = 0
    
    for batch in dataloader:
        if num_samples >= max_samples:
            break
        
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        reference_mel = batch["reference_mel"].to(device)
        reference_mask = batch["reference_mask"].to(device)
        target_duration = batch["durations"].to(device)
        
        ref_latent = latent_encoder(reference_mel)
        ref_compressed = compress_latents(ref_latent, compression_factor=6)
        
        predicted_duration = model(
            text_ids=text_ids,
            text_mask=text_mask,
            reference_latent=ref_compressed,
            reference_mask=reference_mask
        )
        
        losses = loss_fn(predicted_duration, target_duration)
        
        total_loss += losses["total"].item() * text_ids.size(0)
        total_percent_error += losses["percent_error"].item() * text_ids.size(0)
        num_samples += text_ids.size(0)
    
    model.train()
    
    return {
        "val_loss": total_loss / max(num_samples, 1),
        "val_percent_error": total_percent_error / max(num_samples, 1)
    }


def main(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override
    if args.batch_size:
        config.train_duration.batch_size = args.batch_size
    if args.lr:
        config.train_duration.learning_rate = args.lr
    
    # Logging
    if not args.no_wandb:
        wandb.init(
            project=config.logging.project,
            name=f"duration_predictor_{time.strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(config)
        )
    
    # Output directories
    checkpoint_dir = Path(config.output.checkpoint_dir) / "duration"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio processor & tokenizer
    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels
    )
    
    tokenizer = CharacterTokenizer(languages=config.languages.supported)
    
    # Dataset
    train_dataset = DurationDataset(
        manifest_path=config.data.train_manifest,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        duration_unit="frames"
    )
    
    val_dataset = DurationDataset(
        manifest_path=config.data.val_manifest,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        duration_unit="frames"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_duration.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=duration_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train_duration.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=duration_collate_fn
    )
    
    # Load pretrained latent encoder
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
    
    # Duration Predictor
    model = DurationPredictor(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config.duration_predictor.hidden_dim,
        num_convnext_blocks=config.duration_predictor.num_convnext_blocks,
        kernel_size=config.duration_predictor.kernel_size
    ).to(device)
    
    # Loss
    loss_fn = DurationLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train_duration.learning_rate,
        betas=tuple(config.train_duration.optimizer.betas),
        weight_decay=config.train_duration.optimizer.weight_decay
    )
    
    # Resume
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(Path(args.resume), model, optimizer)
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    total_iterations = config.train_duration.total_iterations
    checkpoint_interval = config.train_duration.checkpoint_interval
    log_interval = 50
    
    iteration = start_iteration
    epoch = 0
    
    pbar = tqdm(total=total_iterations, initial=start_iteration, desc="Training Duration")
    
    while iteration < total_iterations:
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            
            losses = train_step(
                batch, model, latent_encoder, optimizer, loss_fn, device
            )
            
            iteration += 1
            
            # Logging
            if iteration % log_interval == 0:
                if not args.no_wandb:
                    wandb.log(losses, step=iteration)
                
                pbar.set_postfix({
                    "loss": f"{losses['loss']:.4f}",
                    "err%": f"{losses['percent_error']:.1f}"
                })
                pbar.update(log_interval)
            
            # Checkpoint
            if iteration % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_{iteration}.pt",
                    iteration, model, optimizer,
                    OmegaConf.to_container(config)
                )
                
                # Validation
                val_metrics = validate(
                    val_loader, model, latent_encoder, loss_fn, device
                )
                
                if not args.no_wandb:
                    wandb.log(val_metrics, step=iteration)
                
                print(f"\nValidation: {val_metrics}")
        
        epoch += 1
    
    # Final checkpoint
    save_checkpoint(
        checkpoint_dir / "checkpoint_final.pt",
        iteration, model, optimizer,
        OmegaConf.to_container(config)
    )
    
    pbar.close()
    
    if not args.no_wandb:
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Duration Predictor")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--autoencoder-checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    main(args)
