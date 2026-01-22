#!/usr/bin/env python3
"""
Interactive Transfer Vocos Reconstruction Test.

Features:
- Auto-discover all checkpoints
- Select checkpoints via checkboxes
- Random audio samples from each dataset
- Separate output folder per checkpoint

Usage:
    python scripts/test_transfer_vocos.py
    python scripts/test_transfer_vocos.py --non-interactive --checkpoint checkpoints/transfer_vocos/checkpoint_400.pt
"""

import argparse
import sys
import json
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from supertonic.models.transfer_vocos import create_transfer_learning_adapter

try:
    from questionary import checkbox, select, confirm
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False


def find_checkpoints(checkpoint_dir: str = "checkpoints/transfer_vocos") -> list:
    """Find all available checkpoints."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return []
    
    checkpoints = sorted(ckpt_path.glob("checkpoint_*.pt"), key=lambda x: x.stat().st_mtime)
    return checkpoints


def get_random_audio_samples(manifest_path: str, num_samples: int = 5) -> list:
    """Get random audio samples from manifest."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        return []
    
    with open(manifest, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    # Filter existing files
    valid_samples = [s for s in samples if Path(s["audio_path"]).exists()]
    
    if len(valid_samples) <= num_samples:
        return valid_samples
    
    return random.sample(valid_samples, num_samples)


def discover_datasets() -> dict:
    """Discover all available datasets from manifests."""
    manifests_dir = Path("data/manifests")
    datasets = {}
    
    if manifests_dir.exists():
        for manifest in manifests_dir.glob("*.json"):
            name = manifest.stem
            samples = get_random_audio_samples(str(manifest), num_samples=5)
            if samples:
                datasets[name] = {
                    "manifest": str(manifest),
                    "samples": samples
                }
    
    return datasets


def test_single_checkpoint(
    checkpoint_path: Path,
    audio_files: list,
    output_dir: Path,
    device: torch.device
):
    """Test single checkpoint with given audio files."""
    print(f"\n{'='*60}")
    print(f"Testing: {checkpoint_path.name}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get("config", {})
    pretrained_checkpoint = config.get("transfer", {}).get("pretrained_checkpoint", "checkpoints/autoencoder/checkpoint_80000.pt")
    vocos_model = config.get("vocos", {}).get("pretrained_model", "charactr/vocos-mel-24khz")
    
    # Create model
    model = create_transfer_learning_adapter(
        pretrained_checkpoint=pretrained_checkpoint,
        vocos_model=vocos_model,
        freeze_encoder=True,
        freeze_vocos=True,
        device=device
    )
    
    # Load adapter weights
    model.adapter.load_state_dict(checkpoint["adapter"])
    model.eval()
    
    iteration = checkpoint.get("iteration", "unknown")
    print(f"Iteration: {iteration}")
    
    # Create output subdir
    ckpt_output = output_dir / checkpoint_path.stem
    ckpt_output.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, audio_info in enumerate(audio_files):
        if isinstance(audio_info, dict):
            audio_path = Path(audio_info["audio_path"])
            text = audio_info.get("text", "")[:50]
        else:
            audio_path = Path(audio_info)
            text = ""
        
        if not audio_path.exists():
            print(f"  [{i+1}] Skipping {audio_path.name} (not found)")
            continue
        
        print(f"\n  [{i+1}] {audio_path.name}")
        if text:
            print(f"      Text: {text}...")
        
        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        
        # Resample to 24kHz
        if sr != 24000:
            audio = F.resample(audio, sr, 24000)
        
        # Limit length (10 sec max)
        max_len = 24000 * 10
        if len(audio) > max_len:
            audio = audio[:max_len]
        
        audio = audio.unsqueeze(0).to(device)
        
        # Reconstruct
        with torch.no_grad():
            recon_audio = model(audio)
        
        # Match lengths
        min_len = min(audio.shape[-1], recon_audio.shape[-1])
        audio = audio[..., :min_len]
        recon_audio = recon_audio[..., :min_len]
        
        # Calculate metrics
        l1_loss = torch.nn.functional.l1_loss(recon_audio, audio).item()
        mse_loss = torch.nn.functional.mse_loss(recon_audio, audio).item()
        
        print(f"      L1: {l1_loss:.4f} | MSE: {mse_loss:.6f}")
        
        results.append({
            "file": audio_path.name,
            "l1": l1_loss,
            "mse": mse_loss
        })
        
        # Save outputs
        orig_path = ckpt_output / f"{i+1:02d}_original_{audio_path.stem}.wav"
        recon_path = ckpt_output / f"{i+1:02d}_recon_{audio_path.stem}.wav"
        
        torchaudio.save(str(orig_path), audio.cpu(), 24000)
        torchaudio.save(str(recon_path), recon_audio.cpu(), 24000)
    
    # Summary
    if results:
        avg_l1 = sum(r["l1"] for r in results) / len(results)
        avg_mse = sum(r["mse"] for r in results) / len(results)
        print(f"\n  Average L1: {avg_l1:.4f} | MSE: {avg_mse:.6f}")
        print(f"  Saved to: {ckpt_output}/")
    
    return results


def interactive_mode():
    """Run interactive test session."""
    if not HAS_QUESTIONARY:
        print("❌ Please install questionary: pip install questionary")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # 1. Find checkpoints
    print("🔍 Searching for checkpoints...")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found in checkpoints/transfer_vocos/")
        return
    
    print(f"   Found {len(checkpoints)} checkpoints\n")
    
    # 2. Select checkpoints
    ckpt_choices = [
        {"name": f"{c.name} ({c.stat().st_size / 1024 / 1024:.1f} MB)", "value": c}
        for c in checkpoints
    ]
    
    selected_ckpts = checkbox(
        "Select checkpoints to test:",
        choices=ckpt_choices,
        instruction="(Use space to select, enter to confirm)"
    ).ask()
    
    if not selected_ckpts:
        print("No checkpoints selected. Exiting.")
        return
    
    # 3. Discover datasets
    print("\n🔍 Discovering datasets...")
    datasets = discover_datasets()
    
    if not datasets:
        print("❌ No datasets found in data/manifests/")
        return
    
    print(f"   Found {len(datasets)} datasets: {', '.join(datasets.keys())}\n")
    
    # 4. Select audio samples
    all_audio = []
    
    for name, data in datasets.items():
        print(f"\n📁 Dataset: {name} ({len(data['samples'])} samples)")
        
        sample_choices = [
            {
                "name": f"{Path(s['audio_path']).name} - {s.get('text', '')[:40]}...",
                "value": s
            }
            for s in data["samples"]
        ]
        
        selected_samples = checkbox(
            f"Select samples from {name}:",
            choices=sample_choices,
            instruction="(Space to select, Enter to confirm)"
        ).ask()
        
        if selected_samples:
            all_audio.extend(selected_samples)
    
    if not all_audio:
        print("\n❌ No audio samples selected. Exiting.")
        return
    
    print(f"\n✅ Selected {len(all_audio)} audio files")
    print(f"✅ Testing {len(selected_ckpts)} checkpoints")
    
    # 5. Confirm
    if not confirm("Start testing?").ask():
        return
    
    # 6. Run tests
    output_dir = Path("test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for ckpt in selected_ckpts:
        results = test_single_checkpoint(ckpt, all_audio, output_dir, device)
        all_results[ckpt.name] = results
    
    # 7. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for ckpt_name, results in all_results.items():
        if results:
            avg_l1 = sum(r["l1"] for r in results) / len(results)
            print(f"  {ckpt_name}: avg L1 = {avg_l1:.4f}")
    
    print(f"\n✅ All outputs saved to: {output_dir}/")


def non_interactive_mode(checkpoint_path: str, audio_path: str = None, output_dir: str = "test_outputs"):
    """Run non-interactive test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return
    
    # Get audio files
    if audio_path:
        audio_files = [{"audio_path": audio_path}]
    else:
        datasets = discover_datasets()
        audio_files = []
        for data in datasets.values():
            audio_files.extend(data["samples"][:3])  # 3 from each
    
    if not audio_files:
        print("❌ No audio files found")
        return
    
    test_single_checkpoint(ckpt, audio_files, Path(output_dir), device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run in non-interactive mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (for non-interactive mode)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Audio file path (optional)")
    parser.add_argument("--output", type=str, default="test_outputs",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.non_interactive:
        if not args.checkpoint:
            print("❌ --checkpoint required for non-interactive mode")
            sys.exit(1)
        non_interactive_mode(args.checkpoint, args.audio, args.output)
    else:
        interactive_mode()
