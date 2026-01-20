#!/bin/bash
# RTX PRO 6000 OPTIMAL training script for Supertonic v2 TTS
# OPTIMAL balance: batch_size=32, workers=8, no OOM
# Expected: ~1.2s/it (33% faster than baseline)

set -e

echo "⚡ RTX PRO 6000 OPTIMAL MODE - Training Speech Autoencoder"
echo "==============================================="
echo "Batch size: 28 (sweet spot, no memory pressure)"
echo "Workers: 8"
echo "Validation: every 2000 steps"
echo "Expected: ~1.35s/it"
echo "VRAM usage: ~70GB (safe)"
echo "Speed gain: ~15% faster than baseline!"
echo "==============================================="

# CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Training
python train_autoencoder.py \
    --config config/rtx6000_optimal.yaml \
    --resume checkpoints/autoencoder/checkpoint_10000.pt

echo "✅ Training complete!"
