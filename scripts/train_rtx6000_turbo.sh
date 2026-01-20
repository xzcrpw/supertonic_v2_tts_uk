#!/bin/bash
# RTX PRO 6000 TURBO training script for Supertonic v2 TTS
# MAXIMIZED for speed: batch_size=40, workers=8, validation_interval=2000

set -e

echo "🚀 RTX PRO 6000 TURBO MODE - Training Speech Autoencoder"
echo "==============================================="
echo "Batch size: 40 (was 24)"
echo "Workers: 8 (was 4)"
echo "Validation: every 2000 steps (was 1000)"
echo "Expected: ~0.95s/it (was 1.55s/it)"
echo "Speed gain: ~60% faster!"
echo "==============================================="

# CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Training
python train_autoencoder.py \
    --config config/rtx6000_turbo.yaml \
    --resume checkpoints/autoencoder/checkpoint_10000.pt

echo "✅ Training complete!"
