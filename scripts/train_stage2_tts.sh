#!/bin/bash
# RTX PRO 6000 - Stage 2: Text-to-Latent Training
# Використовує pretrained autoencoder для генерації тексту

set -e

echo "🧠 Stage 2: Text-to-Latent Training"
echo "==============================================="
echo "Autoencoder: checkpoint_75000.pt"
echo "Batch size: 64 (optimal for 96GB)"
echo "==============================================="

# CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export WANDB_MODE=disabled

# Перевірка чекпоінта
AUTOENCODER_CKPT="checkpoints/autoencoder/checkpoint_75000.pt"

if [ ! -f "$AUTOENCODER_CKPT" ]; then
    # Шукаємо останній чекпоінт
    AUTOENCODER_CKPT=$(ls -t checkpoints/autoencoder/checkpoint_*.pt 2>/dev/null | head -1)
    if [ -z "$AUTOENCODER_CKPT" ]; then
        echo "❌ No autoencoder checkpoint found!"
        exit 1
    fi
    echo "📦 Using latest checkpoint: $AUTOENCODER_CKPT"
fi

# Training
python train_text_to_latent.py \
    --config config/rtx6000_optimal.yaml \
    --autoencoder-checkpoint "$AUTOENCODER_CKPT" \
    --batch-size 64 \
    --no-wandb

echo "✅ Training complete!"
