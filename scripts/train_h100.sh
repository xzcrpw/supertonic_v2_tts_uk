#!/bin/bash
# =============================================================================
# Supertonic v2 TTS - H100 Quick Start Script
# =============================================================================
# Швидкий старт для H100 SXM з оптимальними налаштуваннями
# Включає: часті checkpoints, моніторинг, auto-resume
# =============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Supertonic v2 TTS - H100 SXM Optimized                 ║"
echo "║     ~2-3 дні замість 12-14 на RTX 4090!                    ║"
echo "╚════════════════════════════════════════════════════════════╝"

# =============================================================================
# 1. Перевірка GPU
# =============================================================================
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Перевіряємо чи це H100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" == *"H100"* ]]; then
    echo "✅ H100 detected! Using optimized config."
    CONFIG="config/h100_optimized.yaml"
else
    echo "⚠️ Not H100, using default config."
    CONFIG="config/default.yaml"
fi

# =============================================================================
# 2. CUDA Optimizations для H100
# =============================================================================
echo ""
echo "=== Applying H100 CUDA Optimizations ==="

# TF32 для максимальної швидкості
export CUDA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# NCCL optimizations
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# Memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✅ CUDA optimizations applied"

# =============================================================================
# 3. Встановлення залежностей
# =============================================================================
echo ""
echo "=== Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

# Flash Attention 2 для H100 (важливо!)
pip install flash-attn --no-build-isolation

echo "✅ Dependencies installed"

# =============================================================================
# 4. Завантаження датасетів
# =============================================================================
echo ""
echo "=== Downloading Datasets ==="
python scripts/download_datasets.py --minimal

# =============================================================================
# 5. Підготовка manifests
# =============================================================================
echo ""
echo "=== Preparing Manifests ==="
python scripts/prepare_manifest.py --data-dir data/raw --output-dir data/manifests

# =============================================================================
# 6. WandB Login (опційно)
# =============================================================================
echo ""
echo "=== WandB Setup ==="
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "✅ WandB configured"
else
    echo "⚠️ WANDB_API_KEY not set. Logging to local files."
    echo "   Set it with: export WANDB_API_KEY=your_key"
fi

# =============================================================================
# 7. Training Pipeline
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Starting Training Pipeline                             ║"
echo "║     Estimated time: ~2-3 days on H100                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Функція для auto-resume
run_with_resume() {
    SCRIPT=$1
    CHECKPOINT_DIR=$2
    EXTRA_ARGS=$3
    
    # Шукаємо останній checkpoint
    LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/checkpoint_*.pt 2>/dev/null | head -1 || echo "")
    
    if [ -n "$LATEST_CKPT" ]; then
        echo "📂 Found checkpoint: $LATEST_CKPT"
        echo "   Resuming training..."
        python $SCRIPT --config $CONFIG --resume $LATEST_CKPT $EXTRA_ARGS
    else
        echo "🆕 Starting fresh training..."
        python $SCRIPT --config $CONFIG $EXTRA_ARGS
    fi
}

# -----------------------------------------------------------------------------
# Етап 1: Autoencoder (~1 день на H100)
# -----------------------------------------------------------------------------
echo ""
echo "=== Stage 1/3: Autoencoder Training ==="
echo "    Batch size: 48 | Iterations: 1.5M"
echo "    Estimated time: ~20-24 hours"
echo ""

run_with_resume "train_autoencoder.py" "checkpoints/autoencoder" ""

# -----------------------------------------------------------------------------
# Етап 2: Text-to-Latent (~1-1.5 дні на H100)
# -----------------------------------------------------------------------------
echo ""
echo "=== Stage 2/3: Text-to-Latent Training ==="
echo "    Batch size: 128×4=512 | Iterations: 700k"
echo "    Estimated time: ~24-36 hours"
echo ""

AE_CKPT=$(ls -t checkpoints/autoencoder/checkpoint_*.pt | head -1)
run_with_resume "train_text_to_latent.py" "checkpoints/tts" "--autoencoder-checkpoint $AE_CKPT"

# -----------------------------------------------------------------------------
# Етап 3: Duration Predictor (~10-15 хвилин на H100)
# -----------------------------------------------------------------------------
echo ""
echo "=== Stage 3/3: Duration Predictor Training ==="
echo "    Batch size: 256 | Iterations: 3k"
echo "    Estimated time: ~10-15 minutes"
echo ""

run_with_resume "train_duration_predictor.py" "checkpoints/duration" ""

# =============================================================================
# 8. Done!
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🎉 Training Complete!                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Checkpoints saved in:"
echo "  - checkpoints/autoencoder/"
echo "  - checkpoints/tts/"
echo "  - checkpoints/duration/"
echo ""
echo "Test inference:"
echo "  python inference.py --text 'Привіт, як справи?' \\"
echo "      --reference samples/reference.wav \\"
echo "      --output output.wav"
echo ""
echo "Export to ONNX:"
echo "  python export_onnx.py --checkpoint-dir checkpoints"
