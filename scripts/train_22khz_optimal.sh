#!/bin/bash
# ============================================================================
# SUPERTONIC V2 TTS - 22kHz OPTIMAL TRAINING SCRIPT
# ============================================================================
# Hardware: RTX PRO 6000 96GB (vast.ai $0.446/hr)
# Sample rate: 22050 Hz (TTS standard)
# Waveform L1 loss: ENABLED (fixes metallic sound)
#
# Expected performance:
#   - ~1.2 s/iteration
#   - 100k iterations = ~33 hours = ~$15
#   - 200k iterations = ~66 hours = ~$30
#
# Usage:
#   ./scripts/train_22khz_optimal.sh [--resume CHECKPOINT]
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        SUPERTONIC V2 TTS - 22kHz OPTIMAL TRAINING                   ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Sample Rate:     22050 Hz (TTS standard)                           ║"
echo "║  Batch Size:      32 (optimal for 96GB)                             ║"
echo "║  Segment:         6 seconds                                          ║"
echo "║  Workers:         8                                                  ║"
echo "║  Waveform Loss:   ENABLED (λ=10, fixes metallic sound)              ║"
echo "║                                                                      ║"
echo "║  Expected:        ~1.2 s/it                                          ║"
echo "║  100k steps:      ~33 hours (~\$15)                                  ║"
echo "║  200k steps:      ~66 hours (~\$30)                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ========== ENVIRONMENT SETUP ==========
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

# ========== CHECK GPU ==========
echo -e "${YELLOW}Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ========== CHECK DISK SPACE ==========
echo -e "${YELLOW}Checking disk space...${NC}"
df -h . | head -2
echo ""

# ========== PARSE ARGUMENTS ==========
RESUME_ARG=""
if [[ "$1" == "--resume" && -n "$2" ]]; then
    if [[ -f "$2" ]]; then
        RESUME_ARG="--resume $2"
        echo -e "${GREEN}Resuming from: $2${NC}"
    else
        echo -e "${RED}Checkpoint not found: $2${NC}"
        exit 1
    fi
fi

# ========== CREATE DIRECTORIES ==========
mkdir -p checkpoints/autoencoder
mkdir -p logs
mkdir -p samples

# ========== START TRAINING ==========
echo -e "${GREEN}Starting training...${NC}"
echo "Timestamp: $(date)"
echo ""

python train_autoencoder.py \
    --config config/22khz_optimal.yaml \
    $RESUME_ARG \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo -e "${GREEN}✅ Training complete!${NC}"
echo "Timestamp: $(date)"
