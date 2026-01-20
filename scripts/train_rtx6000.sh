#!/bin/bash
# =============================================================================
# Supertonic v2 TTS - RTX PRO 6000 Blackwell 96GB Training Script
# Optimized for: 1x RTX PRO 6000 96GB VRAM, 128GB RAM
# Expected training time: ~2-3 days
# =============================================================================

set -e

# =============================================================================
# CUDA Memory Optimization (less aggressive - we have 96GB!)
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Supertonic v2 TTS - RTX PRO 6000 96GB Training Pipeline      ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration - RTX 6000 optimized!
CONFIG="config/rtx6000_optimized.yaml"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"

# Create directories
mkdir -p $CHECKPOINT_DIR $LOG_DIR data/raw data/processed data/manifests

# =============================================================================
# Step 0: System Check
# =============================================================================
echo -e "\n${YELLOW}[Step 0] System Check${NC}"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    
    # Show VRAM
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✓ VRAM: ${VRAM} MiB${NC}"
    
    if [ "$VRAM" -ge 90000 ]; then
        echo -e "${GREEN}✓ 96GB VRAM detected - using aggressive batch sizes!${NC}"
    fi
else
    echo -e "${RED}✗ No GPU detected!${NC}"
    exit 1
fi

# Check Python
python3 --version

# Check RAM
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
echo -e "${GREEN}✓ Available RAM: ${FREE_RAM}GB${NC}"

# =============================================================================
# Step 1: Install Dependencies
# =============================================================================
echo -e "\n${YELLOW}[Step 1] Installing dependencies...${NC}"

pip install --upgrade pip
pip install -r requirements.txt

# Try to install flash-attn (optional)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available, continuing..."

echo -e "${GREEN}✓ Dependencies installed${NC}"

# =============================================================================
# Step 2: Download Datasets
# =============================================================================
echo -e "\n${YELLOW}[Step 2] Downloading datasets...${NC}"

if [ ! -f "data/raw/.download_complete" ]; then
    python scripts/download_datasets.py --full --eurospeech-hours 300
    touch data/raw/.download_complete
    echo -e "${GREEN}✓ Datasets downloaded${NC}"
else
    echo -e "${GREEN}✓ Datasets already downloaded${NC}"
fi

# =============================================================================
# Step 3: Prepare Manifests
# =============================================================================
echo -e "\n${YELLOW}[Step 3] Preparing manifests...${NC}"

if [ ! -f "data/manifests/train.json" ]; then
    python scripts/prepare_manifest.py --data-dir data/raw --output-dir data/manifests
    echo -e "${GREEN}✓ Manifests prepared${NC}"
else
    echo -e "${GREEN}✓ Manifests already exist${NC}"
fi

# =============================================================================
# Step 4: Preprocess Audio
# =============================================================================
echo -e "\n${YELLOW}[Step 4] Preprocessing audio...${NC}"

if [ ! -f "data/processed/.preprocess_complete" ]; then
    python scripts/preprocess.py --config $CONFIG
    touch data/processed/.preprocess_complete
    echo -e "${GREEN}✓ Audio preprocessed${NC}"
else
    echo -e "${GREEN}✓ Audio already preprocessed${NC}"
fi

# =============================================================================
# Step 5: Train Speech Autoencoder
# =============================================================================
echo -e "\n${YELLOW}[Step 5] Training Speech Autoencoder...${NC}"
echo "With 96GB VRAM and batch_size=24, this should take ~1.5-2 days"

AUTOENCODER_CKPT="$CHECKPOINT_DIR/autoencoder_final.pt"

if [ ! -f "$AUTOENCODER_CKPT" ]; then
    # Check for resume
    RESUME_FLAG=""
    LATEST_AE=$(ls -t $CHECKPOINT_DIR/autoencoder/checkpoint_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_AE" ]; then
        echo -e "${YELLOW}Resuming from: $LATEST_AE${NC}"
        RESUME_FLAG="--resume $LATEST_AE"
    fi
    
    python train_autoencoder.py --config $CONFIG $RESUME_FLAG --no-wandb
    
    # Copy final checkpoint
    FINAL_AE=$(ls -t $CHECKPOINT_DIR/autoencoder/checkpoint_*.pt 2>/dev/null | head -1)
    if [ -n "$FINAL_AE" ]; then
        cp "$FINAL_AE" "$AUTOENCODER_CKPT"
    fi
    
    echo -e "${GREEN}✓ Autoencoder training complete${NC}"
else
    echo -e "${GREEN}✓ Autoencoder already trained${NC}"
fi

# =============================================================================
# Step 6: Train Text-to-Latent (Flow Matching)
# =============================================================================
echo -e "\n${YELLOW}[Step 6] Training Text-to-Latent...${NC}"
echo "This will take approximately 24-30 hours on RTX 6000"

TTS_CKPT="$CHECKPOINT_DIR/tts_final.pt"

if [ ! -f "$TTS_CKPT" ]; then
    RESUME_FLAG=""
    LATEST_TTS=$(ls -t $CHECKPOINT_DIR/tts_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_TTS" ]; then
        echo -e "${YELLOW}Resuming from: $LATEST_TTS${NC}"
        RESUME_FLAG="--resume $LATEST_TTS"
    fi
    
    python train_text_to_latent.py --config $CONFIG --autoencoder $AUTOENCODER_CKPT $RESUME_FLAG --no-wandb
    echo -e "${GREEN}✓ Text-to-Latent training complete${NC}"
else
    echo -e "${GREEN}✓ Text-to-Latent already trained${NC}"
fi

# =============================================================================
# Step 7: Train Duration Predictor
# =============================================================================
echo -e "\n${YELLOW}[Step 7] Training Duration Predictor...${NC}"
echo "This will take approximately 10-15 minutes on RTX 6000"

DURATION_CKPT="$CHECKPOINT_DIR/duration_final.pt"

if [ ! -f "$DURATION_CKPT" ]; then
    python train_duration_predictor.py --config $CONFIG --no-wandb
    echo -e "${GREEN}✓ Duration Predictor training complete${NC}"
else
    echo -e "${GREEN}✓ Duration Predictor already trained${NC}"
fi

# =============================================================================
# Step 8: Export ONNX
# =============================================================================
echo -e "\n${YELLOW}[Step 8] Exporting to ONNX...${NC}"

if [ ! -d "onnx" ]; then
    python export_onnx.py \
        --autoencoder $AUTOENCODER_CKPT \
        --tts $TTS_CKPT \
        --duration $DURATION_CKPT \
        --output onnx/
    echo -e "${GREEN}✓ ONNX export complete${NC}"
else
    echo -e "${GREEN}✓ ONNX already exported${NC}"
fi

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 Training Complete! 🎉                    ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Checkpoints: checkpoints/                                     ║"
echo "║  ONNX Models: onnx/                                            ║"
echo "║  Logs:        logs/                                            ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Test synthesis:                                               ║"
echo "║  python inference.py --text 'Привіт, світ!' --output test.wav  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
