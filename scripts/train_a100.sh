#!/bin/bash
# =============================================================================
# Supertonic v2 TTS - A100 Training Script
# Optimized for: 1x NVIDIA A100 PCIE 80GB
# =============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Supertonic v2 TTS - A100 80GB Training Pipeline            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
CONFIG="config/a100_optimized.yaml"
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
else
    echo -e "${RED}✗ No GPU detected!${NC}"
    exit 1
fi

# Check Python
python3 --version

# =============================================================================
# Step 1: Install Dependencies
# =============================================================================
echo -e "\n${YELLOW}[Step 1] Installing dependencies...${NC}"

pip install --upgrade pip
pip install -r requirements.txt

# Install additional A100-optimized packages
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
echo "This will take approximately 20-24 hours on A100"

AUTOENCODER_CKPT="$CHECKPOINT_DIR/autoencoder_final.pt"

if [ ! -f "$AUTOENCODER_CKPT" ]; then
    # Check for resume
    RESUME_FLAG=""
    LATEST_AE=$(ls -t $CHECKPOINT_DIR/autoencoder_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_AE" ]; then
        echo -e "${YELLOW}Resuming from: $LATEST_AE${NC}"
        RESUME_FLAG="--resume $LATEST_AE"
    fi
    
    python train_autoencoder.py --config $CONFIG $RESUME_FLAG
    echo -e "${GREEN}✓ Autoencoder training complete${NC}"
else
    echo -e "${GREEN}✓ Autoencoder already trained${NC}"
fi

# =============================================================================
# Step 6: Train Text-to-Latent (Flow Matching)
# =============================================================================
echo -e "\n${YELLOW}[Step 6] Training Text-to-Latent...${NC}"
echo "This will take approximately 30-36 hours on A100"

TTS_CKPT="$CHECKPOINT_DIR/tts_final.pt"

if [ ! -f "$TTS_CKPT" ]; then
    # Check for resume
    RESUME_FLAG=""
    LATEST_TTS=$(ls -t $CHECKPOINT_DIR/tts_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_TTS" ]; then
        echo -e "${YELLOW}Resuming from: $LATEST_TTS${NC}"
        RESUME_FLAG="--resume $LATEST_TTS"
    fi
    
    python train_tts.py --config $CONFIG --autoencoder $AUTOENCODER_CKPT $RESUME_FLAG
    echo -e "${GREEN}✓ Text-to-Latent training complete${NC}"
else
    echo -e "${GREEN}✓ Text-to-Latent already trained${NC}"
fi

# =============================================================================
# Step 7: Train Duration Predictor
# =============================================================================
echo -e "\n${YELLOW}[Step 7] Training Duration Predictor...${NC}"
echo "This will take approximately 15-20 minutes on A100"

DURATION_CKPT="$CHECKPOINT_DIR/duration_final.pt"

if [ ! -f "$DURATION_CKPT" ]; then
    python train_duration.py --config $CONFIG
    echo -e "${GREEN}✓ Duration Predictor training complete${NC}"
else
    echo -e "${GREEN}✓ Duration Predictor already trained${NC}"
fi

# =============================================================================
# Step 8: Export ONNX (Optional)
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
echo "║  python inference.py --text 'Привіт, світ!' --output test.wav ║"
echo "╚════════════════════════════════════════════════════════════════╝"
