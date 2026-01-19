#!/bin/bash
# =============================================================================
# Supertonic v2 TTS - Vast.ai Setup Script
# =============================================================================
# Повне налаштування середовища для тренування TTS на vast.ai
#
# Usage:
#   chmod +x scripts/setup_vast.sh
#   ./scripts/setup_vast.sh
#
# Опції:
#   --skip-datasets    Пропустити завантаження датасетів
#   --minimal          Тільки українські датасети (~50GB)
#   --full             Всі датасети включно з англійськими (~500GB)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Supertonic v2 TTS - Vast.ai Setup                      ║${NC}"
echo -e "${BLUE}║     Ukrainian TTS Training Environment                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Parse arguments
SKIP_DATASETS=false
DATASET_MODE="minimal"

for arg in "$@"; do
    case $arg in
        --skip-datasets)
            SKIP_DATASETS=true
            shift
            ;;
        --minimal)
            DATASET_MODE="minimal"
            shift
            ;;
        --full)
            DATASET_MODE="full"
            shift
            ;;
    esac
done

# =============================================================================
# 1. System Setup
# =============================================================================
echo -e "\n${GREEN}[1/6] System Setup...${NC}"

# Update system
apt-get update -qq
apt-get install -y -qq wget curl git ffmpeg sox libsox-dev aria2 pigz pv

# Check GPU
echo -e "${YELLOW}GPU Info:${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# =============================================================================
# 2. Python Environment
# =============================================================================
echo -e "\n${GREEN}[2/6] Python Environment...${NC}"

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# =============================================================================
# 3. Directory Structure
# =============================================================================
echo -e "\n${GREEN}[3/6] Creating Directory Structure...${NC}"

mkdir -p data/{raw,processed,manifests}
mkdir -p data/raw/{ukrainian,english}
mkdir -p checkpoints/{autoencoder,tts,duration}
mkdir -p logs
mkdir -p cache
mkdir -p samples

echo "Directory structure created."

# =============================================================================
# 4. Download Datasets
# =============================================================================
if [ "$SKIP_DATASETS" = false ]; then
    echo -e "\n${GREEN}[4/6] Downloading Datasets (mode: ${DATASET_MODE})...${NC}"
    
    cd data/raw/ukrainian
    
    # -------------------------------------------------------------------------
    # M-AILABS Ukrainian (~20 hours, ~3GB)
    # -------------------------------------------------------------------------
    echo -e "${YELLOW}NOTE: M-AILABS (caito.de) is currently UNAVAILABLE${NC}"
    echo -e "${YELLOW}Using OpenTTS voices instead (higher quality anyway!)${NC}"
    
    # -------------------------------------------------------------------------
    # OpenTTS Individual Voices (RECOMMENDED - always available)
    # -------------------------------------------------------------------------
    echo -e "${YELLOW}Downloading OpenTTS Ukrainian Voices...${NC}"
    
    pip install -q huggingface_hub
    
    # Download each voice
    for voice in lada tetiana kateryna mykyta oleksa; do
        if [ ! -d "$voice" ]; then
            echo "Downloading voice: $voice..."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='speech-uk/opentts-$voice',
    repo_type='dataset',
    local_dir='$voice',
    ignore_patterns=['*.md', '.git*']
)
print('✓ $voice downloaded')
" || echo "Failed to download $voice"
        else
            echo "$voice already exists, skipping..."
        fi
    done
    echo -e "${GREEN}✓ OpenTTS voices downloaded${NC}"
    
    # -------------------------------------------------------------------------
    # Ukrainian Podcasts (high quality, ~100+ hours)
    # -------------------------------------------------------------------------
    echo -e "${YELLOW}Downloading Ukrainian Podcasts...${NC}"
    if [ ! -d "uk-pods" ]; then
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='taras-sereda/uk-pods',
    repo_type='dataset',
    local_dir='uk-pods'
)
print('Ukrainian Podcasts downloaded')
"
        echo -e "${GREEN}✓ Ukrainian Podcasts downloaded${NC}"
    else
        echo "Ukrainian Podcasts already exists, skipping..."
    fi
    
    # -------------------------------------------------------------------------
    # Common Voice Ukrainian (~80 hours)
    # -------------------------------------------------------------------------
    echo -e "${YELLOW}Common Voice Ukrainian...${NC}"
    echo -e "${YELLOW}Note: Common Voice requires manual download from:${NC}"
    echo -e "${BLUE}https://commonvoice.mozilla.org/uk/datasets${NC}"
    echo -e "After downloading, extract to: data/raw/ukrainian/common_voice_uk/"
    
    # -------------------------------------------------------------------------
    # Voice of America (~390 hours) - Optional
    # -------------------------------------------------------------------------
    if [ "$DATASET_MODE" = "full" ]; then
        echo -e "${YELLOW}Downloading Voice of America...${NC}"
        if [ ! -d "voice-of-america" ]; then
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='speech-uk/voice-of-america',
    repo_type='dataset',
    local_dir='voice-of-america'
)
"
            echo -e "${GREEN}✓ Voice of America downloaded${NC}"
        fi
        
        # Ukrainian Broadcast (~300 hours)
        echo -e "${YELLOW}Downloading Ukrainian Broadcast...${NC}"
        if [ ! -d "broadcast-speech-uk" ]; then
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Yehor/broadcast-speech-uk',
    repo_type='dataset',
    local_dir='broadcast-speech-uk'
)
"
            echo -e "${GREEN}✓ Ukrainian Broadcast downloaded${NC}"
        fi
    fi
    
    cd ../../..
    
    # -------------------------------------------------------------------------
    # English datasets (for pretraining, optional)
    # -------------------------------------------------------------------------
    if [ "$DATASET_MODE" = "full" ]; then
        cd data/raw/english
        
        echo -e "${YELLOW}Downloading LJSpeech...${NC}"
        if [ ! -d "LJSpeech-1.1" ]; then
            wget -c https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
            tar -xjf LJSpeech-1.1.tar.bz2
            rm LJSpeech-1.1.tar.bz2
            echo -e "${GREEN}✓ LJSpeech downloaded${NC}"
        fi
        
        cd ../../..
    fi
    
else
    echo -e "\n${YELLOW}[4/6] Skipping dataset download...${NC}"
fi

# =============================================================================
# 5. Prepare Manifests
# =============================================================================
echo -e "\n${GREEN}[5/6] Preparing Manifests...${NC}"

python scripts/prepare_manifest.py \
    --data-dir data/raw \
    --output-dir data/manifests \
    --val-split 0.02 \
    --test-split 0.01

echo -e "${GREEN}✓ Manifests created${NC}"

# =============================================================================
# 6. Verify Setup
# =============================================================================
echo -e "\n${GREEN}[6/6] Verifying Setup...${NC}"

python -c "
import torch
import torchaudio
from pathlib import Path
import json

print('=== Environment Check ===')
print(f'PyTorch version: {torch.__version__}')
print(f'Torchaudio version: {torchaudio.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

print()
print('=== Dataset Check ===')
manifest_path = Path('data/manifests/train_manifest.json')
if manifest_path.exists():
    with open(manifest_path) as f:
        data = json.load(f)
    print(f'Training samples: {len(data)}')
    
    # Calculate total hours
    total_duration = sum(item.get('duration', 0) for item in data)
    print(f'Total duration: {total_duration/3600:.1f} hours')
    
    # Languages
    langs = set(item.get('language', 'uk') for item in data)
    print(f'Languages: {langs}')
else:
    print('No manifest found. Run prepare_manifest.py first.')

print()
print('=== Model Import Check ===')
try:
    from supertonic.models.speech_autoencoder import LatentEncoder, LatentDecoder
    from supertonic.models.text_to_latent import TextToLatent
    from supertonic.models.duration_predictor import DurationPredictor
    print('✓ All models imported successfully')
except Exception as e:
    print(f'✗ Import error: {e}')
"

# =============================================================================
# Done!
# =============================================================================
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Setup Complete!                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "
${BLUE}Next steps:${NC}

1. If you haven't downloaded Common Voice Ukrainian:
   - Go to: https://commonvoice.mozilla.org/uk/datasets
   - Download and extract to: data/raw/ukrainian/common_voice_uk/

2. Start training:
   ${YELLOW}# Train autoencoder first (~7-8 days on 1x5090)${NC}
   python train_autoencoder.py --config config/default.yaml

   ${YELLOW}# Then train TTS (~4-5 days on 1x5090)${NC}
   python train_text_to_latent.py --config config/default.yaml \\
       --autoencoder-checkpoint checkpoints/autoencoder/checkpoint_final.pt

   ${YELLOW}# Finally, train duration predictor (~20 min)${NC}
   python train_duration_predictor.py --config config/default.yaml

3. Monitor training:
   ${YELLOW}wandb login${NC}
   # Or use tensorboard:
   ${YELLOW}tensorboard --logdir logs/ --port 6006${NC}

4. Test inference:
   python inference.py --text \"Привіт, як справи?\" \\
       --reference samples/reference.wav \\
       --output samples/output.wav
"
