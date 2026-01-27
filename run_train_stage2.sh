#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  🎤 SUPERTONIC v2 - STAGE 2: Text-to-Latent Training
#═══════════════════════════════════════════════════════════════════════════════
#
#  Тренування Text-to-Latent module з voice cloning capabilities.
#  Використовує заморожений autoencoder з Stage 1.
#
#  Usage:
#    ./run_train_stage2.sh                    # Start training (auto-resume)
#    ./run_train_stage2.sh --fresh            # Start from scratch
#    ./run_train_stage2.sh --stop             # Stop training
#    ./run_train_stage2.sh --status           # Check status
#    ./run_train_stage2.sh --logs             # Tail log file
#
#  Prerequisites:
#    1. Trained autoencoder checkpoint from Stage 1
#    2. Multi-speaker data prepared with prepare_data_stage2.py
#
#═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE="config/22khz_optimal.yaml"
DATA_DIR="data"
MANIFEST_DIR="data/manifests_stage2"
OUTPUT_DIR="outputs/text_to_latent"
LOG_DIR="logs"
NUM_GPUS=4
MAX_RESTARTS=100
RESTART_DELAY=30

# Autoencoder checkpoint (from Stage 1)
AUTOENCODER_CHECKPOINT=""  # Will be auto-detected

PID_FILE=".training_stage2.pid"
MAIN_LOG="training_stage2.log"
MODE_FILE=".training_stage2_mode"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

find_autoencoder_checkpoint() {
    # Find best autoencoder checkpoint
    local dirs="checkpoints/autoencoder outputs/autoencoder_hifigan/checkpoints/autoencoder"
    
    for dir in $dirs; do
        if [[ -d "$dir" ]]; then
            local latest=$(ls -1 "$dir"/checkpoint_*.pt 2>/dev/null | \
                           sed 's/.*checkpoint_\([0-9]*\).*/\1/' | \
                           sort -n | tail -1)
            if [[ -n "$latest" ]]; then
                echo "${dir}/checkpoint_${latest}.pt"
                return
            fi
        fi
    done
}

find_stage2_checkpoint() {
    local dirs="checkpoints/text_to_latent ${OUTPUT_DIR}/checkpoints"
    
    for dir in $dirs; do
        if [[ -d "$dir" ]]; then
            local latest=$(ls -1 "$dir"/checkpoint_*.pt 2>/dev/null | \
                           sed 's/.*checkpoint_\([0-9]*\).*/\1/' | \
                           sort -n | tail -1)
            if [[ -n "$latest" ]]; then
                echo "${dir}/checkpoint_${latest}.pt"
                return
            fi
        fi
    done
}

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null
}

show_gpu_status() {
    echo -e "\n${CYAN}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx mem_used mem_total util; do
        local pct=$((100 * mem_used / mem_total))
        printf "  GPU %s: %5d/%5dMB (%2d%%)\n" "$idx" "$mem_used" "$mem_total" "$pct"
    done
}

check_manifests() {
    if [[ ! -f "${MANIFEST_DIR}/train.json" ]]; then
        echo -e "${RED}❌ Stage 2 manifests not found!${NC}"
        echo -e "   Run: ${CYAN}python scripts/prepare_data_stage2.py --medium${NC}"
        return 1
    fi
    
    local train_count=$(python3 -c "import json; print(len(json.load(open('${MANIFEST_DIR}/train.json'))))" 2>/dev/null || echo "0")
    local speaker_count=$(python3 -c "import json; print(len(set(e['speaker_id'] for e in json.load(open('${MANIFEST_DIR}/train.json')))))" 2>/dev/null || echo "0")
    
    echo -e "${GREEN}✅ Stage 2 manifests found${NC}"
    echo -e "   Train samples: ${train_count}"
    echo -e "   Speakers: ${speaker_count}"
    
    if [[ "$speaker_count" -lt 50 ]]; then
        echo -e "${YELLOW}⚠️  Warning: Only ${speaker_count} speakers. Recommend 100+ for good voice cloning.${NC}"
    fi
    
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# Command Handlers
# ═══════════════════════════════════════════════════════════════════════════════

cmd_stop() {
    echo -e "${YELLOW}🛑 Stopping Stage 2 training...${NC}"
    
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
            fi
            echo -e "${GREEN}✅ Training stopped${NC}"
        fi
        rm -f "$PID_FILE"
    fi
    
    # Kill any remaining processes
    pkill -f "train_text_to_latent.py" 2>/dev/null || true
    
    exit 0
}

cmd_status() {
    echo -e "\n${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  🎤 STAGE 2: Text-to-Latent Training Status${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo -e "\n${GREEN}✅ Training is RUNNING (PID: $pid)${NC}"
        
        # Show recent log
        if [[ -f "$MAIN_LOG" ]]; then
            echo -e "\n${CYAN}Recent progress:${NC}"
            tail -5 "$MAIN_LOG" | grep -E "\[.*\].*\|" | tail -3
        fi
    else
        echo -e "\n${YELLOW}⏸️  Training is NOT running${NC}"
    fi
    
    # Checkpoints
    local ae_ckpt=$(find_autoencoder_checkpoint)
    local s2_ckpt=$(find_stage2_checkpoint)
    
    echo -e "\n${CYAN}Checkpoints:${NC}"
    if [[ -n "$ae_ckpt" ]]; then
        local ae_step=$(echo "$ae_ckpt" | grep -oP 'checkpoint_\K[0-9]+')
        echo -e "  Autoencoder: ${GREEN}${ae_ckpt}${NC} (step ${ae_step})"
    else
        echo -e "  Autoencoder: ${RED}NOT FOUND${NC}"
    fi
    
    if [[ -n "$s2_ckpt" ]]; then
        local s2_step=$(echo "$s2_ckpt" | grep -oP 'checkpoint_\K[0-9]+')
        echo -e "  Stage 2: ${GREEN}${s2_ckpt}${NC} (step ${s2_step})"
    else
        echo -e "  Stage 2: ${YELLOW}No checkpoint yet${NC}"
    fi
    
    show_gpu_status
    
    exit 0
}

cmd_logs() {
    if [[ -f "$MAIN_LOG" ]]; then
        echo -e "${CYAN}📋 Tailing ${MAIN_LOG} (Ctrl+C to stop)${NC}\n"
        tail -f "$MAIN_LOG"
    else
        echo -e "${YELLOW}No log file found yet${NC}"
    fi
    exit 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# Training Function
# ═══════════════════════════════════════════════════════════════════════════════

run_training() {
    local fresh_start=$1
    
    # Find autoencoder checkpoint
    AUTOENCODER_CHECKPOINT=$(find_autoencoder_checkpoint)
    
    if [[ -z "$AUTOENCODER_CHECKPOINT" ]]; then
        echo -e "${RED}❌ No autoencoder checkpoint found!${NC}"
        echo -e "   Complete Stage 1 training first."
        exit 1
    fi
    
    local ae_step=$(echo "$AUTOENCODER_CHECKPOINT" | grep -oP 'checkpoint_\K[0-9]+')
    echo -e "${GREEN}✅ Using autoencoder: ${AUTOENCODER_CHECKPOINT} (step ${ae_step})${NC}"
    
    # Check manifests
    check_manifests || exit 1
    
    # Build resume args
    local resume_args=""
    
    if [[ "$fresh_start" != "true" ]]; then
        local s2_ckpt=$(find_stage2_checkpoint)
        if [[ -n "$s2_ckpt" ]]; then
            local s2_step=$(echo "$s2_ckpt" | grep -oP 'checkpoint_\K[0-9]+')
            echo -e "${CYAN}📂 Resuming from: ${s2_ckpt} (step ${s2_step})${NC}"
            resume_args="--resume ${s2_ckpt}"
        fi
    else
        echo -e "${YELLOW}🆕 Fresh start (ignoring existing checkpoints)${NC}"
    fi
    
    # Environment
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export OMP_NUM_THREADS=4
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Build command
    local cmd="torchrun --nproc_per_node=${NUM_GPUS} --master_port=29501 train_text_to_latent.py"
    cmd+=" --config ${CONFIG_FILE}"
    cmd+=" --autoencoder-checkpoint ${AUTOENCODER_CHECKPOINT}"
    cmd+=" --train-manifest ${MANIFEST_DIR}/train.json"
    cmd+=" --val-manifest ${MANIFEST_DIR}/val.json"
    cmd+=" --data_dir ${DATA_DIR}"
    cmd+=" --output_dir ${OUTPUT_DIR}"
    cmd+=" --no-wandb"
    cmd+=" ${resume_args}"
    
    echo -e "${GREEN}🚀 ${cmd}${NC}\n"
    
    # Run
    eval "$cmd"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

main() {
    cd "$(dirname "$0")"
    
    local fresh_start="false"
    
    # Parse args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stop)
                cmd_stop
                ;;
            --status)
                cmd_status
                ;;
            --logs)
                cmd_logs
                ;;
            --fresh)
                fresh_start="true"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if already running
    if is_running; then
        echo -e "${YELLOW}⚠️  Training already running!${NC}"
        echo -e "   Use: ${CYAN}./run_train_stage2.sh --status${NC}"
        exit 1
    fi
    
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  🎤 SUPERTONIC v2 - STAGE 2: Text-to-Latent${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    
    # Start training in background with auto-restart
    mkdir -p "$LOG_DIR"
    
    local restart_count=0
    
    while [[ $restart_count -lt $MAX_RESTARTS ]]; do
        echo -e "\n${CYAN}Starting training (attempt $((restart_count + 1)))...${NC}"
        
        # Save mode
        echo "$fresh_start" > "$MODE_FILE"
        
        # Run training
        run_training "$fresh_start" 2>&1 | tee -a "$MAIN_LOG" &
        local pid=$!
        echo $pid > "$PID_FILE"
        
        wait $pid
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            echo -e "${GREEN}✅ Training completed successfully!${NC}"
            break
        else
            echo -e "${YELLOW}⚠️  Training exited with code ${exit_code}${NC}"
            restart_count=$((restart_count + 1))
            
            if [[ $restart_count -lt $MAX_RESTARTS ]]; then
                echo -e "   Restarting in ${RESTART_DELAY}s..."
                sleep $RESTART_DELAY
                fresh_start="false"  # Resume on restart
            fi
        fi
    done
    
    rm -f "$PID_FILE" "$MODE_FILE"
}

main "$@"
