#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  🎤 SUPERTONIC v2 - Stage 2: Text-to-Latent Training (nohup)
#═══════════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    ./run_train_stage2_nohup.sh              # Start training (auto-resume)
#    ./run_train_stage2_nohup.sh --fresh      # Start from scratch
#    ./run_train_stage2_nohup.sh --stop       # Stop training
#    ./run_train_stage2_nohup.sh --status     # Check status
#    ./run_train_stage2_nohup.sh --logs       # Tail the log file
#
#  After starting, you can safely close the terminal!
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

# Training mode (set by CLI flags)
TRAINING_MODE="auto"  # auto | fresh

PID_FILE=".training_stage2.pid"
MAIN_LOG="training_stage2.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

find_autoencoder_checkpoint() {
    local dirs="checkpoints/autoencoder outputs/autoencoder/checkpoints outputs/autoencoder_hifigan/checkpoints/autoencoder"
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
        echo -e "   Run: ${CYAN}python scripts/prepare_data.py${NC}"
        return 1
    fi
    
    # Support both JSON array and JSON Lines format
    local train_count=$(python3 -c "
import json
try:
    data = json.load(open('${MANIFEST_DIR}/train.json'))
except:
    data = [json.loads(l) for l in open('${MANIFEST_DIR}/train.json') if l.strip()]
print(len(data))
" 2>/dev/null || echo "0")
    local speaker_count=$(python3 -c "
import json
try:
    data = json.load(open('${MANIFEST_DIR}/train.json'))
except:
    data = [json.loads(l) for l in open('${MANIFEST_DIR}/train.json') if l.strip()]
print(len(set(e.get('speaker_id', 'unk') for e in data)))
" 2>/dev/null || echo "0")
    
    echo -e "   Train samples: ${train_count}, Speakers: ${speaker_count}"
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop (called with --loop flag)
# ═══════════════════════════════════════════════════════════════════════════════

run_training_loop() {
    local restart_count=0
    
    mkdir -p "$LOG_DIR"
    
    # Environment - 4 GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export OMP_NUM_THREADS=4
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "🎤 SUPERTONIC v2 - Stage 2: Text-to-Latent Training Started"
    echo "═══════════════════════════════════════════════════════════════"
    date
    
    # Find autoencoder checkpoint (required)
    local ae_checkpoint=$(find_autoencoder_checkpoint)
    if [[ -z "$ae_checkpoint" ]]; then
        echo "❌ No autoencoder checkpoint found! Complete Stage 1 first."
        exit 1
    fi
    local ae_step=$(echo "$ae_checkpoint" | sed 's/.*checkpoint_\([0-9]*\).*/\1/')
    echo "✅ Using autoencoder: $ae_checkpoint (step $ae_step)"
    
    check_manifests
    
    while [[ $restart_count -lt $MAX_RESTARTS ]]; do
        # Read training mode from file (persists across restarts)
        [[ -f ".training_stage2_mode" ]] && TRAINING_MODE=$(cat .training_stage2_mode)
        
        local checkpoint=$(find_stage2_checkpoint)
        local iteration=0
        [[ -n "$checkpoint" ]] && iteration=$(echo "$checkpoint" | sed 's/.*checkpoint_\([0-9]*\).*/\1/')
        
        local run_log="${LOG_DIR}/train_stage2_$(date '+%Y%m%d_%H%M%S').log"
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "Run #$((restart_count + 1)) - $(date)"
        if [[ -n "$checkpoint" ]]; then
            echo "📂 Resuming from: $checkpoint (step $iteration)"
        else
            echo "🆕 Starting fresh"
        fi
        echo "═══════════════════════════════════════════════════════════════"
        
        # Build command
        local cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=29501"
        cmd+=" train_text_to_latent.py"
        cmd+=" --config $CONFIG_FILE"
        cmd+=" --autoencoder-checkpoint $ae_checkpoint"
        cmd+=" --train-manifest ${MANIFEST_DIR}/train.json"
        cmd+=" --val-manifest ${MANIFEST_DIR}/val.json"
        cmd+=" --data_dir $DATA_DIR"
        cmd+=" --output_dir $OUTPUT_DIR"
        cmd+=" --no-wandb"
        
        # Handle resume modes
        if [[ "$TRAINING_MODE" == "fresh" ]]; then
            echo "🆕 Fresh start (ignoring checkpoints)"
        elif [[ -n "$checkpoint" ]]; then
            cmd+=" --resume $checkpoint"
        fi
        
        echo "🚀 $cmd"
        echo ""
        
        # Run
        set +e
        $cmd 2>&1 | tee -a "$run_log"
        local exit_code=${PIPESTATUS[0]}
        set -e
        
        if [[ $exit_code -eq 0 ]]; then
            echo "✅ Training completed!"
            break
        fi
        
        restart_count=$((restart_count + 1))
        echo "⚠️ Crashed (exit $exit_code). Restart $restart_count/$MAX_RESTARTS in ${RESTART_DELAY}s..."
        sleep $RESTART_DELAY
    done
    
    [[ $restart_count -ge $MAX_RESTARTS ]] && echo "❌ Max restarts reached."
    rm -f "$PID_FILE"
    echo "Ended at $(date)"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

cmd_start() {
    if is_running; then
        echo -e "${YELLOW}⚠️ Already running (PID: $(cat $PID_FILE))${NC}"
        echo "  ./run_train_stage2_nohup.sh --logs  # Watch"
        echo "  ./run_train_stage2_nohup.sh --stop  # Stop"
        exit 1
    fi
    
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  🎤 SUPERTONIC v2 - Stage 2: Text-to-Latent               ║${NC}"
    echo -e "${CYAN}║  ${NUM_GPUS}× GPU (RTX 5090) | nohup mode                      ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    
    show_gpu_status
    
    # Check autoencoder
    local ae_checkpoint=$(find_autoencoder_checkpoint)
    if [[ -z "$ae_checkpoint" ]]; then
        echo -e "\n${RED}❌ No autoencoder checkpoint found!${NC}"
        echo -e "   Complete Stage 1 training first."
        exit 1
    fi
    local ae_step=$(echo "$ae_checkpoint" | sed 's/.*checkpoint_\([0-9]*\).*/\1/')
    echo -e "\n${GREEN}✅ Autoencoder: ${ae_checkpoint} (step ${ae_step})${NC}"
    
    # Check manifests
    if [[ ! -f "${MANIFEST_DIR}/train.json" ]]; then
        echo -e "${RED}❌ Stage 2 manifests not found!${NC}"
        exit 1
    fi
    check_manifests
    
    local checkpoint=$(find_stage2_checkpoint)
    
    if [[ "$TRAINING_MODE" == "fresh" ]]; then
        echo -e "\n🆕 Fresh start requested (ignoring checkpoints)"
    elif [[ -n "$checkpoint" ]]; then
        local iter=$(echo "$checkpoint" | sed 's/.*checkpoint_\([0-9]*\).*/\1/')
        echo -e "\n📂 Will resume from: ${YELLOW}$checkpoint${NC} (step $iter)"
    else
        echo -e "\n🆕 No checkpoint found, starting fresh"
    fi
    
    echo -e "\n🚀 Starting in background..."
    
    # Save training mode to file BEFORE starting background process
    echo "$TRAINING_MODE" > ".training_stage2_mode"
    
    # Start THIS script with --loop flag via nohup
    TRAINING_MODE="$TRAINING_MODE" nohup bash "$0" --loop > "$MAIN_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 1
    
    if is_running; then
        echo -e "\n${GREEN}${BOLD}✅ Started (PID: $(cat $PID_FILE))${NC}"
        echo -e "\n${GREEN}🔒 You can safely close this terminal!${NC}\n"
        echo "Commands:"
        echo "  ./run_train_stage2_nohup.sh --logs    # Watch progress"
        echo "  ./run_train_stage2_nohup.sh --status  # Check status"
        echo "  ./run_train_stage2_nohup.sh --stop    # Stop"
    else
        echo -e "\n${RED}❌ Failed to start. Check $MAIN_LOG${NC}"
        cat "$MAIN_LOG"
    fi
}

cmd_stop() {
    echo "🛑 Stopping..."
    
    if [[ -f "$PID_FILE" ]]; then
        kill "$(cat $PID_FILE)" 2>/dev/null || true
    fi
    pkill -f "torchrun.*train_text_to_latent" 2>/dev/null || true
    pkill -f "train_text_to_latent.py" 2>/dev/null || true
    
    rm -f "$PID_FILE"
    sleep 1
    echo "✅ Stopped"
}

cmd_status() {
    if is_running; then
        echo -e "${GREEN}✅ RUNNING (PID: $(cat $PID_FILE))${NC}"
        show_gpu_status
        echo -e "\nLast 5 lines:"
        tail -5 "$MAIN_LOG" 2>/dev/null || echo "(no log)"
    else
        echo -e "${RED}❌ NOT running${NC}"
        rm -f "$PID_FILE" 2>/dev/null
        local ckpt=$(find_stage2_checkpoint)
        [[ -n "$ckpt" ]] && echo "Last checkpoint: $ckpt"
    fi
}

cmd_logs() {
    if [[ ! -f "$MAIN_LOG" ]]; then
        echo "No log file yet"
        exit 1
    fi
    echo -e "${CYAN}📝 Tailing $MAIN_LOG (Ctrl+C to exit)${NC}\n"
    tail -f "$MAIN_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

case "${1:-}" in
    --loop)
        # Internal: called by nohup
        run_training_loop
        ;;
    --stop)
        cmd_stop
        ;;
    --status)
        cmd_status
        ;;
    --logs|--attach)
        cmd_logs
        ;;
    --fresh)
        TRAINING_MODE="fresh"
        cmd_start
        ;;
    --start|"")
        TRAINING_MODE="auto"
        cmd_start
        ;;
    --help|-h)
        echo -e "${CYAN}🎤 SUPERTONIC v2 - Stage 2: Text-to-Latent Training${NC}"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  (none), --start      Start training (auto-resume from last checkpoint)"
        echo "  --fresh              Start from scratch (ignore existing checkpoints)"
        echo "  --stop               Stop training"
        echo "  --status             Show training status"
        echo "  --logs               Tail training logs"
        ;;
    *)
        echo "Usage: $0 [--start|--fresh|--stop|--status|--logs|--help]"
        ;;
esac
