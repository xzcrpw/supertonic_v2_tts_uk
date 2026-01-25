#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  🔊 SUPERTONIC v2 - Multi-GPU Training Script (nohup version)
#═══════════════════════════════════════════════════════════════════════════════
#
#  Features:
#    ✅ Survives terminal disconnect (nohup)
#    ✅ Auto-restart on crash from last checkpoint
#    ✅ Beautiful logging with timestamps
#    ✅ GPU monitoring
#
#  Usage:
#    ./run_train_4gpu.sh              # Start training in background
#    ./run_train_4gpu.sh --start      # Same as above
#    ./run_train_4gpu.sh --stop       # Stop training
#    ./run_train_4gpu.sh --status     # Check status
#    ./run_train_4gpu.sh --logs       # Tail the log file
#    ./run_train_4gpu.sh --attach     # Tail logs (alias for --logs)
#
#  After starting, you can safely close the terminal!
#
#═══════════════════════════════════════════════════════════════════════════════

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE="config/22khz_optimal.yaml"
DATA_DIR="data/audio"
OUTPUT_DIR="outputs/autoencoder_4gpu"
LOG_DIR="logs"
NUM_GPUS=4
MAX_RESTARTS=100
RESTART_DELAY=30

# PID and log files
PID_FILE=".training.pid"
MAIN_LOG="training.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'
BOLD='\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

log() {
    local level=$1
    local msg=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}[$timestamp]${NC} ${GREEN}[$level]${NC} $msg"
}

find_latest_checkpoint() {
    local checkpoint_dirs=(
        "checkpoints/autoencoder"
        "${OUTPUT_DIR}/checkpoints/autoencoder"
    )
    
    for dir in "${checkpoint_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local latest=$(ls -1 "$dir"/checkpoint_*.pt 2>/dev/null | \
                           grep -oP 'checkpoint_\K[0-9]+' | \
                           sort -n | tail -1)
            if [[ -n "$latest" ]]; then
                echo "${dir}/checkpoint_${latest}.pt"
                return
            fi
        fi
    done
    echo ""
}

get_checkpoint_iteration() {
    local checkpoint=$1
    if [[ -n "$checkpoint" ]]; then
        echo "$checkpoint" | grep -oP 'checkpoint_\K[0-9]+' || echo "0"
    else
        echo "0"
    fi
}

is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

show_gpu_status() {
    echo -e "\n${WHITE}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx mem_used mem_total util; do
        local mem_pct=$((100 * mem_used / mem_total))
        local bar=""
        for ((i=0; i<20; i++)); do
            if ((i < mem_pct / 5)); then bar+="█"; else bar+="░"; fi
        done
        echo -e "  GPU $idx: [${GREEN}${bar}${NC}] ${mem_used}/${mem_total}MB (${util}%)"
    done
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop (runs in background via nohup)
# ═══════════════════════════════════════════════════════════════════════════════

training_loop() {
    local restart_count=0
    
    mkdir -p "$LOG_DIR"
    
    # Environment
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "🔊 SUPERTONIC v2 - Training Started"
    echo "═══════════════════════════════════════════════════════════════"
    date
    echo ""
    
    while [[ $restart_count -lt $MAX_RESTARTS ]]; do
        local checkpoint=$(find_latest_checkpoint)
        local iteration=$(get_checkpoint_iteration "$checkpoint")
        local log_file="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "Training Run #$((restart_count + 1)) - $(date)"
        echo "═══════════════════════════════════════════════════════════════"
        
        if [[ -n "$checkpoint" ]]; then
            echo "📂 Resuming from: $checkpoint (step $iteration)"
        else
            echo "🆕 Starting fresh training"
        fi
        echo "📝 Run log: $log_file"
        echo ""
        
        # Build command
        local cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500"
        cmd+=" train_autoencoder.py"
        cmd+=" --config $CONFIG_FILE"
        cmd+=" --data_dir $DATA_DIR"
        cmd+=" --output_dir $OUTPUT_DIR"
        
        if [[ -n "$checkpoint" ]]; then
            cmd+=" --resume $checkpoint"
        fi
        
        echo "🚀 Running: $cmd"
        echo ""
        
        # Run training
        set +e
        $cmd 2>&1 | tee -a "$log_file"
        local exit_code=${PIPESTATUS[0]}
        set -e
        
        if [[ $exit_code -eq 0 ]]; then
            echo ""
            echo "✅ Training completed successfully!"
            break
        else
            restart_count=$((restart_count + 1))
            echo ""
            echo "⚠️  Training crashed (exit code: $exit_code)"
            echo "🔄 Restarting in ${RESTART_DELAY}s... (attempt $restart_count/$MAX_RESTARTS)"
            sleep $RESTART_DELAY
        fi
    done
    
    if [[ $restart_count -ge $MAX_RESTARTS ]]; then
        echo "❌ Max restarts reached. Giving up."
    fi
    
    # Cleanup PID file
    rm -f "$PID_FILE"
    echo "Training loop ended at $(date)"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

cmd_start() {
    if is_running; then
        echo -e "${YELLOW}⚠️  Training already running (PID: $(cat $PID_FILE))${NC}"
        echo ""
        echo "Commands:"
        echo "  ./run_train_4gpu.sh --logs    # Watch progress"
        echo "  ./run_train_4gpu.sh --stop    # Stop training"
        exit 1
    fi
    
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║   🔊 SUPERTONIC v2 - Ukrainian TTS Training                               ║"
    echo "║   GPUs: ${NUM_GPUS}× RTX 5090 | nohup mode (survives disconnect)                ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    show_gpu_status
    
    local checkpoint=$(find_latest_checkpoint)
    if [[ -n "$checkpoint" ]]; then
        local iteration=$(get_checkpoint_iteration "$checkpoint")
        log "INFO" "📂 Will resume from: ${YELLOW}$checkpoint${NC} (step $iteration)"
    else
        log "INFO" "🆕 Will start fresh training"
    fi
    
    echo ""
    log "INFO" "🚀 Starting training in background..."
    
    # Start in background with nohup
    nohup bash -c "$(declare -f training_loop find_latest_checkpoint get_checkpoint_iteration); training_loop" \
        > "$MAIN_LOG" 2>&1 &
    
    echo $! > "$PID_FILE"
    
    echo ""
    log "INFO" "✅ Training started (PID: $(cat $PID_FILE))"
    echo ""
    echo -e "${GREEN}${BOLD}🔒 You can now safely close this terminal!${NC}"
    echo ""
    echo "Commands:"
    echo "  ./run_train_4gpu.sh --logs    # Watch progress"
    echo "  ./run_train_4gpu.sh --status  # Check status"
    echo "  ./run_train_4gpu.sh --stop    # Stop training"
    echo "  tail -f $MAIN_LOG             # Same as --logs"
    echo ""
}

cmd_stop() {
    if ! is_running; then
        echo -e "${YELLOW}⚠️  Training is not running${NC}"
        # Try to kill any orphaned processes anyway
        pkill -f "torchrun.*train_autoencoder" 2>/dev/null || true
        pkill -f "train_autoencoder.py" 2>/dev/null || true
        rm -f "$PID_FILE"
        return
    fi
    
    local pid=$(cat "$PID_FILE")
    log "INFO" "🛑 Stopping training (PID: $pid)..."
    
    # Kill the main process and its children
    kill "$pid" 2>/dev/null || true
    pkill -P "$pid" 2>/dev/null || true
    pkill -f "torchrun.*train_autoencoder" 2>/dev/null || true
    pkill -f "train_autoencoder.py" 2>/dev/null || true
    
    sleep 2
    rm -f "$PID_FILE"
    
    log "INFO" "✅ Training stopped"
}

cmd_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo -e "${GREEN}✅ Training is RUNNING (PID: $pid)${NC}"
        echo ""
        show_gpu_status
        echo "Last 5 lines of log:"
        echo "─────────────────────────────────────────────────────────────"
        tail -5 "$MAIN_LOG" 2>/dev/null || echo "(no log yet)"
        echo ""
    else
        echo -e "${RED}❌ Training is NOT running${NC}"
        rm -f "$PID_FILE" 2>/dev/null
        
        # Show last checkpoint
        local checkpoint=$(find_latest_checkpoint)
        if [[ -n "$checkpoint" ]]; then
            local iteration=$(get_checkpoint_iteration "$checkpoint")
            echo ""
            echo "Last checkpoint: $checkpoint (step $iteration)"
        fi
    fi
}

cmd_logs() {
    if [[ ! -f "$MAIN_LOG" ]]; then
        echo -e "${YELLOW}⚠️  No log file found${NC}"
        exit 1
    fi
    
    echo -e "${CYAN}📝 Tailing $MAIN_LOG (Ctrl+C to exit)${NC}"
    echo ""
    tail -f "$MAIN_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

case "${1:-}" in
    --stop)
        cmd_stop
        ;;
    --status)
        cmd_status
        ;;
    --logs|--attach)
        cmd_logs
        ;;
    --start|"")
        cmd_start
        ;;
    *)
        echo "Usage: $0 [--start|--stop|--status|--logs]"
        exit 1
        ;;
esac
