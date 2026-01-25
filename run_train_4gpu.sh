#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  🔊 SUPERTONIC v2 - Multi-GPU Training Script
#═══════════════════════════════════════════════════════════════════════════════
#
#  Features:
#    ✅ Auto-restart on crash from last checkpoint
#    ✅ tmux session management
#    ✅ Beautiful logging with timestamps
#    ✅ GPU monitoring
#    ✅ Crash detection & recovery
#
#  Usage:
#    ./run_train_4gpu.sh              # Start training in tmux
#    ./run_train_4gpu.sh --attach     # Start and attach to tmux
#    ./run_train_4gpu.sh --stop       # Stop training
#    ./run_train_4gpu.sh --status     # Check status
#    ./run_train_4gpu.sh --logs       # View logs
#
#═══════════════════════════════════════════════════════════════════════════════

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
SESSION_NAME="supertonic_train"
CONFIG_FILE="config/22khz_optimal.yaml"
DATA_DIR="data/audio"
OUTPUT_DIR="outputs/autoencoder_4gpu"
LOG_DIR="logs"
NUM_GPUS=4
MAX_RESTARTS=100  # Maximum auto-restarts before giving up
RESTART_DELAY=30  # Seconds to wait before restart

# Colors for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                           ║"
    echo "║   🔊 ${WHITE}SUPERTONIC v2${CYAN} - Ukrainian TTS Training                            ║"
    echo "║                                                                           ║"
    echo "║   ${YELLOW}Paper:${CYAN} arXiv:2503.23108v3                                             ║"
    echo "║   ${YELLOW}Stage:${CYAN} Speech Autoencoder (WaveNeXt Head)                            ║"
    echo "║   ${YELLOW}GPUs:${CYAN}  ${NUM_GPUS}× RTX 5090 (32GB each)                                       ║"
    echo "║                                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log() {
    local level=$1
    local msg=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${CYAN}[$timestamp]${NC} ${GREEN}[INFO]${NC}  $msg" ;;
        "WARN")  echo -e "${CYAN}[$timestamp]${NC} ${YELLOW}[WARN]${NC}  $msg" ;;
        "ERROR") echo -e "${CYAN}[$timestamp]${NC} ${RED}[ERROR]${NC} $msg" ;;
        "START") echo -e "${CYAN}[$timestamp]${NC} ${MAGENTA}[START]${NC} $msg" ;;
        "GPU")   echo -e "${CYAN}[$timestamp]${NC} ${BLUE}[GPU]${NC}   $msg" ;;
    esac
}

find_latest_checkpoint() {
    # Try multiple possible checkpoint locations
    local checkpoint_dirs=(
        "checkpoints/autoencoder"
        "${OUTPUT_DIR}/checkpoints/autoencoder"
        "outputs/autoencoder_4gpu/checkpoints/autoencoder"
    )
    
    local checkpoint_dir=""
    for dir in "${checkpoint_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            checkpoint_dir="$dir"
            break
        fi
    done
    
    if [[ -z "$checkpoint_dir" ]]; then
        echo ""
        return
    fi
    
    # Find the latest checkpoint by iteration number
    local latest=$(ls -1 "$checkpoint_dir"/checkpoint_*.pt 2>/dev/null | \
                   grep -oP 'checkpoint_\K[0-9]+' | \
                   sort -n | \
                   tail -1)
    
    if [[ -n "$latest" ]]; then
        echo "${checkpoint_dir}/checkpoint_${latest}.pt"
    else
        echo ""
    fi
}

get_checkpoint_iteration() {
    local checkpoint=$1
    if [[ -n "$checkpoint" ]]; then
        echo "$checkpoint" | grep -oP 'checkpoint_\K[0-9]+'
    else
        echo "0"
    fi
}

show_gpu_status() {
    log "GPU" "${WHITE}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name mem_used mem_total util temp; do
        local mem_pct=$((100 * mem_used / mem_total))
        local mem_bar=""
        for ((i=0; i<20; i++)); do
            if ((i < mem_pct / 5)); then
                mem_bar+="█"
            else
                mem_bar+="░"
            fi
        done
        echo -e "      ${YELLOW}GPU $idx${NC}: $name | ${GREEN}${mem_bar}${NC} ${mem_used}/${mem_total}MB | ${util}% | ${temp}°C"
    done
}

kill_training() {
    log "WARN" "Killing any existing training processes..."
    
    # Kill by session
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    
    # Kill any orphaned torchrun processes
    pkill -f "torchrun.*train_autoencoder" 2>/dev/null || true
    pkill -f "train_autoencoder.py" 2>/dev/null || true
    
    sleep 2
    log "INFO" "Cleanup complete"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Loop (runs inside tmux)
# ═══════════════════════════════════════════════════════════════════════════════

run_training_loop() {
    local restart_count=0
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Environment setup
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export PYTHONUNBUFFERED=1
    export WANDB_CONSOLE=off  # Let our logger handle it
    
    print_banner
    show_gpu_status
    
    while [[ $restart_count -lt $MAX_RESTARTS ]]; do
        local checkpoint=$(find_latest_checkpoint)
        local iteration=$(get_checkpoint_iteration "$checkpoint")
        local log_file="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"
        
        echo ""
        log "START" "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
        log "START" "${BOLD}Starting Training Run #$((restart_count + 1))${NC}"
        log "START" "${WHITE}═══════════════════════════════════════════════════════════════${NC}"
        
        if [[ -n "$checkpoint" ]]; then
            log "INFO" "📂 Resuming from checkpoint: ${YELLOW}$checkpoint${NC}"
            log "INFO" "📊 Iteration: ${GREEN}${iteration}${NC} / 1,500,000"
        else
            log "INFO" "🆕 Starting fresh training (no checkpoint found)"
        fi
        
        log "INFO" "📝 Log file: ${CYAN}$log_file${NC}"
        echo ""
        
        # Fix CUDA memory fragmentation (critical for long training!)
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        
        # Build command
        local cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500"
        cmd+=" train_autoencoder.py"
        cmd+=" --config $CONFIG_FILE"
        cmd+=" --data_dir $DATA_DIR"
        cmd+=" --output_dir $OUTPUT_DIR"
        
        if [[ -n "$checkpoint" ]]; then
            cmd+=" --resume $checkpoint"
        fi
        
        # Run training with logging
        log "INFO" "🚀 Launching: ${CYAN}$cmd${NC}"
        echo ""
        
        # Execute and capture exit code
        set +e
        $cmd 2>&1 | tee -a "$log_file"
        local exit_code=${PIPESTATUS[0]}
        set -e
        
        echo ""
        
        if [[ $exit_code -eq 0 ]]; then
            log "INFO" "✅ ${GREEN}Training completed successfully!${NC}"
            break
        else
            log "ERROR" "❌ Training crashed with exit code: $exit_code"
            
            restart_count=$((restart_count + 1))
            
            if [[ $restart_count -lt $MAX_RESTARTS ]]; then
                log "WARN" "🔄 Auto-restart in ${RESTART_DELAY}s... (attempt $restart_count/$MAX_RESTARTS)"
                
                # Kill any zombie processes
                pkill -f "train_autoencoder.py" 2>/dev/null || true
                
                # Show GPU status before restart
                show_gpu_status
                
                # Wait before restart
                sleep $RESTART_DELAY
            else
                log "ERROR" "💀 Max restarts ($MAX_RESTARTS) reached. Giving up."
                exit 1
            fi
        fi
    done
    
    log "INFO" "🎉 Training session ended."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Command Line Interface
# ═══════════════════════════════════════════════════════════════════════════════

case "${1:-}" in
    --stop)
        print_banner
        kill_training
        log "INFO" "Training stopped."
        ;;
        
    --status)
        print_banner
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            log "INFO" "✅ Training session is ${GREEN}RUNNING${NC}"
            echo ""
            
            # Show checkpoint info
            checkpoint=$(find_latest_checkpoint)
            iteration=$(get_checkpoint_iteration "$checkpoint")
            if [[ -n "$checkpoint" ]]; then
                local total=1500000
                local pct=$((100 * iteration / total))
                log "INFO" "📊 Progress: ${GREEN}${iteration}${NC} / ${total} (${pct}%)"
            fi
            
            echo ""
            show_gpu_status
        else
            log "WARN" "⚪ Training session is ${YELLOW}NOT RUNNING${NC}"
        fi
        ;;
        
    --logs)
        print_banner
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux attach-session -t "$SESSION_NAME"
        else
            log "WARN" "No active session. Showing latest log file..."
            latest_log=$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1)
            if [[ -n "$latest_log" ]]; then
                tail -f "$latest_log"
            else
                log "ERROR" "No log files found"
            fi
        fi
        ;;
        
    --attach)
        print_banner
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            log "INFO" "Attaching to existing session..."
            tmux attach-session -t "$SESSION_NAME"
        else
            log "INFO" "Starting new training session and attaching..."
            kill_training
            tmux new-session -d -s "$SESSION_NAME" "bash $0 --run-loop"
            sleep 2
            tmux attach-session -t "$SESSION_NAME"
        fi
        ;;
        
    --run-loop)
        # Internal: called by tmux
        cd "$(dirname "$0")"
        run_training_loop
        ;;
        
    ""|--start)
        print_banner
        
        # Check for existing session
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            log "WARN" "Training session already exists!"
            log "INFO" "Use ${CYAN}./run_train_4gpu.sh --attach${NC} to view"
            log "INFO" "Use ${CYAN}./run_train_4gpu.sh --stop${NC} to stop"
            exit 1
        fi
        
        kill_training
        
        log "INFO" "🚀 Starting training in tmux session: ${CYAN}$SESSION_NAME${NC}"
        
        # Start tmux session
        tmux new-session -d -s "$SESSION_NAME" "bash $0 --run-loop"
        
        sleep 2
        
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            log "INFO" "✅ Training started successfully!"
            echo ""
            log "INFO" "📺 ${WHITE}Commands:${NC}"
            echo -e "      ${CYAN}./run_train_4gpu.sh --attach${NC}  - View training output"
            echo -e "      ${CYAN}./run_train_4gpu.sh --status${NC}  - Check progress"
            echo -e "      ${CYAN}./run_train_4gpu.sh --stop${NC}    - Stop training"
            echo -e "      ${CYAN}./run_train_4gpu.sh --logs${NC}    - Follow logs"
            echo ""
            log "INFO" "💡 Tip: Press ${YELLOW}Ctrl+B${NC} then ${YELLOW}D${NC} to detach from tmux"
        else
            log "ERROR" "Failed to start training session"
            exit 1
        fi
        ;;
        
    *)
        echo "Usage: $0 [--start|--stop|--status|--attach|--logs]"
        echo ""
        echo "  --start   Start training in background (default)"
        echo "  --attach  Start and attach to training session"
        echo "  --stop    Stop training"
        echo "  --status  Show training status and progress"
        echo "  --logs    Attach to session or tail logs"
        exit 1
        ;;
esac
