#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  🔊 SUPERTONIC v2 - Multi-GPU Training Script (HiFi-GAN version)
#═══════════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    ./run_train_4gpu.sh                    # Start training (auto-resume)
#    ./run_train_4gpu.sh --fresh            # Start from scratch (ignore checkpoints)
#    ./run_train_4gpu.sh --partial-resume   # Resume with partial weights (after arch change)
#    ./run_train_4gpu.sh --stop             # Stop training
#    ./run_train_4gpu.sh --status           # Check status
#    ./run_train_4gpu.sh --logs             # Tail the log file
#
#  Partial Resume:
#    Use --partial-resume when switching from WaveNeXt to HiFi-GAN decoder.
#    This loads Encoder fully, Decoder partially (ConvNeXt blocks only),
#    and resets iteration to 0 for fine-tuning the new HiFi-GAN head.
#
#  After starting, you can safely close the terminal!
#
#═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE="config/22khz_optimal.yaml"
DATA_DIR="data/audio"
OUTPUT_DIR="outputs/autoencoder_hifigan"
LOG_DIR="logs"
NUM_GPUS=4
MAX_RESTARTS=100
RESTART_DELAY=30

# Training mode (set by CLI flags)
TRAINING_MODE="auto"  # auto | fresh | partial

PID_FILE=".training.pid"
MAIN_LOG="training.log"

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

find_latest_checkpoint() {
    local dirs="checkpoints/autoencoder ${OUTPUT_DIR}/checkpoints/autoencoder"
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

# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop (called with --loop flag)
# ═══════════════════════════════════════════════════════════════════════════════

run_training_loop() {
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
    
    while [[ $restart_count -lt $MAX_RESTARTS ]]; do
        local checkpoint=$(find_latest_checkpoint)
        local iteration=0
        [[ -n "$checkpoint" ]] && iteration=$(echo "$checkpoint" | sed 's/.*checkpoint_\([0-9]*\).*/\1/')
        
        local run_log="${LOG_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"
        
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
        local cmd="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500"
        cmd+=" train_autoencoder.py"
        cmd+=" --config $CONFIG_FILE"
        cmd+=" --data_dir $DATA_DIR"
        cmd+=" --output_dir $OUTPUT_DIR"
        
        # Handle resume modes
        if [[ "$TRAINING_MODE" == "fresh" ]]; then
            echo "🆕 Fresh start (ignoring checkpoints)"
        elif [[ -n "$checkpoint" ]]; then
            cmd+=" --resume $checkpoint"
            if [[ "$TRAINING_MODE" == "partial" ]]; then
                cmd+=" --partial-resume"
                echo "🔧 Partial resume: Loading Encoder fully, Decoder partially (HiFi-GAN head will be random)"
            fi
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
        echo "  ./run_train_4gpu.sh --logs  # Watch"
        echo "  ./run_train_4gpu.sh --stop  # Stop"
        exit 1
    fi
    
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  🔊 SUPERTONIC v2 - Ukrainian TTS Training                ║${NC}"
    echo -e "${CYAN}║  ${NUM_GPUS}× GPU | nohup mode (survives disconnect)            ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    
    show_gpu_status
    
    local checkpoint=$(find_latest_checkpoint)
    
    if [[ "$TRAINING_MODE" == "fresh" ]]; then
        echo -e "\n🆕 Fresh start requested (ignoring checkpoints)"
    elif [[ -n "$checkpoint" ]]; then
        local iter=$(echo "$checkpoint" | sed 's/.*checkpoint_\([0-9]*\).*/\1/')
        if [[ "$TRAINING_MODE" == "partial" ]]; then
            echo -e "\n🔧 ${YELLOW}PARTIAL RESUME${NC} from: $checkpoint (step $iter)"
            echo -e "   Encoder: ✅ Full weights"
            echo -e "   Decoder: ⚠️ Partial (ConvNeXt only, HiFi-GAN head = random)"
            echo -e "   Iteration: Reset to 0"
        else
            echo -e "\n📂 Will resume from: ${YELLOW}$checkpoint${NC} (step $iter)"
        fi
    else
        echo -e "\n🆕 No checkpoint found, starting fresh"
    fi
    
    echo -e "\n🚀 Starting in background..."
    
    # Start THIS script with --loop flag via nohup (pass training mode)
    TRAINING_MODE="$TRAINING_MODE" nohup bash "$0" --loop > "$MAIN_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 1
    
    if is_running; then
        echo -e "\n${GREEN}${BOLD}✅ Started (PID: $(cat $PID_FILE))${NC}"
        echo -e "\n${GREEN}🔒 You can safely close this terminal!${NC}\n"
        echo "Commands:"
        echo "  ./run_train_4gpu.sh --logs    # Watch progress"
        echo "  ./run_train_4gpu.sh --status  # Check status"
        echo "  ./run_train_4gpu.sh --stop    # Stop"
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
    pkill -f "torchrun.*train_autoencoder" 2>/dev/null || true
    pkill -f "train_autoencoder.py" 2>/dev/null || true
    
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
        local ckpt=$(find_latest_checkpoint)
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
        # Start from scratch, ignore checkpoints
        TRAINING_MODE="fresh"
        cmd_start
        ;;
    --partial-resume|--partial)
        # Resume with partial weights (after WaveNeXt → HiFi-GAN change)
        TRAINING_MODE="partial"
        cmd_start
        ;;
    --start|"")
        TRAINING_MODE="auto"
        cmd_start
        ;;
    --help|-h)
        echo -e "${CYAN}🔊 SUPERTONIC v2 - Training Script${NC}"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  (none), --start      Start training (auto-resume from last checkpoint)"
        echo "  --fresh              Start from scratch (ignore existing checkpoints)"
        echo "  --partial-resume     Resume with partial weights (use after arch change)"
        echo "  --stop               Stop training"
        echo "  --status             Show training status"
        echo "  --logs               Tail training logs"
        echo ""
        echo "Partial Resume:"
        echo "  Use --partial-resume when you changed the decoder architecture"
        echo "  (e.g., WaveNeXt → HiFi-GAN). This will:"
        echo "    ✅ Load Encoder weights fully (preserves your training!)"
        echo "    ⚠️ Load Decoder partially (only ConvNeXt blocks match)"
        echo "    🔧 HiFi-GAN head will initialize randomly"
        echo "    🔄 Iteration resets to 0 for fine-tuning"
        ;;
    *)
        echo "Usage: $0 [--start|--fresh|--partial-resume|--stop|--status|--logs|--help]"
        ;;
esac
