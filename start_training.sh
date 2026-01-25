#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  🔊 SUPERTONIC v2 - Background Training Script
#═══════════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    ./start_training.sh          # Start training in background (survives disconnect!)
#    ./start_training.sh --stop   # Stop training
#    ./start_training.sh --status # Check if running
#    tail -f training.log         # Watch progress
#
#═══════════════════════════════════════════════════════════════════════════════

set -e

CONFIG_FILE="config/22khz_optimal.yaml"
DATA_DIR="data/audio"
OUTPUT_DIR="outputs/autoencoder_4gpu"
NUM_GPUS=4
LOG_FILE="training.log"
PID_FILE="training.pid"

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

find_latest_checkpoint() {
    local ckpt_dir="checkpoints/autoencoder"
    if [[ -d "$ckpt_dir" ]]; then
        ls -t "$ckpt_dir"/checkpoint_*.pt 2>/dev/null | head -1
    fi
}

start_training() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "⚠️  Training already running (PID: $(cat $PID_FILE))"
        echo "   Use: tail -f $LOG_FILE"
        exit 1
    fi
    
    # Find checkpoint
    local checkpoint=$(find_latest_checkpoint)
    local resume_arg=""
    if [[ -n "$checkpoint" ]]; then
        resume_arg="--resume $checkpoint"
        echo "📂 Resuming from: $checkpoint"
    else
        echo "🆕 Starting fresh training"
    fi
    
    echo "🚀 Starting training in background..."
    echo "📝 Log file: $LOG_FILE"
    echo ""
    echo "Commands:"
    echo "  tail -f $LOG_FILE        # Watch progress"
    echo "  ./start_training.sh --stop   # Stop training"
    echo ""
    
    # Start with nohup - survives terminal disconnect!
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
        train_autoencoder.py \
        --config $CONFIG_FILE \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        $resume_arg \
        > "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    echo "✅ Training started (PID: $(cat $PID_FILE))"
    echo ""
    echo "🔒 You can now safely close this terminal!"
    echo "   Training will continue in background."
}

stop_training() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "🛑 Stopping training (PID: $pid)..."
            kill "$pid"
            # Also kill child processes
            pkill -P "$pid" 2>/dev/null || true
            pkill -f "torchrun.*train_autoencoder" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "✅ Training stopped"
        else
            echo "⚠️  Process not running, cleaning up PID file"
            rm -f "$PID_FILE"
        fi
    else
        echo "⚠️  No PID file found"
        # Try to kill any running training anyway
        pkill -f "torchrun.*train_autoencoder" 2>/dev/null || true
    fi
}

check_status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "✅ Training is RUNNING (PID: $(cat $PID_FILE))"
        echo ""
        echo "Last 10 lines of log:"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "No log file yet"
    else
        echo "❌ Training is NOT running"
        if [[ -f "$PID_FILE" ]]; then
            rm -f "$PID_FILE"
        fi
    fi
}

# Parse arguments
case "${1:-}" in
    --stop)
        stop_training
        ;;
    --status)
        check_status
        ;;
    *)
        start_training
        ;;
esac
