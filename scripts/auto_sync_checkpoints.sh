#!/bin/bash
# =============================================================================
# AUTO SYNC CHECKPOINTS
# =============================================================================
# Моніторить нові чекпоінти і пушить їх на GitHub LFS
# 
# Запуск:
#   chmod +x scripts/auto_sync_checkpoints.sh
#   nohup ./scripts/auto_sync_checkpoints.sh > logs/sync.log 2>&1 &
# =============================================================================

CHECKPOINT_DIR="checkpoints/autoencoder"
LAST_CHECKPOINT=""
KEEP_LAST_N=3  # Скільки останніх чекпоінтів тримати на сервері

echo "🔄 Starting checkpoint sync monitor..."
echo "   Watching: $CHECKPOINT_DIR"
echo "   Keeping last $KEEP_LAST_N checkpoints on server"

while true; do
    # Знайти найновіший чекпоінт
    NEWEST=$(ls -t $CHECKPOINT_DIR/checkpoint_*.pt 2>/dev/null | head -1)
    
    if [ -n "$NEWEST" ] && [ "$NEWEST" != "$LAST_CHECKPOINT" ]; then
        echo ""
        echo "$(date '+%Y-%m-%d %H:%M:%S') - New checkpoint: $NEWEST"
        
        # Push to git (якщо налаштований LFS)
        # git add "$NEWEST"
        # git commit -m "checkpoint: $(basename $NEWEST)"
        # git push
        
        # Або просто логуємо для ручного скачування
        STEP=$(basename "$NEWEST" | grep -oP '\d+')
        echo "   Step: $STEP"
        echo "   Size: $(du -h "$NEWEST" | cut -f1)"
        
        # Видалити старі (залишити тільки KEEP_LAST_N)
        CHECKPOINTS=($(ls -t $CHECKPOINT_DIR/checkpoint_*.pt 2>/dev/null))
        NUM_CHECKPOINTS=${#CHECKPOINTS[@]}
        
        if [ $NUM_CHECKPOINTS -gt $KEEP_LAST_N ]; then
            echo "   Cleaning old checkpoints (keeping last $KEEP_LAST_N)..."
            for ((i=$KEEP_LAST_N; i<$NUM_CHECKPOINTS; i++)); do
                OLD="${CHECKPOINTS[$i]}"
                echo "   Removing: $(basename $OLD)"
                rm -f "$OLD"
            done
        fi
        
        LAST_CHECKPOINT="$NEWEST"
    fi
    
    sleep 60  # Перевіряти кожну хвилину
done
