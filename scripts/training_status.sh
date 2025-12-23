#!/bin/bash
# OssammaNER Training Status Monitor
# Usage: ./scripts/training_status.sh [--watch]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_status() {
    clear

    # Get GPU stats
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null)
    GPU_UTIL=$(echo "$GPU_STATS" | cut -d',' -f1 | tr -d ' %')
    GPU_MEM_USED=$(echo "$GPU_STATS" | cut -d',' -f2 | tr -d ' MiB')
    GPU_MEM_TOTAL=$(echo "$GPU_STATS" | cut -d',' -f3 | tr -d ' MiB')
    GPU_TEMP=$(echo "$GPU_STATS" | cut -d',' -f4 | tr -d ' ')

    # Find training output file
    OUTPUT_FILE=$(ls -t /tmp/claude/-root-Ossamma/tasks/*.output 2>/dev/null | head -1)
    if [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="/root/Ossamma/training.log"
    fi

    # Get latest training stats
    if [ -f "$OUTPUT_FILE" ]; then
        LATEST=$(tail -500 "$OUTPUT_FILE" 2>/dev/null | grep -E "(step:|loss:|grad_norm:|Training:.*ETA|Step.*Loss)" | tail -20)

        STEP=$(echo "$LATEST" | grep "step:" | tail -1 | grep -oP '\d+' | head -1)
        LOSS=$(echo "$LATEST" | grep "loss:" | tail -1 | grep -oP '[\d.]+' | head -1)
        GRAD=$(echo "$LATEST" | grep "grad_norm:" | tail -1 | grep -oP '[\d.]+' | head -1)
        ETA=$(echo "$LATEST" | grep "ETA:" | tail -1 | grep -oP 'ETA: [^(]+' | sed 's/ETA: //')
        SPEED=$(echo "$LATEST" | grep "ETA:" | tail -1 | grep -oP '\( *[\d.]+ *s/it\)' | tr -d '() ')
    fi

    # Default values
    STEP=${STEP:-0}
    LOSS=${LOSS:-0}
    GRAD=${GRAD:-0}
    ETA=${ETA:-"N/A"}
    SPEED=${SPEED:-"N/A"}

    # Calculate progress
    TOTAL_STEPS=50000
    PROGRESS=$((STEP * 100 / TOTAL_STEPS))
    BAR_WIDTH=40
    FILLED=$((PROGRESS * BAR_WIDTH / 100))
    EMPTY=$((BAR_WIDTH - FILLED))

    # Build progress bar
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')
    BAR+=$(printf "%${EMPTY}s" | tr ' ' '░')

    # Determine status colors
    if (( $(echo "$LOSS < 1.5" | bc -l 2>/dev/null || echo 0) )); then
        LOSS_COLOR=$GREEN
    elif (( $(echo "$LOSS < 2.5" | bc -l 2>/dev/null || echo 0) )); then
        LOSS_COLOR=$YELLOW
    else
        LOSS_COLOR=$RED
    fi

    if [ "$GPU_UTIL" -gt 30 ] 2>/dev/null; then
        GPU_COLOR=$GREEN
    elif [ "$GPU_UTIL" -gt 20 ] 2>/dev/null; then
        GPU_COLOR=$YELLOW
    else
        GPU_COLOR=$RED
    fi

    # Check if training is running
    if pgrep -f "julia.*train_ner" > /dev/null; then
        STATUS="${GREEN}RUNNING${NC}"
    else
        STATUS="${RED}STOPPED${NC}"
    fi

    # Display
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}          ${BLUE}OSSAMMANER TRAINING STATUS${NC}     $(date '+%Y-%m-%d %H:%M:%S')      ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  Status: $STATUS                                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Step: ${GREEN}${STEP}${NC} / ${TOTAL_STEPS}  (${PROGRESS}%)                                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  [${GREEN}${BAR}${NC}]  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Loss: ${LOSS_COLOR}${LOSS}${NC}                                                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Grad Norm: ${GRAD}                                               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Speed: ${SPEED}                                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ETA: ${ETA}                                            ${CYAN}║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  GPU Util: ${GPU_COLOR}${GPU_UTIL}%${NC}  |  Memory: ${GPU_MEM_USED}/${GPU_MEM_TOTAL} MiB  |  Temp: ${GPU_TEMP}°C    ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Show recent log entries
    echo -e "${YELLOW}Recent Training Log:${NC}"
    tail -5 "$OUTPUT_FILE" 2>/dev/null | grep -E "(Step|Eval|checkpoint)" | tail -3
}

# Main
if [ "$1" == "--watch" ] || [ "$1" == "-w" ]; then
    echo "Watching training status (Ctrl+C to exit)..."
    while true; do
        show_status
        sleep 30
    done
else
    show_status
fi
