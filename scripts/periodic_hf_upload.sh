#!/bin/bash
# Periodic HuggingFace checkpoint uploader
# Uploads checkpoint_latest.jls every hour until training finishes (step 50000)

CHECKPOINT_DIR="/root/Ossamma/checkpoints/ner_110m"
LOG_FILE="/root/Ossamma/hf_upload.log"

echo "$(date): Starting periodic HF upload (every 1 hour)" >> "$LOG_FILE"

while true; do
    # Check current training step from latest checkpoint filename or training log
    CURRENT_STEP=$(ls -t "$CHECKPOINT_DIR"/checkpoint_step_*.jls 2>/dev/null | head -1 | grep -oP 'step_\K[0-9]+')
    
    if [ -z "$CURRENT_STEP" ]; then
        CURRENT_STEP=0
    fi
    
    echo "$(date): Current step: $CURRENT_STEP" >> "$LOG_FILE"
    
    # Upload latest checkpoint
    echo "$(date): Uploading checkpoint_latest.jls to HuggingFace..." >> "$LOG_FILE"
    
    python3 << 'PYEOF'
import os
from huggingface_hub import HfApi

token = os.environ.get('HUGGING_FACE_BIOTZ_TOKEN')
repo_id = 'Biotz/ossamma-ner-checkpoints'
api = HfApi()

checkpoint_dir = "/root/Ossamma/checkpoints/ner_110m"

try:
    api.upload_file(
        path_or_fileobj=f"{checkpoint_dir}/checkpoint_latest.jls",
        path_in_repo="checkpoint_latest.jls",
        repo_id=repo_id,
        token=token,
    )
    print("SUCCESS: Uploaded checkpoint_latest.jls")
except Exception as e:
    print(f"ERROR: {e}")
PYEOF
    
    echo "$(date): Upload complete" >> "$LOG_FILE"
    
    # Check if training is done (step >= 50000)
    if [ "$CURRENT_STEP" -ge 50000 ]; then
        echo "$(date): Training complete (step $CURRENT_STEP >= 50000). Uploading final best checkpoint..." >> "$LOG_FILE"
        
        python3 << 'PYEOF'
import os
from huggingface_hub import HfApi

token = os.environ.get('HUGGING_FACE_BIOTZ_TOKEN')
repo_id = 'Biotz/ossamma-ner-checkpoints'
api = HfApi()

checkpoint_dir = "/root/Ossamma/checkpoints/ner_110m"

try:
    api.upload_file(
        path_or_fileobj=f"{checkpoint_dir}/checkpoint_best.jls",
        path_in_repo="checkpoint_best.jls",
        repo_id=repo_id,
        token=token,
    )
    print("SUCCESS: Uploaded final checkpoint_best.jls")
except Exception as e:
    print(f"ERROR: {e}")
PYEOF
        
        echo "$(date): Final upload complete. Exiting." >> "$LOG_FILE"
        exit 0
    fi
    
    # Sleep for 1 hour
    echo "$(date): Sleeping for 1 hour..." >> "$LOG_FILE"
    sleep 3600
done
