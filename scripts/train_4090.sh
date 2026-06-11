#!/usr/bin/env bash
# One-shot setup + training launch for the 4090 box (or any CUDA Linux machine).
# Run from anywhere inside the cloned repo: bash scripts/train_4090.sh
#
# BEFORE running this, sync the dataset from the Mac over LAN (~58GB):
#   rsync -avh --progress --exclude 'real_legacy_384/' \
#       ~/detect-model/data/ <user>@<4090-host>:~/detect-model/data/
#
# Training: ViT-S/16 full fine-tune from the Community Forensics 224 checkpoint,
# ~1-3h for 8 epochs (CPU image decode is the bottleneck, hence --num-workers 12).
# Outputs land in models/<timestamp>/ (best/ checkpoint + inference_config.json).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "ERROR: no nvidia-smi — is this the right box?"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

free_gb=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
[ "${free_gb:-0}" -ge 70 ] || echo "WARNING: only ${free_gb}GB free disk (want >=70GB incl. dataset)"

[ -d venv ] || python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q -r requirements.txt

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check driver / torch wheel"
print(f"CUDA OK: {torch.cuda.get_device_name(0)}  bf16: {torch.cuda.is_bf16_supported()}")
EOF

# dataset sanity — catches a partial/forgotten rsync before burning GPU time
declare -A min_counts=([real]=20000 [ai]=35000 [val_real]=2200 [val_ai]=4300 [hard_negatives]=170 [hard_positives]=100)
for d in "${!min_counts[@]}"; do
  n=$(find "data/$d" -maxdepth 1 -type f 2>/dev/null | wc -l)
  echo "data/$d: $n files"
  if [ "$n" -lt "${min_counts[$d]}" ]; then
    echo "ERROR: data/$d has $n files (expected >=${min_counts[$d]}) — rsync incomplete?"
    exit 1
  fi
done

mkdir -p logs
log="logs/train_$(date +%Y%m%d_%H%M%S).log"
nohup venv/bin/python scripts/train.py \
    --epochs 8 --batch-size 64 \
    --base OwensLab/commfor-model-224 \
    --num-workers 12 \
    > "$log" 2>&1 &
echo "training started (pid $!) — follow with: tail -f $log"
echo "when done: metrics.json + inference_config.json in models/<timestamp>/"
