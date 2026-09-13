#!/bin/bash
# Wrapper around `train_ensemble_full4_mldg.py` (4-encoder ensemble + MLDG).
# Single GPU, no torchrun/DDP. Mirrors run_train_ensemble_full4.sh: full log
# per attempt, auto-restart from latest checkpoint on crash -- needed since
# this environment recurringly SIGSEGVs mid-training for reasons never
# root-caused (see conversation), and this trainer has no auto-restart of
# its own when launched with a bare `nohup python3 ...`.
set -uo pipefail
cd "$(dirname "$0")"

mkdir -p log
CKPT="checkpoints/ensemble_full4_mldg/generator_latest.pt"
PYTHON="python  (in the "TTS" conda env)"

export PYTHONFAULTHANDLER=1

MAX_RESTARTS=${MAX_RESTARTS:-100000}
GPU=${GPU:-5}
SPEAKERS_PER_BATCH=${SPEAKERS_PER_BATCH:-16}
INIT_FROM=${INIT_FROM:-checkpoints/ensemble_full4/generator_latest.pt}
attempt=0

while true; do
  attempt=$((attempt + 1))
  TS=$(date +%Y%m%d_%H%M%S)
  FULL_LOG="log/train_ensemble_full4_mldg_full_${TS}.log"

  RESUME_ARGS=()
  if [ -f "$CKPT" ]; then
    echo "=== Attempt $attempt: resuming MLDG run from $CKPT — full log: $FULL_LOG ==="
    RESUME_ARGS=(--resume "$CKPT")
  else
    echo "=== Attempt $attempt: starting fresh MLDG run, warm-started from $INIT_FROM — full log: $FULL_LOG ==="
    RESUME_ARGS=(--init_from "$INIT_FROM")
  fi

  "$PYTHON" -u train_ensemble_full4_mldg.py --epochs 500 --gpu "$GPU" --speakers_per_batch "$SPEAKERS_PER_BATCH" \
    --log_file train_log_ensemble_full4_mldg.txt "${RESUME_ARGS[@]}" \
    2>&1 | tee "$FULL_LOG"
  EXIT_CODE=${PIPESTATUS[0]}

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Training finished successfully."
    break
  fi

  if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
    echo "train_ensemble_full4_mldg.py failed (exit $EXIT_CODE) and hit MAX_RESTARTS=$MAX_RESTARTS — giving up. Check $FULL_LOG."
    exit 1
  fi

  echo "process exited with code $EXIT_CODE — restarting from checkpoint in 10s... (attempt $attempt/$MAX_RESTARTS)"
  sleep 10
done
