#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CSV_PATH="${1:-"$PROJ_ROOT/data/sample.csv"}"
LOGDIR="$PROJ_ROOT/results"
CKPTDIR="$LOGDIR/spot_ckpts"

mkdir -p "$LOGDIR" "$CKPTDIR"
export PYTHONPATH="$PROJ_ROOT"

# Mac/MPS: allow CPU fallback for StudentT gamma sampling
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m cli.train_deepar_gluonts \
  --csv_path "$CSV_PATH" \
  --seq_len 96 --pred_len 96 \
  --epochs 2 --batch_size 32 \
  --freq h \
  --checkpoint_dir "$CKPTDIR"

echo
echo "— DeepAR last row —"
[ -f "$LOGDIR/deepar_official.csv" ] && tail -n 1 "$LOGDIR/deepar_official.csv" || echo "No deepar_official.csv yet."
