#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CSV_PATH="${1:-"$PROJ_ROOT/data/sample.csv"}"
LOGDIR="$PROJ_ROOT/results"
CKPTDIR="$LOGDIR/spot_ckpts"

mkdir -p "$LOGDIR" "$CKPTDIR"
export PYTHONPATH="$PROJ_ROOT"

python -m cli.train \
  --model patchtst \
  --csv_path "$CSV_PATH" \
  --seq_len 96 --pred_len 96 \
  --epochs 2 --batch_size 32 \
  --patch_len 16 --stride 16 \
  --checkpoint_dir "$CKPTDIR" \
  --no_resume

echo
echo "— PatchTST last row —"
[ -f "$LOGDIR/patchtst.csv" ] && tail -n 1 "$LOGDIR/patchtst.csv" || echo "No patchtst.csv yet."
