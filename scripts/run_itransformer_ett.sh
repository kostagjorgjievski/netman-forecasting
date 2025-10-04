#!/usr/bin/env bash
set -e
PY=python
CSV=data/ETT/ETTh1.csv

for H in 96 192 336 720; do
  $PY cli/train.py \
    --model itransformer \
    --csv_path $CSV \
    --seq_len 96 --pred_len $H \
    --batch_size 32 --epochs 10 \
    --use_norm --lr 1e-3 --seed 42
done
