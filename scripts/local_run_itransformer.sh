#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CSV_PATH="${1:-"$PROJ_ROOT/data/sample.csv"}"
LOGDIR="$PROJ_ROOT/results"
CKPTDIR="$LOGDIR/spot_ckpts"

mkdir -p "$LOGDIR" "$CKPTDIR"
