#!/usr/bin/env bash
# Train both candidates one after the other (2 CPU cores do not share well).
set -e
cd "$(dirname "$0")/.."
python3 scripts/train.py --size 64  --widths 32 64 128 256 --epochs 25 --patience 5
python3 scripts/train.py --size 112 --widths 24 48 96 192 --epochs 25 --patience 5
