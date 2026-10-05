#!/usr/bin/env bash
# Version 2 training, one job at a time (two CPU cores do not share well).
#
# Gatekeeper: one run that never sees BCCD (to test it on an unseen lab), then the final one.
# Classifier: two leave-one-lab-out runs estimate accuracy on a lab the model has never seen;
# the final run trains on every lab's training half. All three start from the version 1
# weights (trained on PBC training cells only) and use the standard framing (prepare_canonical.py).
set -e
cd "$(dirname "$0")/.."
python3 scripts/train_gate.py --holdout bccd --name gate_lolo_bccd
python3 scripts/train_gate.py --name gate_final
COMMON="--size 64 --widths 32 64 128 256 --epochs 10 --lr 1e-3 --patience 4 --framing canonical --init models/candidates/cnn_64.keras"
python3 scripts/train.py $COMMON --holdout bccd --name v2_lolo_bccd
python3 scripts/train.py $COMMON --holdout jtsc --name v2_lolo_jtsc
python3 scripts/train.py $COMMON --name v2_final
