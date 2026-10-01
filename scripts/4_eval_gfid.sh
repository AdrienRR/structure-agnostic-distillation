#!/bin/bash
# Stage 4: unguided gFID (50k samples, 250 Euler steps, ADM evaluator) of the DiT checkpoints.
#   bash scripts/4_eval_gfid.sh 2d_pool_align            # final checkpoint only
#   LAST_N=16 STRIDE=3 bash scripts/4_eval_gfid.sh 2d_pool_align   # a training curve (Figure 2)
# Appends to output/<arm>/fid_curve.json. Requires ADM_EVALUATOR (and ADM_PYTHON), see README.
set -euo pipefail
ARM=${1:?usage: $0 <arm>}
python compute_fid_curve.py --config configs/dit/${ARM}.yaml --num_gpus ${GPUS:-8} \
    --last_n ${LAST_N:-1} --stride ${STRIDE:-1} --cfg_scale 1.0
