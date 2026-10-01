#!/bin/bash
# Stage 3: train the LightningDiT-XL/1 prior on the extracted latents (80k steps, batch 1024).
#   bash scripts/3_train_dit.sh 2d_pool_align
# Writes output/<arm>/checkpoints/<step>.pt every 5k steps.
set -euo pipefail
ARM=${1:?usage: $0 <arm>}
# torchrun (not `accelerate launch`) so a process group exists even with GPUS=1; train.py wraps DDP itself.
ACCELERATE_MIXED_PRECISION=bf16 torchrun --nproc_per_node=${GPUS:-8} --master_port=${MASTER_PORT:-29502} \
    train.py --config configs/dit/${ARM}.yaml
