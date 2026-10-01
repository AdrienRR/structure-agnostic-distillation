#!/bin/bash
# Stage 1: train the tokenizer (autoencoder + distillation) for one arm.
#   bash scripts/1_train_tokenizer.sh 2d_pool_align [extra dotlist overrides]
# The configs assume 4 nodes x 8 GPUs (global batch 256). Run once per node with
# NNODES / NODE_RANK / MASTER_ADDR set, or override for a single node, e.g.
#   bash scripts/1_train_tokenizer.sh 2d_pool_align lightning.trainer.num_nodes=1 data.params.batch_size=32
# Writes logs/<arm>/checkpoints/epoch=000049.ckpt. Requires IMAGENET_ROOT (and TEXT_EMB_ROOT for text_*).
set -euo pipefail
ARM=${1:?usage: $0 <arm> [overrides]}; shift
torchrun --nproc_per_node=${GPUS_PER_NODE:-8} --nnodes=${NNODES:-1} --node_rank=${NODE_RANK:-0} \
    --master_addr=${MASTER_ADDR:-localhost} --master_port=${MASTER_PORT:-29500} \
    vavae/main.py --base configs/tokenizer/${ARM}.yaml --train --seed ${SEED:-23} "$@"
