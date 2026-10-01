#!/bin/bash
# Stage 2: encode ImageNet train (and its horizontal flip) with the frozen tokenizer.
#   bash scripts/2_extract_latents.sh 2d_pool_align /path/to/imagenet/train
# Writes latents/<arm>/imagenet_train_256/ (safetensors shards + latent statistics).
set -euo pipefail
ARM=${1:?usage: $0 <arm> <imagenet_train_dir>}; DATA=${2:?imagenet train dir}
torchrun --nproc_per_node=${GPUS_PER_NODE:-8} --master_port=${MASTER_PORT:-29501} extract_features.py \
    --config configs/tokenizer_inference/${ARM}.yaml --data_path "$DATA" --output_path latents \
    --data_split imagenet_train --image_size 256 --batch_size 32
