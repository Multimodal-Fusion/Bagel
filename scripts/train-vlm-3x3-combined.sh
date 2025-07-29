#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Visual Jigsaw 3x3 Combined Training Script
# This script trains Bagel on all three 3x3 visual jigsaw datasets

export MAIN_DIR="/home/colligo/project/vlm/FusionBench/src/train/bagel"

conda activate bagel

########################################################
# experiment config
########################################################
export DATASET_CONFIG="$MAIN_DIR/data/configs/visual_jigsaw_3x3_combined.yaml"
export MODEL_PATH="$MAIN_DIR/models/BAGEL-7B-MoT"

export EXPERIMENT_NAME="bagel-vlm-visual-jigsaw-3x3-combined-760k-sft-v1"

########################################################
# set the variables
########################################################
export num_nodes=${WORLD_SIZE:-1}
export node_rank=${RANK:-0}
export master_addr=${MASTER_ADDR:-localhost}
export master_port=${MASTER_PORT:-29500}
export model_path=$MODEL_PATH
export WANDB_ENTITY="genai-x"

# add bagel to python path
export PYTHONPATH="$MAIN_DIR:$PYTHONPATH"

########################################################
# print the variables
########################################################
echo "=========================================="
echo "Visual Jigsaw 3x3 Combined Training"
echo "=========================================="
echo "num_nodes: $num_nodes"
echo "node_rank: $node_rank"
echo "master_addr: $master_addr"
echo "master_port: $master_port"
echo "model_path: $model_path"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "DATASET_CONFIG: $DATASET_CONFIG"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "Dataset size: 760k samples (360k position + 40k mapping + 360k generation)"
echo "=========================================="

########################################################
# run the experiment
########################################################
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file $DATASET_CONFIG \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --visual_gen False \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --wandb_project "bagel" \
  --wandb_name $EXPERIMENT_NAME \
  --checkpoint_dir "results/$EXPERIMENT_NAME/checkpoints" \
  --results_dir "results/$EXPERIMENT_NAME" \
  --wandb_runid "3x3_combined_v1"

echo "Training completed!"