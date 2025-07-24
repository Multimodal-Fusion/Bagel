#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Visual Jigsaw Generation Training Script
# Based on train-vlm-edit-ft.sh but configured for our visual jigsaw generation task

export MAIN_DIR="/home/colligo/project/vlm/FusionBench/src/train/bagel"

# conda init
conda activate bagel

########################################################
# experiment config
########################################################
export DATASET_CONFIG="$MAIN_DIR/data/configs/visual-jigsaw-generation.yaml"
export MODEL_PATH="$MAIN_DIR/models/BAGEL-7B-MoT"
export EXPERIMENT_NAME="bagel-visual-jigsaw-generation-sft-160k-v2"

########################################################
# set the variables
########################################################
export num_nodes=${WORLD_SIZE:-1}
export node_rank=${RANK:-0}
export master_addr=${MASTER_ADDR:-"localhost"}
export master_port=${MASTER_PORT:-"29500"}
export model_path=$MODEL_PATH
export WANDB_ENTITY="genai-x"

# add main directory to python path
export PYTHONPATH="$MAIN_DIR:$PYTHONPATH"

########################################################
# print the variables
########################################################
echo "================================================"
echo "🧩 Visual Jigsaw Generation Training"
echo "================================================"
echo "Dataset: Visual Jigsaw Generation (160k samples)"
echo "Task: Puzzle → Original Image Reconstruction"
echo "Model: BAGEL-7B-MoT"
echo "================================================"
echo "num_nodes: $num_nodes"
echo "node_rank: $node_rank" 
echo "master_addr: $master_addr"
echo "master_port: $master_port"
echo "model_path: $model_path"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "DATASET_CONFIG: $DATASET_CONFIG"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "================================================"

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
  --visual_gen True \
  --visual_und False \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 8192 \
  --max_num_tokens 8592 \
  --max_num_tokens_per_sample 8192 \
  --wandb_project "bagel" \
  --wandb_name $EXPERIMENT_NAME \
  --checkpoint_dir "results/$EXPERIMENT_NAME/checkpoints" \
  --results_dir "results/$EXPERIMENT_NAME" \
  --wandb_runid "0" \
  --text_cond_dropout_prob 0.0 \
  --vit_cond_dropout_prob 0.0 \
  --save_every 500

echo "🎉 Training completed!"