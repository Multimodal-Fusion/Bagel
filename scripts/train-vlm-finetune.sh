# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

export MAIN_DIR="/home/colligo/project/vlm/FusionBench/src/train/bagel"

conda activate bagel
########################################################
# experiment config
########################################################
# export DATASET_CONFIG="./data/configs/example.yaml"
export DATASET_CONFIG="$MAIN_DIR/data/configs/vlm-ft-v1.yaml"
export MODEL_PATH="$MAIN_DIR/models/BAGEL-7B-MoT"


########################################################
# set the variables
########################################################
# set the variables
export num_nodes=$WORLD_SIZE
export node_rank=$RANK
export master_addr=$MASTER_ADDR
export master_port=$MASTER_PORT
export model_path=$MODEL_PATH
# export WANDB_ENTITY='transfusion'
export WANDB_ENTITY="genai-x"
# export WANDB_API_KEY='local-c6ef2052c6e94ee78087778830163f42cbaef274'

# add /home/colligo/project/vlm/FusionBench/src/train/bagel to the python path
export PYTHONPATH="$MAIN_DIR:$PYTHONPATH"

########################################################
# print the variables
########################################################
echo "num_nodes: $num_nodes"
echo "node_rank: $node_rank"
echo "master_addr: $master_addr"
echo "master_port: $master_port"
echo "model_path: $model_path"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "DATASET_CONFIG: $DATASET_CONFIG"
echo "MODEL_PATH: $MODEL_PATH"

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
  --expected_num_tokens 20480 \
  --max_num_tokens 23760 \
  --max_num_tokens_per_sample 20480 \
  --wandb_project "bagel" \
  --wandb_name "bagel-vlm-visualjigsaw-position-1p6M-sft-maxnumtokens20480-v1" \
  --wandb_runid "0"

# torchrun \
#   --nnodes=$num_nodes \
#   --node_rank=$node_rank \
#   --nproc_per_node=8 \
#   --master_addr=$master_addr \
#   --master_port=$master_port \
#   train/pretrain_unified_navit.py \
#   --dataset_config_file $DATASET_CONFIG \
#   --model_path $model_path \
#   --layer_module Qwen2MoTDecoderLayer \
#   --max_latent_size 64 \
#   --resume-from $model_path \
#   --finetune_from_hf True \
#   --auto_resume True \
#   --resume-model-only True \
#   --finetune-from-ema True \
#   --visual_gen False \
#   --log_every 1 \
#   --lr 2e-5 \
#   --num_worker 1 \
#   --expected_num_tokens 10240 \
#   --max_num_tokens 11520 \
#   --max_num_tokens_per_sample 10240 \
#   --wandb_project "bagel" \
#   --wandb_name "bagel-vlm-test-sft-v2" \
#   --wandb_runid "1" \