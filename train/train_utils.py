# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    # exclude ones with "_hf" in the name
    step_dirs = [d for d in step_dirs if "_hf" not in d]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir


def copy_missing_configs(checkpoint_path, base_model_path="models/BAGEL-7B-MoT"):
    """
    Copy missing config files from base model directory to checkpoint directory.
    This is useful for evaluation scripts that need config files not saved in checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        base_model_path: Path to base model directory with config files
    """
    config_files = ["llm_config.json", "vit_config.json"]
    
    for config_file in config_files:
        checkpoint_config_path = os.path.join(checkpoint_path, config_file)
        base_config_path = os.path.join(base_model_path, config_file)
        
        # If config doesn't exist in checkpoint but exists in base model
        if not os.path.exists(checkpoint_config_path) and os.path.exists(base_config_path):
            try:
                shutil.copy2(base_config_path, checkpoint_config_path)
                print(f"✓ Copied {config_file} from {base_model_path} to {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Failed to copy {config_file}: {e}")
        elif not os.path.exists(checkpoint_config_path):
            print(f"⚠ Config file {config_file} not found in {checkpoint_path} or {base_model_path}")
