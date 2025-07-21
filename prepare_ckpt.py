#!/usr/bin/env python3
"""
Script to prepare a checkpoint for evaluation by copying base model files
and replacing with trained checkpoint weights.
"""

import os
import shutil
import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

def prepare_checkpoint(checkpoint_path, base_model_path, output_suffix="_hf", convert_to_bf16=True):
    """
    Copy base model files to a new directory and replace with checkpoint weights.
    
    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., results/.../0000500/)
        base_model_path: Path to the base Bagel model directory
        output_suffix: Suffix to add to output directory name
        convert_to_bf16: Whether to convert checkpoint weights to bfloat16
    """
    checkpoint_path = Path(checkpoint_path)
    base_model_path = Path(base_model_path)
    
    # Create output directory path
    output_path = checkpoint_path.parent / f"{checkpoint_path.name}{output_suffix}"
    
    print(f"Preparing checkpoint for evaluation:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Base model: {base_model_path}")
    print(f"  Output: {output_path}")
    
    # Verify paths exist
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    
    # Check for required checkpoint files
    ema_checkpoint = checkpoint_path / "ema.safetensors"
    if not ema_checkpoint.exists():
        raise FileNotFoundError(f"EMA checkpoint not found: {ema_checkpoint}")
    
    # Create output directory
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    print(f"Copying base model files to: {output_path}")
    shutil.copytree(base_model_path, output_path)
    
    # Replace ema.safetensors with trained checkpoint
    target_ema = output_path / "ema.safetensors"
    
    if convert_to_bf16:
        print(f"Loading checkpoint weights and converting to bfloat16...")
        try:
            # Load the checkpoint weights
            checkpoint_weights = load_file(str(ema_checkpoint))
            
            # Convert to bfloat16
            converted_weights = {}
            total_params = 0
            converted_params = 0
            
            for name, tensor in checkpoint_weights.items():
                total_params += tensor.numel()
                if tensor.dtype == torch.float32:
                    converted_weights[name] = tensor.to(torch.bfloat16)
                    converted_params += tensor.numel()
                else:
                    converted_weights[name] = tensor
            
            print(f"  Converted {converted_params:,}/{total_params:,} parameters to bfloat16")
            
            # Save the converted weights
            print(f"Saving converted weights to {target_ema}")
            save_file(converted_weights, str(target_ema))
            
            # Calculate memory savings
            original_size = sum(t.numel() * t.element_size() for t in checkpoint_weights.values()) / (1024**3)
            new_size = sum(t.numel() * t.element_size() for t in converted_weights.values()) / (1024**3)
            print(f"  Memory usage: {original_size:.2f} GB -> {new_size:.2f} GB (saved {original_size-new_size:.2f} GB)")
            
        except Exception as e:
            print(f"❌ Error converting checkpoint: {e}")
            print(f"Falling back to direct copy...")
            shutil.copy2(ema_checkpoint, target_ema)
    else:
        print(f"Copying checkpoint weights (no conversion) to {target_ema}")
        shutil.copy2(ema_checkpoint, target_ema)
    
    print(f"✅ Checkpoint prepared successfully at: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Prepare checkpoint for evaluation')
    parser.add_argument('checkpoint_path', type=str,
                       help='Path to checkpoint directory (e.g., results/.../checkpoints/0000500/)')
    parser.add_argument('--base_model_path', type=str, 
                       default='/home/colligo/project/vlm/FusionBench/src/train/bagel/models/BAGEL-7B-MoT',
                       help='Path to base Bagel model directory')
    parser.add_argument('--output_suffix', type=str, default='_hf',
                       help='Suffix for output directory name')
    parser.add_argument('--no_bf16_conversion', action='store_true',
                       help='Skip converting weights to bfloat16')
    
    args = parser.parse_args()
    
    try:
        output_path = prepare_checkpoint(args.checkpoint_path, args.base_model_path, args.output_suffix, 
                                        not args.no_bf16_conversion)
        print(f"\nNow you can evaluate with:")
        print(f"--model_path {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())