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

def prepare_checkpoint(checkpoint_path, base_model_path, output_suffix="_hf", convert_to_bf16=True, merge_missing_weights=True):
    """
    Copy base model files to a new directory and replace with checkpoint weights.
    Optionally merge missing weights from original BAGEL model.
    
    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., results/.../0000500/)
        base_model_path: Path to the base Bagel model directory
        output_suffix: Suffix to add to output directory name
        convert_to_bf16: Whether to convert checkpoint weights to bfloat16
        merge_missing_weights: Whether to merge missing weights from original BAGEL
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
    
    try:
        print(f"Loading checkpoint weights...")
        checkpoint_weights = load_file(str(ema_checkpoint))
        print(f"  Loaded {len(checkpoint_weights):,} weight tensors from checkpoint")
        
        final_weights = checkpoint_weights.copy()
        
        # Merge missing weights if requested
        if merge_missing_weights:
            original_ema = base_model_path / "ema.safetensors"
            if original_ema.exists():
                print(f"Loading original BAGEL weights for merging...")
                original_weights = load_file(str(original_ema))
                print(f"  Loaded {len(original_weights):,} weight tensors from original BAGEL")
                
                # Find ALL missing weights (not just generation-specific)
                missing_weights = {}
                
                for weight_name in original_weights:
                    if weight_name not in checkpoint_weights:
                        missing_weights[weight_name] = original_weights[weight_name]
                
                if missing_weights:
                    print(f"🔧 Merging {len(missing_weights)} missing weights from original BAGEL:")
                    
                    # Group by prefix for better reporting
                    prefixes = {}
                    missing_params = 0
                    for weight_name, tensor in missing_weights.items():
                        prefix = weight_name.split('.')[0]
                        if prefix not in prefixes:
                            prefixes[prefix] = []
                        prefixes[prefix].append((weight_name, tensor.shape, tensor.numel()))
                        missing_params += tensor.numel()
                    
                    for prefix, weights in prefixes.items():
                        prefix_params = sum(w[2] for w in weights)
                        print(f"  🔸 {prefix}.* ({len(weights)} weights, {prefix_params:,} params)")
                        for weight_name, shape, params in weights[:3]:  # Show first 3
                            print(f"    - {weight_name}: {shape}")
                        if len(weights) > 3:
                            print(f"    ... and {len(weights) - 3} more")
                    
                    print(f"  📊 Total missing parameters: {missing_params:,}")
                    
                    # Merge the weights
                    final_weights.update(missing_weights)
                    print(f"  ✅ Merged weights: {len(checkpoint_weights):,} + {len(missing_weights):,} = {len(final_weights):,}")
                else:
                    print(f"  ℹ️  No missing weights found - checkpoint appears complete")
            else:
                print(f"  ⚠️  Original BAGEL ema.safetensors not found at {original_ema}")
                print(f"  📋 Proceeding with checkpoint weights only")
        
        # Convert to bfloat16 if requested
        if convert_to_bf16:
            print(f"Converting weights to bfloat16...")
            converted_weights = {}
            total_params = 0
            converted_params = 0
            
            for name, tensor in final_weights.items():
                total_params += tensor.numel()
                if tensor.dtype == torch.float32:
                    converted_weights[name] = tensor.to(torch.bfloat16)
                    converted_params += tensor.numel()
                else:
                    converted_weights[name] = tensor
            
            final_weights = converted_weights
            print(f"  📏 Converted {converted_params:,}/{total_params:,} parameters to bfloat16")
        
        # Save the final weights
        print(f"Saving enhanced checkpoint to {target_ema}")
        save_file(final_weights, str(target_ema))
        
        # Calculate final statistics
        final_params = sum(tensor.numel() for tensor in final_weights.values())
        final_size = sum(t.numel() * t.element_size() for t in final_weights.values()) / (1024**3)
        
        print(f"  📈 Final checkpoint: {len(final_weights):,} weights, {final_params:,} parameters")
        print(f"  💾 Memory usage: {final_size:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error processing checkpoint: {e}")
        print(f"Falling back to direct copy...")
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
    parser.add_argument('--no_merge_weights', action='store_true',
                       help='Skip merging missing weights from original BAGEL')
    
    args = parser.parse_args()
    
    try:
        output_path = prepare_checkpoint(args.checkpoint_path, args.base_model_path, args.output_suffix, 
                                        not args.no_bf16_conversion, not args.no_merge_weights)
        print(f"\nNow you can evaluate with:")
        print(f"--model_path {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())