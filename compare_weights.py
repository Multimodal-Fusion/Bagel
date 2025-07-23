#!/usr/bin/env python3
"""
Script to compare weights between VLM checkpoint and original BAGEL model
to identify exactly which weights are missing.
"""

import os
import argparse
from safetensors.torch import load_file
from pathlib import Path

def compare_weights(checkpoint_path, original_path):
    """
    Compare weights between checkpoint and original model to find missing weights.
    
    Args:
        checkpoint_path: Path to VLM checkpoint ema.safetensors
        original_path: Path to original BAGEL ema.safetensors
    """
    print(f"Loading checkpoint weights from: {checkpoint_path}")
    checkpoint_weights = load_file(str(checkpoint_path))
    
    print(f"Loading original BAGEL weights from: {original_path}")
    original_weights = load_file(str(original_path))
    
    print(f"\n📊 Weight Comparison Summary:")
    print(f"  Checkpoint weights: {len(checkpoint_weights):,}")
    print(f"  Original weights: {len(original_weights):,}")
    
    # Find weights in original but missing in checkpoint
    missing_in_checkpoint = set(original_weights.keys()) - set(checkpoint_weights.keys())
    
    # Find weights in checkpoint but not in original (new weights)
    new_in_checkpoint = set(checkpoint_weights.keys()) - set(original_weights.keys())
    
    # Find common weights
    common_weights = set(checkpoint_weights.keys()) & set(original_weights.keys())
    
    print(f"  Common weights: {len(common_weights):,}")
    print(f"  Missing in checkpoint: {len(missing_in_checkpoint):,}")
    print(f"  New in checkpoint: {len(new_in_checkpoint):,}")
    
    # Group missing weights by prefix
    if missing_in_checkpoint:
        print(f"\n❌ Weights Missing in Checkpoint ({len(missing_in_checkpoint)}):")
        
        # Group by prefix for better organization
        prefixes = {}
        for weight_name in sorted(missing_in_checkpoint):
            prefix = weight_name.split('.')[0] if '.' in weight_name else weight_name
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(weight_name)
        
        for prefix, weights in prefixes.items():
            print(f"\n  🔸 {prefix}.* ({len(weights)} weights):")
            for weight in weights[:10]:  # Show first 10
                shape = original_weights[weight].shape
                dtype = original_weights[weight].dtype
                print(f"    - {weight}: {shape} [{dtype}]")
            if len(weights) > 10:
                print(f"    ... and {len(weights) - 10} more")
    
    # Show new weights if any
    if new_in_checkpoint:
        print(f"\n✅ New Weights in Checkpoint ({len(new_in_checkpoint)}):")
        for weight_name in sorted(list(new_in_checkpoint)[:10]):
            shape = checkpoint_weights[weight_name].shape
            dtype = checkpoint_weights[weight_name].dtype
            print(f"  - {weight_name}: {shape} [{dtype}]")
        if len(new_in_checkpoint) > 10:
            print(f"  ... and {len(new_in_checkpoint) - 10} more")
    
    # Calculate parameter counts
    checkpoint_params = sum(tensor.numel() for tensor in checkpoint_weights.values())
    original_params = sum(tensor.numel() for tensor in original_weights.values())
    missing_params = sum(original_weights[key].numel() for key in missing_in_checkpoint)
    
    print(f"\n📈 Parameter Count Analysis:")
    print(f"  Checkpoint parameters: {checkpoint_params:,}")
    print(f"  Original parameters: {original_params:,}")
    print(f"  Missing parameters: {missing_params:,}")
    print(f"  Coverage: {(checkpoint_params / original_params * 100):.1f}%")
    
    return {
        'missing_weights': missing_in_checkpoint,
        'new_weights': new_in_checkpoint,
        'common_weights': common_weights,
        'missing_params': missing_params,
        'checkpoint_params': checkpoint_params,
        'original_params': original_params
    }

def main():
    parser = argparse.ArgumentParser(description='Compare weights between VLM checkpoint and original BAGEL')
    parser.add_argument('--checkpoint_path', type=str,
                       default='/home/colligo/project/vlm/FusionBench/src/train/bagel/results/bagel-vlm-visual-jigsaw-mapping-160k-sft-v1/checkpoints/0002000/ema.safetensors',
                       help='Path to VLM checkpoint ema.safetensors')
    parser.add_argument('--original_path', type=str,
                       default='/home/colligo/project/vlm/FusionBench/src/train/bagel/models/BAGEL-7B-MoT/ema.safetensors',
                       help='Path to original BAGEL ema.safetensors')
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not Path(args.checkpoint_path).exists():
        print(f"❌ Checkpoint path does not exist: {args.checkpoint_path}")
        return 1
    
    if not Path(args.original_path).exists():
        print(f"❌ Original path does not exist: {args.original_path}")
        return 1
    
    try:
        results = compare_weights(args.checkpoint_path, args.original_path)
        
        print(f"\n🎯 Summary:")
        print(f"The VLM checkpoint is missing {len(results['missing_weights'])} weight tensors")
        print(f"({results['missing_params']:,} parameters) from the original BAGEL model.")
        print(f"These are likely the generation-related weights that were excluded")
        print(f"during VLM-only fine-tuning with --visual_gen False.")
        
    except Exception as e:
        print(f"❌ Error comparing weights: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())