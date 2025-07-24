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

# Add imports for distributed checkpoint loading
try:
    from torch.distributed.checkpoint import FileSystemReader, load
    import torch.distributed as dist
    DISTRIBUTED_CHECKPOINT_AVAILABLE = True
except ImportError:
    DISTRIBUTED_CHECKPOINT_AVAILABLE = False
    print("Warning: torch.distributed.checkpoint not available. Sharded checkpoint support disabled.")

def reconstruct_safetensors_from_sharded(sharded_dir_path):
    """
    Convert sharded FSDP checkpoint to single safetensors format.
    
    Args:
        sharded_dir_path: Path to sharded checkpoint directory (e.g., path/to/ema/)
    
    Returns:
        Path to reconstructed safetensors file
    """
    if not DISTRIBUTED_CHECKPOINT_AVAILABLE:
        raise RuntimeError("torch.distributed.checkpoint not available. Cannot load sharded checkpoints.")
    
    sharded_dir_path = Path(sharded_dir_path)
    print(f"🔄 Converting sharded checkpoint from {sharded_dir_path}")
    
    # Initialize process group for single-process loading if not already initialized
    if not dist.is_initialized():
        print("  🔧 Initializing single-process distributed group for checkpoint loading...")
        import tempfile
        import os
        
        # Use file-based init method which is more reliable for single process
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_file = f.name
        
        try:
            dist.init_process_group(
                backend='gloo', 
                init_method=f'file://{init_file}',
                world_size=1, 
                rank=0
            )
            cleanup_process_group = True
        except Exception as e:
            os.unlink(init_file)
            raise e
    else:
        cleanup_process_group = False
        init_file = None
    
    try:
        # Try to load sharded state dict
        state_dict = {}
        
        # The issue is that we need to pre-populate the state_dict with the correct keys
        # Let's try a different approach: create a minimal state dict with expected structure
        
        # Method 1: Try loading with pre-populated keys from metadata
        try:
            print("  🔍 Attempting to parse checkpoint metadata...")
            import pickle
            metadata_path = sharded_dir_path / ".metadata"
            
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Extract parameter names and shapes from metadata
                if hasattr(metadata, 'state_dict_metadata'):
                    param_metadata = metadata.state_dict_metadata
                    print(f"  📋 Found {len(param_metadata)} parameters in metadata")
                    
                    # Create state dict with parameter keys initialized to correctly-shaped tensors
                    for name, tensor_metadata in param_metadata.items():
                        if hasattr(tensor_metadata, 'size'):
                            # Get the tensor shape from metadata
                            tensor_size = tensor_metadata.size
                            # Create a tensor with the correct shape (filled with zeros)
                            state_dict[name] = torch.zeros(tensor_size)
                            print(f"    📦 {name}: {tensor_size}")
                        else:
                            # Fallback for unexpected metadata format
                            print(f"    ⚠️  No size info for {name}, using empty tensor")
                            state_dict[name] = torch.empty(0)
                    
                    print(f"  🔧 Pre-populated state_dict with {len(state_dict)} correctly-shaped tensors")
                    
                    # Now try loading with the pre-populated structure
                    from torch.distributed.checkpoint import load_state_dict
                    storage_reader = FileSystemReader(str(sharded_dir_path))
                    load_state_dict(state_dict, storage_reader)
                    
                    # Filter out empty tensors (these weren't loaded properly)
                    loaded_params = {k: v for k, v in state_dict.items() if v.numel() > 0}
                    state_dict = loaded_params
                    
                    print(f"  📚 Successfully loaded {len(state_dict):,} tensors from sharded checkpoint!")
                else:
                    raise ValueError("Metadata doesn't contain state_dict_metadata")
            else:
                raise FileNotFoundError("No .metadata file found")
                
        except Exception as e1:
            print(f"  ⚠️  Metadata-based loading failed: {e1}")
            # Fallback to empty dict approach
            try:
                state_dict = {}
                from torch.distributed.checkpoint import load_state_dict
                storage_reader = FileSystemReader(str(sharded_dir_path))
                load_state_dict(state_dict, storage_reader)
                print(f"  📚 Loaded {len(state_dict):,} tensors (fallback method)")
            except Exception as e2:
                print(f"  ❌ All loading attempts failed:")
                print(f"     Metadata method: {e1}")
                print(f"     Fallback method: {e2}")
                raise e2
    finally:
        # Cleanup process group if we created it
        if cleanup_process_group:
            dist.destroy_process_group()
            # Clean up temporary init file
            if init_file and os.path.exists(init_file):
                os.unlink(init_file)
    
    # Create temporary safetensors file
    temp_safetensors = sharded_dir_path.parent / f"{sharded_dir_path.name}_reconstructed.safetensors"
    
    # Save as safetensors
    save_file(state_dict, str(temp_safetensors))
    
    # Calculate size for reporting
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    file_size = temp_safetensors.stat().st_size / (1024**3)
    
    print(f"  💾 Saved reconstructed checkpoint: {total_params:,} parameters, {file_size:.2f} GB")
    print(f"  📁 Temporary file: {temp_safetensors}")
    
    return temp_safetensors

def prepare_checkpoint(checkpoint_path, base_model_path, output_suffix="_hf", convert_to_bf16=True, merge_missing_weights=True):
    """
    Copy base model files to a new directory and replace with checkpoint weights.
    Optionally merge missing generation weights from original BAGEL model.
    
    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., results/.../0000500/)
        base_model_path: Path to the base Bagel model directory
        output_suffix: Suffix to add to output directory name
        convert_to_bf16: Whether to convert checkpoint weights to bfloat16
        merge_missing_weights: Whether to merge missing generation weights from original BAGEL
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
    
    # Check for EMA checkpoint in both formats
    ema_safetensors = checkpoint_path / "ema.safetensors"
    ema_sharded_dir = checkpoint_path / "ema"
    
    # Determine which format is available and set ema_checkpoint path
    ema_checkpoint = None
    is_sharded = False
    cleanup_temp_file = False
    
    if ema_safetensors.exists():
        print(f"📁 Found traditional EMA checkpoint: {ema_safetensors}")
        ema_checkpoint = ema_safetensors
    elif ema_sharded_dir.exists() and ema_sharded_dir.is_dir():
        print(f"📂 Found sharded EMA checkpoint directory: {ema_sharded_dir}")
        if DISTRIBUTED_CHECKPOINT_AVAILABLE:
            # Convert sharded to safetensors
            ema_checkpoint = reconstruct_safetensors_from_sharded(ema_sharded_dir)
            is_sharded = True
            cleanup_temp_file = True
        else:
            raise RuntimeError(f"Sharded checkpoint found at {ema_sharded_dir}, but torch.distributed.checkpoint is not available")
    else:
        raise FileNotFoundError(f"EMA checkpoint not found. Looked for:\n  - {ema_safetensors}\n  - {ema_sharded_dir}")
    
    print(f"✅ Using EMA checkpoint: {ema_checkpoint}")
    
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
                
                # Find missing weights - merge ALL missing weights from original model
                missing_weights = {}
                
                for weight_name in original_weights:
                    if weight_name not in checkpoint_weights:
                        # Merge ALL missing weights, no filtering needed
                        missing_weights[weight_name] = original_weights[weight_name]
                
                if missing_weights:
                    print(f"🔧 Merging {len(missing_weights)} missing critical model weights:")
                    
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
                    print(f"  ℹ️  No missing generation weights found - checkpoint appears complete")
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
        
        # Remove redundant model.safetensors since we only need the trained ema.safetensors
        model_file = output_path / "model.safetensors"
        if model_file.exists():
            model_file.unlink()
            print(f"🧹 Removed redundant model.safetensors (keeping trained ema.safetensors)")
        
        # Calculate final statistics
        final_params = sum(tensor.numel() for tensor in final_weights.values())
        final_size = sum(t.numel() * t.element_size() for t in final_weights.values()) / (1024**3)
        
        print(f"  📈 Final checkpoint: {len(final_weights):,} weights, {final_params:,} parameters")
        print(f"  💾 Memory usage: {final_size:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error processing checkpoint: {e}")
        print(f"Falling back to direct copy...")
        shutil.copy2(ema_checkpoint, target_ema)
    
    # Cleanup temporary files if they were created from sharded reconstruction
    if cleanup_temp_file and ema_checkpoint.exists():
        print(f"🧹 Cleaning up temporary file: {ema_checkpoint}")
        ema_checkpoint.unlink()
    
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
                       help='Skip merging missing generation weights from original BAGEL')
    
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