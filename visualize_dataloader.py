#!/usr/bin/env python3
"""
Script to visualize and understand the packed dataset format in Bagel.
This helps understand what's in each batch from the dataloader.
Supports VLM, T2I, and Unified Edit datasets.
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.qwen2 import Qwen2Tokenizer
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def visualize_tokens(tokenizer, token_ids, prefix=""):
    """Decode and display tokens with their IDs."""
    tokens = []
    for tid in token_ids:
        try:
            token = tokenizer.decode([tid])
            tokens.append(f"{tid}:{repr(token)}")
        except:
            tokens.append(f"{tid}:<special>")
    
    print(f"{prefix}Token IDs and text:")
    # Print in chunks of 10 for readability
    for i in range(0, len(tokens), 10):
        chunk = tokens[i:i+10]
        print(f"{prefix}  {' '.join(chunk)}")


def visualize_sequence_structure(batch, tokenizer):
    """Create ASCII visualization of sequence structure."""
    # Build a map of what's at each position
    sequence_map = ['_'] * batch.sequence_length
    
    # Mark text token positions
    text_indexes = batch.packed_text_indexes.tolist() if torch.is_tensor(batch.packed_text_indexes) else batch.packed_text_indexes
    for idx in text_indexes:
        if idx < len(sequence_map):
            sequence_map[idx] = 'T'
    
    # Mark vision token positions
    if hasattr(batch, 'packed_vit_token_indexes'):
        vit_indexes = batch.packed_vit_token_indexes.tolist() if torch.is_tensor(batch.packed_vit_token_indexes) else batch.packed_vit_token_indexes
        for idx in vit_indexes:
            if idx < len(sequence_map):
                sequence_map[idx] = 'V'
    
    # Mark VAE token positions
    if hasattr(batch, 'packed_vae_token_indexes'):
        vae_indexes = batch.packed_vae_token_indexes.tolist() if torch.is_tensor(batch.packed_vae_token_indexes) else batch.packed_vae_token_indexes
        for idx in vae_indexes:
            if idx < len(sequence_map):
                sequence_map[idx] = 'G'  # G for Generation
    
    # Print the sequence map in rows of 100
    print("  Legend: T=Text, V=Vision(VIT), G=Generation(VAE), _=Empty")
    print("  Position markers every 100 tokens:")
    for i in range(0, len(sequence_map), 100):
        chunk = sequence_map[i:i+100]
        print(f"  [{i:4d}-{min(i+99, len(sequence_map)-1):4d}]: {''.join(chunk)}")
    
    # Show sample boundaries
    if len(batch.sample_lens) > 1:
        print("\n  Sample boundaries:")
        cumsum = 0
        for i, slen in enumerate(batch.sample_lens):
            print(f"    Sample {i}: positions {cumsum} to {cumsum + slen - 1}")
            cumsum += slen


def visualize_batch(batch, tokenizer, batch_idx=0):
    """Visualize the contents of a single batch."""
    print("\n" + "="*80)
    print(f"BATCH {batch_idx} VISUALIZATION")
    print("="*80)
    
    # Explain the batch structure
    print("\n📦 BATCH STRUCTURE EXPLANATION:")
    print("  The batch contains multiple samples packed into a single sequence.")
    print("  Each sample follows this pattern:")
    print("    <|vision_start|> [VISION_TOKENS] <|vision_end|> <|im_start|> [TEXT] <|im_end|>")
    print("  Where:")
    print("    - <|vision_start|> and <|vision_end|> mark image boundaries")
    print("    - [VISION_TOKENS] are the patchified image tokens (not stored in packed_text_ids)")
    print("    - <|im_start|> and <|im_end|> wrap text content")
    print("    - Multiple samples are concatenated together")
    
    # Basic batch info
    print(f"\n📊 BATCH STATISTICS:")
    print(f"  Total sequence length: {batch.sequence_length}")
    print(f"  Number of samples packed: {len(batch.sample_lens)}")
    print(f"  Sample lengths: {batch.sample_lens}")
    
    # Create a visual representation of the sequence structure
    print(f"\n--- SEQUENCE STRUCTURE VISUALIZATION ---")
    visualize_sequence_structure(batch, tokenizer)
    
    # Text tokens
    print(f"\n--- PACKED TEXT TOKENS ---")
    print(f"Total text tokens: {len(batch.packed_text_ids)}")
    text_ids = batch.packed_text_ids.tolist() if torch.is_tensor(batch.packed_text_ids) else batch.packed_text_ids
    visualize_tokens(tokenizer, text_ids[:50], "  ")  # Show first 50 tokens
    if len(text_ids) > 50:
        print(f"  ... ({len(text_ids) - 50} more tokens)")
    
    # Text indexes (where text tokens go in the sequence)
    print(f"\n--- TEXT TOKEN POSITIONS ---")
    text_indexes = batch.packed_text_indexes.tolist() if torch.is_tensor(batch.packed_text_indexes) else batch.packed_text_indexes
    print(f"  Text appears at positions: {text_indexes[:20]}...")
    
    # Position IDs (RoPE positions)
    print(f"\n--- POSITION IDS (RoPE) ---")
    pos_ids = batch.packed_position_ids.tolist() if torch.is_tensor(batch.packed_position_ids) else batch.packed_position_ids
    print(f"  First 50 position IDs: {pos_ids[:50]}")
    if len(pos_ids) > 50:
        print(f"  ... ({len(pos_ids) - 50} more)")
    
    # Vision tokens if present
    if hasattr(batch, 'packed_vit_tokens') and len(batch.packed_vit_tokens) > 0:
        print(f"\n--- VISION TOKENS (IMAGE DATA) ---")
        
        # Get the packed vision tokens tensor
        vit_tokens = batch.packed_vit_tokens
        if torch.is_tensor(vit_tokens):
            print(f"  Packed VIT tokens shape: {vit_tokens.shape}")
            print(f"    - Total vision tokens: {vit_tokens.shape[0]}")
            print(f"    - Feature dimension per token: {vit_tokens.shape[1]}")
        
        # Show how many tokens per image
        if hasattr(batch, 'vit_token_seqlens'):
            seqlens = batch.vit_token_seqlens.tolist() if torch.is_tensor(batch.vit_token_seqlens) else batch.vit_token_seqlens
            print(f"\n  Tokens per image: {seqlens}")
            for i, seq_len in enumerate(seqlens[:5]):  # Show first 5
                # Calculate image dimensions from token count
                # seq_len = (height/patch_size) * (width/patch_size)
                # Assuming patch_size = 14
                import math
                num_patches = seq_len
                patches_per_side = int(math.sqrt(num_patches))
                if patches_per_side * patches_per_side == num_patches:
                    img_size = patches_per_side * 14  # patch_size = 14
                    print(f"    Image {i}: {seq_len} tokens = {patches_per_side}x{patches_per_side} patches = ~{img_size}x{img_size} pixels")
                else:
                    # Non-square image
                    print(f"    Image {i}: {seq_len} tokens (non-square image)")
        
        # Show where vision tokens are placed in the sequence
        if hasattr(batch, 'packed_vit_token_indexes'):
            vit_indexes = batch.packed_vit_token_indexes.tolist() if torch.is_tensor(batch.packed_vit_token_indexes) else batch.packed_vit_token_indexes
            print(f"\n  Vision tokens occupy positions: {vit_indexes[:20]}... (showing first 20)")
            
        # Show position embeddings for vision
        if hasattr(batch, 'packed_vit_position_ids'):
            vit_pos_ids = batch.packed_vit_position_ids
            print(f"\n  Vision position IDs shape: {vit_pos_ids.shape if torch.is_tensor(vit_pos_ids) else 'list'}")
            if torch.is_tensor(vit_pos_ids) and len(vit_pos_ids) > 0:
                first_img_pos = vit_pos_ids[0] if vit_pos_ids.dim() > 1 else vit_pos_ids
                print(f"    First image position IDs (first 20): {first_img_pos[:20].tolist() if torch.is_tensor(first_img_pos) else first_img_pos[:20]}...")
    
    # VAE/Generation tokens if present
    if hasattr(batch, 'padded_images') or hasattr(batch, 'padded_latent'):
        print(f"\n--- VAE/GENERATION TOKENS ---")
        
        if hasattr(batch, 'padded_images'):
            print(f"  Padded images shape: {batch.padded_images.shape}")
        
        if hasattr(batch, 'patchified_vae_latent_shapes'):
            print(f"  VAE latent shapes: {batch.patchified_vae_latent_shapes}")
        
        if hasattr(batch, 'packed_vae_token_indexes'):
            vae_indexes = batch.packed_vae_token_indexes.tolist() if torch.is_tensor(batch.packed_vae_token_indexes) else batch.packed_vae_token_indexes
            print(f"  VAE tokens at positions: {vae_indexes[:20]}... (showing first 20)")
        
        if hasattr(batch, 'packed_latent_position_ids'):
            latent_pos = batch.packed_latent_position_ids.tolist() if torch.is_tensor(batch.packed_latent_position_ids) else batch.packed_latent_position_ids
            print(f"  Latent position IDs: {latent_pos[:20]}...")
        
        if hasattr(batch, 'packed_timesteps'):
            timesteps = batch.packed_timesteps.tolist() if torch.is_tensor(batch.packed_timesteps) else batch.packed_timesteps
            # Show unique timesteps (they're usually repeated for all tokens in an image)
            unique_timesteps = list(set(timesteps))[:5]
            print(f"  Unique timesteps (first 5): {unique_timesteps}")
    
    # Loss indexes
    if hasattr(batch, 'ce_loss_indexes') and len(batch.ce_loss_indexes) > 0:
        print(f"\n--- LOSS COMPUTATION ---")
        ce_indexes = batch.ce_loss_indexes[:20] if len(batch.ce_loss_indexes) > 20 else batch.ce_loss_indexes
        print(f"  CE loss computed at positions: {ce_indexes}...")
        print(f"  Total CE loss positions: {len(batch.ce_loss_indexes)}")
    
    if hasattr(batch, 'mse_loss_indexes') and len(batch.mse_loss_indexes) > 0:
        mse_indexes = batch.mse_loss_indexes[:20] if len(batch.mse_loss_indexes) > 20 else batch.mse_loss_indexes
        print(f"  MSE loss computed at positions: {mse_indexes}...")
        print(f"  Total MSE loss positions: {len(batch.mse_loss_indexes)}")
    
    # Attention configuration
    if hasattr(batch, 'split_lens'):
        print(f"\n--- ATTENTION SPLITS ---")
        print(f"  Split lengths: {batch.split_lens[:10]}")
        if hasattr(batch, 'attn_modes'):
            print(f"  Attention modes: {batch.attn_modes[:10]}")
    elif hasattr(batch, 'nested_attention_masks'):
        print(f"\n--- ATTENTION MASKS ---")
        print(f"  Number of attention mask configs: {len(batch.nested_attention_masks)}")
    
    # Decode some actual text to see what's being processed
    print(f"\n--- DECODED TEXT PREVIEW ---")
    try:
        # Decode the first part of the sequence
        text_preview = tokenizer.decode(text_ids[:500])
        print(f"  First 500 tokens decoded: {repr(text_preview[:])}")
    except Exception as e:
        print(f"  Could not decode: {e}")
    
    print("\n" + "="*80)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize Bagel dataset batches")
    parser.add_argument(
        "--dataset-type", 
        choices=["vlm", "t2i", "edit"], 
        default="vlm",
        help="Type of dataset to visualize"
    )
    args = parser.parse_args()
    
    # Configuration based on dataset type
    config_map = {
        "vlm": "data/configs/vlm_simple.yaml",
        "t2i": "data/configs/t2i_simple.yaml",
        "edit": "data/configs/unified_edit_simple.yaml"
    }
    config_path = config_map[args.dataset_type]
    tokenizer_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading {args.dataset_type.upper()} dataset configuration...")
    
    print("Loading configuration and tokenizer...")
    
    # Load config
    config_dict = load_config(config_path)
    
    # Initialize tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer, special_tokens, num_new_tokens = add_special_tokens(tokenizer)
    
    print(f"Special tokens added: {num_new_tokens}")
    for name, token_id in special_tokens.items():
        print(f"  {name}: {token_id}")
    
    # Create data config
    data_config = DataConfig(
        grouped_datasets=config_dict,
        text_cond_dropout_prob=0.0,  # No dropout for visualization
        vit_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=0,
        world_size=1,
        num_workers=1,
        expected_num_tokens=4096,  # Smaller for visualization
        max_num_tokens_per_sample=4096,
        max_num_tokens=5096, # allows more samples in a packed batch
        prefer_buffer_before=1024,
        max_buffer_size=10,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    )
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_wrapper(),
        num_workers=0,  # Single worker for debugging
    )
    
    # Visualize first few batches
    print("\nStarting batch visualization...")
    print("Showing first 5 batches...\n")
    
    try:
        for i, batch in enumerate(dataloader):
            visualize_batch(batch, tokenizer, i)
            import pdb; pdb.set_trace()
                    
            if i >= 4:  # Show 5 batches total
                print("\nShowed 5 batches.")
                break
                
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nVisualization complete!")

    


if __name__ == "__main__":
    main()