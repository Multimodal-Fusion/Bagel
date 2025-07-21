#!/usr/bin/env python3
"""
Script to test if a checkpoint can be loaded successfully.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from safetensors.torch import load_file
import traceback

# Add Bagel path to sys.path
BAGEL_PATH = "/home/colligo/project/vlm/Bagel"
sys.path.insert(0, BAGEL_PATH)

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, 
    Bagel, 
    Qwen2Config, 
    Qwen2ForCausalLM, 
    SiglipVisionConfig, 
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

def test_checkpoint_loading(model_path, check_memory=True):
    """
    Test loading a Bagel checkpoint and report on memory usage and dtypes.
    
    Args:
        model_path: Path to the model directory (should contain config files and ema.safetensors)
        check_memory: Whether to check GPU memory usage
    """
    print(f"Testing checkpoint loading from: {model_path}")
    model_path = Path(model_path)
    
    # Check required files
    required_files = ['llm_config.json', 'vit_config.json', 'ae.safetensors', 'ema.safetensors']
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    
    # Check ema.safetensors dtype
    print("\n=== Checking ema.safetensors ===")
    try:
        ema_path = model_path / "ema.safetensors"
        ema_tensors = load_file(str(ema_path))
        
        print(f"Number of tensors: {len(ema_tensors)}")
        dtypes = {}
        total_params = 0
        
        for name, tensor in ema_tensors.items():
            dtype = str(tensor.dtype)
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += tensor.numel()
            total_params += tensor.numel()
        
        print(f"Total parameters: {total_params:,}")
        print("Dtype distribution:")
        for dtype, count in dtypes.items():
            percentage = (count / total_params) * 100
            print(f"  {dtype}: {count:,} ({percentage:.1f}%)")
            
        # Estimate memory usage
        memory_gb = 0
        for dtype, count in dtypes.items():
            if 'float32' in dtype:
                memory_gb += count * 4 / (1024**3)
            elif 'bfloat16' in dtype or 'float16' in dtype:
                memory_gb += count * 2 / (1024**3)
            elif 'int32' in dtype:
                memory_gb += count * 4 / (1024**3)
            else:
                print(f"  Unknown dtype for memory calc: {dtype}")
        
        print(f"Estimated memory usage: {memory_gb:.2f} GB")
        
    except Exception as e:
        print(f"❌ Error checking ema.safetensors: {e}")
        return False
    
    if check_memory:
        print(f"\n=== GPU Memory Check ===")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {total_mem:.1f} GB total memory")
                if memory_gb > total_mem * 0.8:
                    print(f"⚠️  Warning: Checkpoint ({memory_gb:.1f} GB) may not fit on GPU {i} ({total_mem:.1f} GB)")
        else:
            print("No CUDA devices found")
    
    # Try to load the model configuration
    print(f"\n=== Testing Model Configuration ===")
    try:
        # LLM config
        llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        print("✅ LLM config loaded")

        # ViT config
        vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        print("✅ ViT config loaded")

        # VAE loading
        vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
        print("✅ VAE loaded")

        # Bagel config
        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )
        print("✅ Bagel config created")

        # Try tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        print("✅ Tokenizer loaded")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        traceback.print_exc()
        return False
    
    # Try to create model structure (without loading weights)
    print(f"\n=== Testing Model Structure ===")
    try:
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        print("✅ Model structure created successfully")
        
        # Test device mapping
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "20GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(f"✅ Device mapping successful: {len(device_map)} modules mapped")
        
    except Exception as e:
        print(f"❌ Error creating model structure: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n✅ Checkpoint appears to be loadable!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test checkpoint loading')
    parser.add_argument('model_path', type=str,
                       help='Path to model directory (should contain configs and ema.safetensors)')
    parser.add_argument('--no_memory_check', action='store_true',
                       help='Skip GPU memory checks')
    
    args = parser.parse_args()
    
    success = test_checkpoint_loading(args.model_path, not args.no_memory_check)
    
    if success:
        print(f"\n🎉 Test passed! Model should load correctly.")
        return 0
    else:
        print(f"\n💥 Test failed! Check errors above.")
        return 1

if __name__ == "__main__":
    exit(main())