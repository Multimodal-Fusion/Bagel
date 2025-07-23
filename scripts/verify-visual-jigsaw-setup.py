#!/usr/bin/env python3
"""
Verify that the visual jigsaw generation dataset and configuration are ready for training.
"""

import os
import sys
import yaml
import json

def main():
    main_dir = "/home/colligo/project/vlm/FusionBench/src/train/bagel"
    sys.path.append(main_dir)
    
    print("🔍 Verifying Visual Jigsaw Generation Training Setup")
    print("=" * 60)
    
    # Check 1: Dataset configuration file
    config_path = os.path.join(main_dir, "data/configs/visual-jigsaw-generation.yaml")
    print(f"1. Dataset config: {config_path}")
    if os.path.exists(config_path):
        print("   ✅ Config file exists")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"   📊 Dataset: {config['unified_edit']['dataset_names']}")
        print(f"   📦 Chunks to use: {config['unified_edit']['num_used_data']}")
    else:
        print("   ❌ Config file missing")
        return False
    
    # Check 2: Dataset info registration
    print(f"\n2. Dataset registration in dataset_info.py")
    try:
        from data.dataset_info import DATASET_INFO
        
        if 'visual_jigsaw_generation' in DATASET_INFO['unified_edit']:
            dataset_config = DATASET_INFO['unified_edit']['visual_jigsaw_generation']
            print("   ✅ Dataset registered")
            print(f"   📂 Data dir: {dataset_config['data_dir']}")
            print(f"   📊 Total samples: {dataset_config['num_total_samples']:,}")
            print(f"   📁 Chunk files: {dataset_config['num_files']}")
        else:
            print("   ❌ Dataset not registered")
            return False
    except Exception as e:
        print(f"   ❌ Error loading dataset_info: {e}")
        return False
    
    # Check 3: Actual data files
    print(f"\n3. Data files verification")
    data_dir = dataset_config['data_dir']
    if os.path.exists(data_dir):
        print(f"   ✅ Data directory exists: {data_dir}")
        
        # Count parquet files
        parquet_files = [f for f in os.listdir(data_dir) 
                        if f.startswith('chunk_') and f.endswith('.parquet')]
        expected_files = dataset_config['num_files']
        
        if len(parquet_files) == expected_files:
            print(f"   ✅ All parquet files present: {len(parquet_files)}/{expected_files}")
        else:
            print(f"   ❌ Missing parquet files: {len(parquet_files)}/{expected_files}")
            return False
    else:
        print(f"   ❌ Data directory missing: {data_dir}")
        return False
    
    # Check 4: Parquet info file
    print(f"\n4. Parquet info file")
    parquet_info_path = dataset_config['parquet_info_path']
    if os.path.exists(parquet_info_path):
        print("   ✅ Parquet info file exists")
        
        with open(parquet_info_path) as f:
            parquet_info = json.load(f)
        
        total_samples = sum(info['num_rows'] for info in parquet_info.values())
        expected_samples = dataset_config['num_total_samples']
        
        if total_samples == expected_samples:
            print(f"   ✅ Sample count matches: {total_samples:,}")
        else:
            print(f"   ❌ Sample count mismatch: {total_samples:,} vs {expected_samples:,}")
            return False
    else:
        print(f"   ❌ Parquet info file missing: {parquet_info_path}")
        return False
    
    # Check 5: Model path
    print(f"\n5. Model verification")
    model_path = os.path.join(main_dir, "models/BAGEL-7B-MoT")
    if os.path.exists(model_path):
        print(f"   ✅ Model directory exists: {model_path}")
        
        # Check for key model files
        key_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        for key_file in key_files:
            if os.path.exists(os.path.join(model_path, key_file)):
                print(f"   ✅ {key_file} found")
            else:
                print(f"   ⚠️  {key_file} not found (may be in different format)")
    else:
        print(f"   ❌ Model directory missing: {model_path}")
        print("   💡 Make sure the BAGEL-7B-MoT model is downloaded")
        return False
    
    # Check 6: Training script
    print(f"\n6. Training script")
    train_script = os.path.join(main_dir, "scripts/train-visual-jigsaw-generation.sh")
    if os.path.exists(train_script):
        print("   ✅ Training script exists")
        print(f"   🚀 Ready to run: {train_script}")
    else:
        print("   ❌ Training script missing")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All verification checks passed!")
    print("🚀 Ready to start Visual Jigsaw Generation training!")
    print("\nTo start training, run:")
    print(f"cd {main_dir}")
    print("bash scripts/train-visual-jigsaw-generation.sh")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)