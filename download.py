import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir, patterns=None):
    """Download a model if it doesn't already exist"""
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"✓ Model already exists: {local_dir}")
        return local_dir
    
    print(f"📥 Downloading {repo_id} to {local_dir}...")
    cache_dir = local_dir + "/cache"
    
    if patterns is None:
        patterns = ["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"]
    
    try:
        snapshot_download(
            cache_dir=cache_dir,
            local_dir=local_dir,
            repo_id=repo_id,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=patterns,
        )
        print(f"✅ Successfully downloaded {repo_id}")
        return local_dir
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return None

# Define models to download
models = [
    {
        "repo_id": "ByteDance-Seed/BAGEL-7B-MoT",
        "local_dir": "models/BAGEL-7B-MoT",
        "patterns": ["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"]
    },
    {
        "repo_id": "black-forest-labs/FLUX.1-dev", 
        "local_dir": "models/FLUX.1-dev",
        "patterns": ["vae/*"]
    }
]

if __name__ == "__main__":
    print("🚀 Model Download Script")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    success_count = 0
    total_count = len(models)
    
    for model in models:
        result = download_model(
            repo_id=model["repo_id"],
            local_dir=model["local_dir"], 
            patterns=model["patterns"]
        )
        if result:
            success_count += 1
        print()
    
    print(f"📊 Download Summary: {success_count}/{total_count} models downloaded successfully")