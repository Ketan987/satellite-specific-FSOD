# FSOD on Kaggle - Complete Guide

## 🎯 Overview

This guide shows you how to train and run inference with FSOD on Kaggle's GPU in 3 simple ways:
1. **Notebook (Easiest)** - Interactive, visual feedback
2. **Script + Shell** - Faster, automated
3. **Custom Dataset** - Use your own Kaggle dataset

---

## 📋 Prerequisites

- Kaggle account (free with GPU limits)
- Dataset uploaded to Kaggle (COCO format)
- This repository cloned or files uploaded

---

## Method 1: Simple Kaggle Notebook (Recommended for Testing)

### Step 1: Create a New Notebook

Go to **Kaggle.com → New Notebook → Python**

### Step 2: Add Your Dataset

1. Click **"Add Data"** button on right side
2. Search for your dataset (or upload)
3. Click **"Add"**

### Step 3: Copy This Code into Notebook

```python
# ============================================================================
# FSOD Training on Kaggle GPU
# ============================================================================

import os
import shutil
import subprocess
import sys

# 1. SETUP KAGGLE ENVIRONMENT
print("📦 Setting up environment...")
os.chdir("/kaggle/working")

# Copy dataset from input directory
dataset_path = "/kaggle/input/your-dataset-name"  # CHANGE THIS
output_path = "/kaggle/working/fsod_data"

# 2. CLONE FSOD REPOSITORY
print("📥 Cloning FSOD repository...")
if not os.path.exists("fsod"):
    subprocess.run(["git", "clone", "https://github.com/yourusername/fsod.git"], check=True)

os.chdir("fsod")

# 3. INSTALL DEPENDENCIES
print("📥 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

# 4. PREPARE DATA
print("📊 Preparing data...")
os.makedirs("data", exist_ok=True)

# Copy COCO JSON files
shutil.copy(f"{dataset_path}/train_coco.json", "data/train_coco.json")
shutil.copy(f"{dataset_path}/val_coco.json", "data/val_coco.json")

# Copy images
shutil.copytree(f"{dataset_path}/train_images", "data/train_images", dirs_exist_ok=True)
shutil.copytree(f"{dataset_path}/val_images", "data/val_images", dirs_exist_ok=True)

print(f"✅ Data ready: {len(os.listdir('data/train_images'))} training images")

# 5. TRAIN MODEL
print("\n🏋️  Starting training on GPU...")
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Train for 100 episodes (quick test)
os.system("python train.py --device cuda --num_episodes 100 --pretrained")

# Or for full training (10000 episodes):
# os.system("python train.py --device cuda --num_episodes 10000 --pretrained")

print("\n✅ Training complete! Model saved to checkpoints/best_model.pth")

# 6. QUICK INFERENCE TEST
print("\n🎯 Running inference test...")
test_images = [f for f in os.listdir("data/val_images") if f.endswith((".jpg", ".jpeg"))][:2]

if test_images:
    support_img = " ".join([f"data/val_images/{img}" for img in test_images[:1]])
    query_img = f"data/val_images/{test_images[1]}"
    
    cmd = f"""python inference.py --mode single \
        --model_path checkpoints/best_model.pth \
        --support_img {support_img} \
        --query_image {query_img} \
        --output_dir output \
        --device cuda"""
    
    os.system(cmd)
    print("✅ Inference complete!")

print("\n📂 Outputs saved to:")
print("   - Checkpoints: checkpoints/best_model.pth")
print("   - Inference: output/")
```

### Step 4: Run the Notebook

1. **Update** `your-dataset-name` to match your Kaggle dataset name
2. **Click "Run"** button
3. **Wait** for training to complete (10-30 minutes for 100 episodes)
4. **Download** outputs from `/kaggle/working/fsod/checkpoints/`

---

## Method 2: Quick Training Script

### For Just Training (No Inference):

Create cell with this code:

```python
# Fast training setup
import os
import sys
import subprocess

os.chdir("/kaggle/working")

# Setup FSOD
if not os.path.exists("fsod"):
    subprocess.run(["git", "clone", "https://github.com/yourusername/fsod.git"], check=True)

os.chdir("fsod")

# Install & prepare
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

# Copy dataset
import shutil
dataset_path = "/kaggle/input/your-dataset"
os.makedirs("data", exist_ok=True)

shutil.copy(f"{dataset_path}/train_coco.json", "data/")
shutil.copy(f"{dataset_path}/val_coco.json", "data/")
shutil.copytree(f"{dataset_path}/train_images", "data/train_images", dirs_exist_ok=True)
shutil.copytree(f"{dataset_path}/val_images", "data/val_images", dirs_exist_ok=True)

# Train
os.system("python train.py --device cuda --num_episodes 10000 --pretrained")

print("✅ Done! Download from /kaggle/working/fsod/checkpoints/")
```

---

## Method 3: Full Batch Script

### Create File: `kaggle_train.py`

```python
#!/usr/bin/env python3
"""
FSOD Training Script for Kaggle
Usage: python kaggle_train.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def setup_kaggle_fsod():
    """Setup FSOD for Kaggle"""
    
    # Configuration
    DATASET_NAME = "your-dataset-name"  # CHANGE THIS
    NUM_EPISODES = 10000
    DEVICE = "cuda"
    
    print("=" * 80)
    print("FSOD Training on Kaggle GPU")
    print("=" * 80)
    
    # Step 1: Navigate to working directory
    os.chdir("/kaggle/working")
    print(f"\n📂 Working directory: {os.getcwd()}")
    
    # Step 2: Clone/Setup FSOD
    if not os.path.exists("fsod"):
        print("\n📥 Cloning FSOD repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", 
             "https://github.com/yourusername/fsod.git"],
            check=True
        )
    
    os.chdir("fsod")
    
    # Step 3: Install dependencies
    print("\n📦 Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True
    )
    
    # Step 4: Prepare data
    print(f"\n📊 Preparing data from Kaggle dataset: {DATASET_NAME}")
    dataset_path = Path(f"/kaggle/input/{DATASET_NAME}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Make sure you've added the dataset in Kaggle notebook!")
        return False
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Copy COCO annotations
    files_to_copy = ["train_coco.json", "val_coco.json"]
    for file in files_to_copy:
        src = dataset_path / file
        dst = Path("data") / file
        if src.exists():
            shutil.copy(src, dst)
            print(f"   ✓ Copied {file}")
    
    # Copy images
    dirs_to_copy = ["train_images", "val_images"]
    for dir_name in dirs_to_copy:
        src = dataset_path / dir_name
        dst = Path("data") / dir_name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            img_count = len(list(dst.glob("*")))
            print(f"   ✓ Copied {dir_name}: {img_count} images")
    
    # Step 5: Verify GPU
    import torch
    print(f"\n🎮 GPU Status:")
    print(f"   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Step 6: Train
    print(f"\n🏋️  Starting training...")
    print(f"   Episodes: {NUM_EPISODES}")
    print(f"   Device: {DEVICE}")
    
    train_cmd = f"python train.py --device {DEVICE} --num_episodes {NUM_EPISODES} --pretrained"
    result = os.system(train_cmd)
    
    if result == 0:
        print("\n✅ Training completed successfully!")
        print(f"   Model saved to: checkpoints/best_model.pth")
    else:
        print("\n❌ Training failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = setup_kaggle_fsod()
    sys.exit(0 if success else 1)
```

### Upload & Run:

```bash
# In Kaggle notebook cell:
%run kaggle_train.py
```

---

## Method 4: Inference Only (Test Pre-trained Model)

```python
import os
import sys
import subprocess

os.chdir("/kaggle/working")

# Setup FSOD
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yourusername/fsod.git"], check=True)
os.chdir("fsod")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

# Copy dataset
import shutil
dataset_path = "/kaggle/input/your-dataset"
os.makedirs("data", exist_ok=True)
shutil.copytree(f"{dataset_path}/train_images", "data/train_images", dirs_exist_ok=True)
shutil.copytree(f"{dataset_path}/val_images", "data/val_images", dirs_exist_ok=True)

# Download pre-trained model (if you have one)
# wget https://your-url/best_model.pth -O checkpoints/best_model.pth

# Run inference on sample
os.system("""python inference.py --mode batch \
    --model_path checkpoints/best_model.pth \
    --support_img data/train_images/img1.jpg data/train_images/img2.jpg \
    --query_dir data/val_images/ \
    --output_csv results.csv \
    --device cuda""")

print("✅ Results saved to results.csv")
```

---

## 🎯 Complete Workflow Summary

### Option A: Interactive Testing (10-15 min)

```
1. Create Kaggle Notebook
2. Add your dataset
3. Paste training code
4. Run → Wait → Download model
```

### Option B: Full Training (2-3 hours)

```
1. Notebook with 10,000 episodes
2. GPU trains while you work
3. Download checkpoint
4. Run inference locally or in new notebook
```

### Option C: Inference Testing (5 min)

```
1. Download trained model from notebook
2. New notebook → upload model
3. Paste inference code
4. Get results.csv
```

---

## 📊 Performance Tips for Kaggle

| Setting | Speed | Quality |
|---------|-------|---------|
| NUM_EPISODES=100 | 10-15 min | Low (test only) |
| NUM_EPISODES=1000 | 1-2 hours | Medium |
| NUM_EPISODES=10000 | 8-12 hours | High (best) |

**Kaggle Tips:**
- Notebook timeout: 9 hours (won't cover 10k full training)
- **Solution:** Train 5000 episodes, download, continue locally
- Use **"Enable GPU"** in notebook settings
- Monitor **"GPU Memory"** in output
- Restart notebook if memory fills up

---

## 🔧 Common Kaggle Dataset Structures

### Standard COCO Format:
```
your-dataset/
├── train_coco.json
├── val_coco.json
├── train_images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── val_images/
    ├── img_100.jpg
    └── ...
```

### Custom Kaggle Structure:
If different, adjust the copy commands:

```python
# Example: If images are in subfolders
src = dataset_path / "images" / "train"
dst = Path("data") / "train_images"
shutil.copytree(src, dst, dirs_exist_ok=True)
```

---

## ✅ Validation Checklist

Before running on Kaggle:

- [ ] Dataset uploaded to Kaggle and added to notebook
- [ ] `train_coco.json` and `val_coco.json` exist
- [ ] Image directories are correct
- [ ] At least 5 images in train_images/
- [ ] GPU enabled in notebook settings
- [ ] Dataset path is correct in code

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: train_coco.json` | Check dataset name in code |
| `CUDA out of memory` | Reduce NUM_EPISODES or use CPU |
| `Image not found` | Verify image paths in notebook |
| `GPU not available` | Enable GPU in notebook settings (top right) |
| `Timeout error after 9 hours` | Train fewer episodes, download, continue locally |

---

## 📝 Example Kaggle Setup

### Assume your Kaggle dataset is named: `fsod-coco-data`

1. **Update dataset name in code:**
   ```python
   DATASET_NAME = "fsod-coco-data"  # Your Kaggle dataset
   ```

2. **Notebook cell 1 - Setup:**
   ```python
   import os
   os.chdir("/kaggle/working")
   ```

3. **Notebook cell 2 - Clone & Install:**
   ```python
   import subprocess
   subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yourusername/fsod.git"], check=True)
   os.chdir("fsod")
   subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
   ```

4. **Notebook cell 3 - Setup Data:**
   ```python
   import shutil
   from pathlib import Path
   
   dataset_path = Path("/kaggle/input/fsod-coco-data")
   os.makedirs("data", exist_ok=True)
   
   shutil.copy(dataset_path / "train_coco.json", "data/")
   shutil.copy(dataset_path / "val_coco.json", "data/")
   shutil.copytree(dataset_path / "train_images", "data/train_images", dirs_exist_ok=True)
   shutil.copytree(dataset_path / "val_images", "data/val_images", dirs_exist_ok=True)
   ```

5. **Notebook cell 4 - Train:**
   ```python
   os.system("python train.py --device cuda --num_episodes 1000 --pretrained")
   ```

6. **Notebook cell 5 - Test Inference:**
   ```python
   os.system("""python inference.py --mode single \
       --model_path checkpoints/best_model.pth \
       --support_img data/train_images/img_001.jpg data/train_images/img_002.jpg \
       --query_image data/val_images/img_100.jpg \
       --output_dir output \
       --device cuda""")
   ```

---

## 🚀 Next Steps

1. **Prepare your dataset** in Kaggle format
2. **Create a Kaggle notebook** with the code above
3. **Add your dataset** to the notebook
4. **Run training** on GPU
5. **Download model** from outputs
6. **Test locally** or share results

---

## 📚 Resources

- Kaggle Datasets: https://kaggle.com/datasets
- Kaggle Notebooks: https://kaggle.com/code
- FSOD Docs: See `documentation/` folder
- COCO Format: http://cocodataset.org/

Happy training on Kaggle! 🎉
