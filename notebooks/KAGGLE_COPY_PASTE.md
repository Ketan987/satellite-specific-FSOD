# FSOD Kaggle - Exact Copy-Paste Code

Just copy each section below and paste into separate Kaggle notebook cells!

---

## CELL 1: Configuration & Setup

```python
# ============================================================
# FSOD on Kaggle - Cell 1: Setup
# ============================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 👇 CHANGE THIS TO YOUR KAGGLE DATASET NAME
DATASET_NAME = "fsod-coco-data"

# Training parameters
NUM_EPISODES = 100  # 👈 Change to 1000+ for real training
DEVICE = "cuda"

print("✅ Configuration loaded")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Episodes: {NUM_EPISODES}")
print(f"   Device: {DEVICE}")
```

---

## CELL 2: Clone Repository

```python
# ============================================================
# FSOD on Kaggle - Cell 2: Clone FSOD
# ============================================================

print("📥 Setting up FSOD repository...")
os.chdir("/kaggle/working")

# Clone FSOD
if not os.path.exists("fsod"):
    subprocess.run(
        ["git", "clone", "--depth", "1", 
         "https://github.com/yourusername/fsod.git"],
        check=True
    )

os.chdir("fsod")
print(f"✅ FSOD ready at: {os.getcwd()}")
```

---

## CELL 3: Install Dependencies

```python
# ============================================================
# FSOD on Kaggle - Cell 3: Install
# ============================================================

print("📦 Installing dependencies...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    capture_output=True
)

if result.returncode == 0:
    print("✅ All dependencies installed")
else:
    print("❌ Installation failed")
    print(result.stderr.decode())
```

---

## CELL 4: Copy Dataset

```python
# ============================================================
# FSOD on Kaggle - Cell 4: Prepare Data
# ============================================================

print("📊 Preparing dataset...")

dataset_path = Path(f"/kaggle/input/{DATASET_NAME}")
print(f"   Looking for: {dataset_path}")

if not dataset_path.exists():
    print(f"❌ ERROR: Dataset not found!")
    print(f"   Expected at: {dataset_path}")
    print(f"   Make sure you added the dataset in 'Add Data'!")
    sys.exit(1)

# Create data directory
os.makedirs("data", exist_ok=True)

# Copy COCO JSON files
print("   Copying JSON files...")
for file in ["train_coco.json", "val_coco.json"]:
    src = dataset_path / file
    dst = Path("data") / file
    if src.exists():
        shutil.copy(src, dst)
        print(f"      ✓ {file}")

# Copy image directories
print("   Copying images...")
for dir_name in ["train_images", "val_images"]:
    src = dataset_path / dir_name
    dst = Path("data") / dir_name
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        count = len(list(dst.glob("*")))
        print(f"      ✓ {dir_name}: {count} images")

print("✅ Data ready!")

# Verify
train_imgs = len(list(Path("data/train_images").glob("*")))
val_imgs = len(list(Path("data/val_images").glob("*")))
print(f"\n   Summary:")
print(f"      Training images: {train_imgs}")
print(f"      Validation images: {val_imgs}")
```

---

## CELL 5: Check GPU

```python
# ============================================================
# FSOD on Kaggle - Cell 5: GPU Check
# ============================================================

import torch

print("🎮 GPU Information:")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1e9
    print(f"   Memory: {memory_gb:.1f} GB")
    print(f"   CUDA Capability: {props.major}.{props.minor}")
    print(f"\n✅ GPU ready for training!")
else:
    print("   ⚠️  No GPU detected!")
    print("   Go to top right → Accelerator → Select GPU")
    sys.exit(1)
```

---

## CELL 6: Train Model

```python
# ============================================================
# FSOD on Kaggle - Cell 6: Training
# ============================================================

print("\n" + "="*60)
print("🏋️  STARTING TRAINING")
print("="*60)
print(f"Episodes: {NUM_EPISODES}")
print(f"Device: {DEVICE}")
print(f"Started at: {pd.Timestamp.now()}")
print("="*60 + "\n")

# Run training
cmd = f"python train.py --device {DEVICE} --num_episodes {NUM_EPISODES} --pretrained"
exit_code = os.system(cmd)

if exit_code == 0:
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print(f"Model: checkpoints/best_model.pth")
    print(f"Finished at: {pd.Timestamp.now()}")
else:
    print("\n❌ Training failed!")
    sys.exit(1)
```

---

## CELL 7: Quick Inference Test

```python
# ============================================================
# FSOD on Kaggle - Cell 7: Test Inference
# ============================================================

print("\n🎯 Testing inference...")

# Get sample images
train_imgs = sorted(Path("data/train_images").glob("*.jpg"))
val_imgs = sorted(Path("data/val_images").glob("*.jpg"))

print(f"   Found {len(train_imgs)} training images")
print(f"   Found {len(val_imgs)} validation images")

if len(train_imgs) >= 2 and len(val_imgs) >= 1:
    # Prepare support and query images
    support_imgs = " ".join([str(img) for img in train_imgs[:2]])
    query_img = str(val_imgs[0])
    
    print(f"\n   Support images: {train_imgs[0].name}, {train_imgs[1].name}")
    print(f"   Query image: {val_imgs[0].name}")
    
    # Run inference
    cmd = f"""python inference.py --mode single \
        --model_path checkpoints/best_model.pth \
        --support_img {support_imgs} \
        --query_image {query_img} \
        --output_dir output \
        --device {DEVICE} \
        --score_threshold 0.3"""
    
    os.system(cmd)
    print("\n✅ Inference test complete!")
else:
    print("⚠️  Not enough images for inference test")
```

---

## CELL 8: Batch Inference (Optional)

```python
# ============================================================
# FSOD on Kaggle - Cell 8: Batch Inference
# ============================================================

print("📊 Running batch inference...")

# Use first 3 training images as support set
train_imgs = sorted(Path("data/train_images").glob("*.jpg"))[:3]
support_imgs = " ".join([str(img) for img in train_imgs])

print(f"   Support set: {len(train_imgs)} images")
print(f"   Query dir: data/val_images/")

# Run batch inference
cmd = f"""python inference.py --mode batch \
    --model_path checkpoints/best_model.pth \
    --support_img {support_imgs} \
    --query_dir data/val_images/ \
    --output_csv results.csv \
    --device {DEVICE}"""

os.system(cmd)

# Show results
import pandas as pd
results_csv = Path("results.csv")
if results_csv.exists():
    df = pd.read_csv("results.csv")
    print(f"\n✅ Batch inference complete!")
    print(f"   Total detections: {len(df)}")
    print(f"   Unique images: {df['filename'].nunique()}")
    print(f"\n   Top detections:")
    print(df.nlargest(5, 'similarity_score')[['filename', 'class_name', 'similarity_score']])
```

---

## CELL 9: Download Results

```python
# ============================================================
# FSOD on Kaggle - Cell 9: Summary & Download Info
# ============================================================

print("\n" + "="*60)
print("📥 FILES READY FOR DOWNLOAD")
print("="*60)

# Check what files exist
files_to_download = {
    "Model": "checkpoints/best_model.pth",
    "Inference Output": "output/",
    "CSV Results": "results.csv",
}

print("\n✓ Available files:")
for name, path in files_to_download.items():
    p = Path(path)
    if p.exists():
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*")) / 1e6
            count = len(list(p.glob("*")))
            print(f"   {name}: {path} ({count} files, {size:.1f} MB)")
        else:
            size = p.stat().st_size / 1e6
            print(f"   {name}: {path} ({size:.1f} MB)")

print("\n📝 How to download:")
print("   1. Click 'Output' tab (right side)")
print("   2. Download the 'fsod' folder")
print("   3. Extract and use locally")

print("\n🎉 Done!")
```

---

## 🚀 Full Workflow

```
CELL 1: ✅ Set config
   ↓
CELL 2: ✅ Clone repo
   ↓
CELL 3: ✅ Install deps
   ↓
CELL 4: ✅ Copy dataset
   ↓
CELL 5: ✅ Check GPU
   ↓
CELL 6: 🏋️  TRAIN (takes 10-30 min for 100 episodes)
   ↓
CELL 7: 🎯 Test inference
   ↓
CELL 8: 📊 Batch inference
   ↓
CELL 9: 📥 Download results
```

---

## ⚡ For Quick Test (5 minutes)

Use this in Kaggle:

```python
import os, subprocess, shutil
from pathlib import Path

os.chdir("/kaggle/working")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yourusername/fsod.git"], check=True)
os.chdir("fsod")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

dataset = Path("/kaggle/input/fsod-coco-data")
os.makedirs("data", exist_ok=True)
shutil.copy(dataset / "train_coco.json", "data/")
shutil.copy(dataset / "val_coco.json", "data/")
shutil.copytree(dataset / "train_images", "data/train_images", dirs_exist_ok=True)
shutil.copytree(dataset / "val_images", "data/val_images", dirs_exist_ok=True)

os.system("python train.py --device cuda --num_episodes 10 --pretrained")
```

---

## ✅ Checklist Before Running

- [ ] Change `DATASET_NAME` to your actual Kaggle dataset name
- [ ] Dataset added to notebook (Add Data → Search → Add)
- [ ] GPU enabled (top right)
- [ ] At least 5 training and 5 validation images in dataset
- [ ] All JSON files present in dataset

---

## 💾 After Download

Use your model locally:

```bash
python inference.py --mode batch \
  --model_path best_model.pth \
  --support_img support1.jpg support2.jpg \
  --query_dir test_images/ \
  --output_csv results.csv \
  --device cuda  # or cpu
```

---

Happy training! 🎉
