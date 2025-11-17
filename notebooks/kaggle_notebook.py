"""
FSOD Simple Kaggle Notebook - Copy & Paste Ready
Just update the dataset name and run!
"""

# =============================================================================
# CELL 1: SETUP IMPORTS & CONFIGURATION
# =============================================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ⚙️ CONFIGURATION - UPDATE THESE
DATASET_NAME = "fsod-coco-data"  # 👈 CHANGE TO YOUR KAGGLE DATASET NAME
NUM_EPISODES = 100              # 👈 Use 100 for quick test, 1000+ for real training
DEVICE = "cuda"                 # GPU available on Kaggle


# =============================================================================
# CELL 2: CLONE FSOD REPOSITORY
# =============================================================================

print("📥 Cloning FSOD repository...")
os.chdir("/kaggle/working")

if not os.path.exists("fsod"):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/yourusername/fsod.git"],
        check=True
    )
    print("✅ Repository cloned")
else:
    print("✅ Repository already exists")

os.chdir("fsod")
print(f"📂 Working directory: {os.getcwd()}")


# =============================================================================
# CELL 3: INSTALL DEPENDENCIES
# =============================================================================

print("📦 Installing dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True
)
print("✅ Dependencies installed")


# =============================================================================
# CELL 4: PREPARE DATA
# =============================================================================

print("📊 Preparing dataset...")

dataset_path = Path(f"/kaggle/input/{DATASET_NAME}")

if not dataset_path.exists():
    print(f"❌ Dataset not found: {dataset_path}")
    print("   Make sure you've added the dataset in Kaggle notebook!")
    print("   Go to 'Add Data' → Search your dataset → Click 'Add'")
    sys.exit(1)

# Create data directory
os.makedirs("data", exist_ok=True)

# Copy JSON files
for file in ["train_coco.json", "val_coco.json"]:
    src = dataset_path / file
    if src.exists():
        shutil.copy(src, f"data/{file}")
        print(f"✓ Copied {file}")

# Copy image directories
for dir_name in ["train_images", "val_images"]:
    src = dataset_path / dir_name
    if src.exists():
        shutil.copytree(src, f"data/{dir_name}", dirs_exist_ok=True)
        count = len(list(Path(f"data/{dir_name}").glob("*")))
        print(f"✓ Copied {dir_name} ({count} images)")

print("✅ Data ready!")


# =============================================================================
# CELL 5: CHECK GPU
# =============================================================================

import torch

print("🎮 GPU Information:")
print(f"   GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Memory: {memory_gb:.1f} GB")
else:
    print("   ⚠️  GPU not available! Make sure to enable GPU in notebook settings")


# =============================================================================
# CELL 6: TRAIN MODEL
# =============================================================================

print(f"\n{'='*70}")
print(f"🏋️  STARTING TRAINING")
print(f"{'='*70}")
print(f"Episodes: {NUM_EPISODES}")
print(f"Device: {DEVICE}")
print(f"Estimated time: {NUM_EPISODES // 100} - {NUM_EPISODES // 20} minutes")
print(f"{'='*70}\n")

# Train
train_cmd = f"python train.py --device {DEVICE} --num_episodes {NUM_EPISODES} --pretrained"
result = os.system(train_cmd)

if result == 0:
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Model saved to: checkpoints/best_model.pth")
else:
    print(f"\n{'='*70}")
    print("❌ TRAINING FAILED!")
    print(f"{'='*70}")
    sys.exit(1)


# =============================================================================
# CELL 7: TEST INFERENCE
# =============================================================================

print("\n🎯 Testing inference...")

# Get sample images
train_images = sorted(Path("data/train_images").glob("*.jpg"))[:3]
val_images = sorted(Path("data/val_images").glob("*.jpg"))[:2]

if len(train_images) >= 2 and len(val_images) >= 1:
    # Use first 2 training images as support set
    support_imgs = " ".join([str(img) for img in train_images[:2]])
    query_img = str(val_images[0])
    
    print(f"Support images: {len(train_images[:2])}")
    print(f"Query image: {val_images[0].name}")
    
    cmd = f"""python inference.py --mode single \
        --model_path checkpoints/best_model.pth \
        --support_img {support_imgs} \
        --query_image {query_img} \
        --output_dir output \
        --device {DEVICE} \
        --score_threshold 0.3"""
    
    os.system(cmd)
    print("✅ Inference test complete!")
else:
    print("⚠️  Not enough images to test inference")


# =============================================================================
# CELL 8: DOWNLOAD RESULTS
# =============================================================================

print("\n📥 RESULTS READY FOR DOWNLOAD:")
print(f"\n✓ Model checkpoint: /kaggle/working/fsod/checkpoints/best_model.pth")
print(f"✓ Inference output: /kaggle/working/fsod/output/")
print(f"\nDownload these files from the 'Output' section of the notebook!")


# =============================================================================
# OPTIONAL: BATCH INFERENCE ON ALL VALIDATION IMAGES
# =============================================================================

# Uncomment to run batch inference:
"""
print("\n📊 Running batch inference on all validation images...")

support_imgs = " ".join([str(img) for img in train_images[:3]])

cmd = f"""python inference.py --mode batch \
    --model_path checkpoints/best_model.pth \
    --support_img {support_imgs} \
    --query_dir data/val_images/ \
    --output_csv batch_results.csv \
    --device {DEVICE}"""

os.system(cmd)
print("✅ Batch inference complete! Results in batch_results.csv")
"""
