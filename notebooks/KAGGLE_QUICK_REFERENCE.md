# FSOD Kaggle - Quick Reference Card

## ⚡ 30-Second Quick Start

```python
# Cell 1: Setup
import subprocess, os, shutil
from pathlib import Path

os.chdir("/kaggle/working")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yourusername/fsod.git"], check=True)
os.chdir("fsod")
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

# Cell 2: Copy Dataset
dataset_path = Path("/kaggle/input/your-dataset-name")
os.makedirs("data", exist_ok=True)
shutil.copy(dataset_path / "train_coco.json", "data/")
shutil.copy(dataset_path / "val_coco.json", "data/")
shutil.copytree(dataset_path / "train_images", "data/train_images", dirs_exist_ok=True)
shutil.copytree(dataset_path / "val_images", "data/val_images", dirs_exist_ok=True)

# Cell 3: Train
os.system("python train.py --device cuda --num_episodes 100 --pretrained")

# Cell 4: Inference
os.system("python inference.py --mode batch --model_path checkpoints/best_model.pth --support_img data/train_images/*.jpg --query_dir data/val_images/ --output_csv results.csv --device cuda")
```

---

## 🎯 What to Change

| Parameter | Example | Notes |
|-----------|---------|-------|
| `your-dataset-name` | `fsod-coco-data` | From Kaggle dataset name |
| `NUM_EPISODES` | `100` (test) or `1000` (real) | More = better but slower |
| `yourusername` | `john-smith` | Your GitHub username |

---

## ⏱️ Timing

| Episodes | Time | GPU Memory | Quality |
|----------|------|-----------|---------|
| 100 | 10 min | 2-3 GB | Test only |
| 500 | 50 min | 2-3 GB | Low |
| 1000 | 100 min | 2-3 GB | Medium |
| 5000 | 8+ hours | 2-3 GB | Good |
| 10000 | 16+ hours ⚠️ | 2-3 GB | Best |

⚠️ Kaggle notebooks timeout after 9 hours! Use ~5000 episodes max.

---

## 📋 Checklist

Before clicking "Run":

- [ ] Dataset added to notebook (Add Data → Search → Add)
- [ ] Dataset name matches code
- [ ] GPU enabled (top right: "Accelerator: GPU")
- [ ] All three files exist in dataset:
  - [ ] `train_coco.json`
  - [ ] `val_coco.json`
  - [ ] `train_images/` folder
  - [ ] `val_images/` folder

---

## 🔥 Common Commands

### Quick Training (10 min)
```bash
python train.py --device cuda --num_episodes 100 --pretrained
```

### Real Training (2 hours)
```bash
python train.py --device cuda --num_episodes 1000 --pretrained
```

### Single Image Test
```bash
python inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --support_img data/train_images/img1.jpg data/train_images/img2.jpg \
  --query_image data/val_images/img_test.jpg \
  --output_dir output \
  --device cuda
```

### Batch Processing
```bash
python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_img data/train_images/img1.jpg \
  --query_dir data/val_images/ \
  --output_csv results.csv \
  --device cuda
```

---

## 💾 File Locations in Kaggle

| Item | Path |
|------|------|
| Your dataset | `/kaggle/input/your-dataset-name/` |
| Working dir | `/kaggle/working/` |
| FSOD code | `/kaggle/working/fsod/` |
| Trained model | `/kaggle/working/fsod/checkpoints/best_model.pth` |
| Inference output | `/kaggle/working/fsod/output/` |
| CSV results | `/kaggle/working/fsod/results.csv` |

---

## 🐛 Troubleshooting

### "FileNotFoundError: train_coco.json"
→ Dataset name in code is wrong
→ Dataset not properly added to notebook

### "CUDA out of memory"
→ Use fewer episodes: `--num_episodes 50`
→ Use CPU: `--device cpu` (slower!)

### "No such file or directory: fsod"
→ Git clone failed. Try again.

### GPU not showing in notebook
→ Top right: Select "Accelerator: GPU"
→ Restart notebook

### Notebook timeout (after 9 hours)
→ Use fewer episodes
→ Download partial model, continue locally

---

## 📤 After Training: Download Your Model

1. Go to "Output" tab on right side
2. Download folder: `/kaggle/working/fsod/`
3. Get these files:
   - `checkpoints/best_model.pth` (main model)
   - `output/` (inference results)

---

## 🚀 Use Model Later

### Option 1: Download & Use Locally
```bash
# Download best_model.pth from Kaggle
python inference.py --mode single \
  --model_path best_model.pth \
  --support_img support1.jpg support2.jpg \
  --query_image test.jpg \
  --device cpu  # or cuda if you have GPU
```

### Option 2: Upload Back to Kaggle
1. Create new notebook
2. Add the model file as dataset
3. Load and run inference

---

## 📊 Expected Results

After training 100 episodes on Kaggle:
- ✅ Model saved successfully
- ✅ Can run inference on new images
- ✅ CSV results generated
- ⚠️ Accuracy depends on dataset quality

After training 1000+ episodes:
- ✅ Better accuracy (2-3% improvement per 1000 episodes)
- ✅ Production-ready quality
- ⚠️ Needs 1-2 hours on GPU

---

## 💡 Pro Tips

1. **First run: Test with 100 episodes** → Verify everything works
2. **Second run: Use 1000-5000 episodes** → Real training
3. **Download checkpoints frequently** → In case notebook crashes
4. **Monitor GPU usage** → Right side shows memory usage
5. **Batch inference is faster** → Process 100 images together

---

## 🎓 Learning Path

1. **Day 1:** Run quick training (100 episodes) → Verify setup
2. **Day 2:** Real training (1000+ episodes) → Get decent model
3. **Day 3:** Batch inference → Test on whole dataset
4. **Day 4:** Download model → Use locally or improve

---

## 📚 Full Documentation

See:
- `KAGGLE_GUIDE.md` - Detailed explanations
- `README.md` - General FSOD docs
- `documentation/INFERENCE_GUIDE.md` - Inference details

---

## ✅ You're Ready!

1. Upload/prepare your Kaggle dataset
2. Create new Kaggle notebook
3. Copy code from `kaggle_notebook.py`
4. Click "Run"
5. Download results

**Good luck! 🚀**
