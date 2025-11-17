# FSOD Kaggle Notebooks

Complete guides and notebooks for training FSOD on Kaggle GPU.

## 📓 Files in This Folder

### 1. **FSOD_Kaggle_Training.ipynb** ⭐ START HERE
Main Jupyter notebook with step-by-step training and inference.

**What it does:**
1. ✅ Configure dataset and training parameters
2. ✅ Clone FSOD repository
3. ✅ Install dependencies
4. ✅ Prepare COCO dataset
5. ✅ Verify GPU
6. ✅ Train model on GPU
7. ✅ Test inference on single image
8. ✅ Batch inference on all validation images
9. ✅ Show results and download info

**How to use:**
1. Create new Kaggle notebook
2. Copy-paste cells from this notebook
3. Update `DATASET_NAME` to your Kaggle dataset
4. Run cells in order
5. Download results from Output tab

**Time:** 10-30 minutes for 100 episodes

---

### 2. **KAGGLE_GUIDE.md** 📚 DETAILED GUIDE
Comprehensive guide with 4 different methods to train on Kaggle.

**Includes:**
- Method 1: Interactive Notebook (Recommended)
- Method 2: Quick Training Script
- Method 3: Full Batch Script
- Method 4: Inference Only

**Also covers:**
- Kaggle environment setup
- Dataset structure
- Performance tips
- Troubleshooting

**Best for:** Understanding different approaches

---

### 3. **KAGGLE_QUICK_REFERENCE.md** ⚡ CHEAT SHEET
Quick reference card for everything Kaggle.

**Contains:**
- 30-second quick start
- Configuration parameters
- Timing estimates
- Checklist
- Common commands
- File locations
- Troubleshooting

**Best for:** Quick lookup while working

---

### 4. **KAGGLE_COPY_PASTE.md** 📋 EXACT CODE
Copy-paste ready code for each step.

**Organized by:**
- Cell 1: Configuration
- Cell 2: Clone Repository
- Cell 3: Install Dependencies
- Cell 4: Copy Dataset
- Cell 5: Check GPU
- Cell 6: Train Model
- Cell 7: Test Inference
- Cell 8: Batch Inference
- Cell 9: Download Results

**Best for:** Direct copy-paste into Kaggle notebook

---

### 5. **kaggle_notebook.py** 🐍 PYTHON SCRIPT
Standalone Python script version of the notebook.

**Can be used for:**
- Running as script in Kaggle notebook
- Local automation
- Understanding the workflow in Python

**How to use:**
```python
%run kaggle_notebook.py
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Dataset on Kaggle
- Upload COCO format dataset to Kaggle
- Note the dataset name

### Step 2: Create Kaggle Notebook
- Go to Kaggle.com → New Notebook
- Add your dataset (Add Data → Search → Add)

### Step 3: Copy & Run Code
**Option A - Easy (Copy from Notebook):**
1. Open `FSOD_Kaggle_Training.ipynb`
2. Copy each cell to Kaggle notebook
3. Run in order

**Option B - Faster (Copy from Markdown):**
1. Open `KAGGLE_COPY_PASTE.md`
2. Copy entire cells
3. Paste into Kaggle notebook
4. Run in order

**Option C - Auto (Run Script):**
1. Copy content of `kaggle_notebook.py`
2. Paste in single Kaggle cell
3. Run

---

## ⚙️ Configuration

In the first cell, update:

```python
DATASET_NAME = "fsod-coco-data"  # Your Kaggle dataset name
NUM_EPISODES = 100               # 100 = quick test, 1000+ = real training
DEVICE = "cuda"                  # Always GPU on Kaggle
```

---

## 📊 Expected Results

### After 100 Episodes (~10 minutes)
- ✅ Model trained successfully
- ✅ Inference works
- ✅ Can download checkpoint
- ⚠️ Low accuracy (test only)

### After 1000 Episodes (~2 hours)
- ✅ Better accuracy
- ✅ Production ready
- ✅ Good for real use

---

## 📥 Dataset Structure

Your Kaggle dataset must have this structure:

```
your-dataset/
├── train_coco.json          # COCO annotations for training
├── val_coco.json            # COCO annotations for validation
├── train_images/            # JPEG images for training
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── val_images/              # JPEG images for validation
    ├── img_100.jpg
    ├── img_101.jpg
    └── ...
```

---

## ✅ Pre-Flight Checklist

Before running on Kaggle:

- [ ] Dataset uploaded to Kaggle
- [ ] Dataset added to notebook (Add Data)
- [ ] `DATASET_NAME` updated in code
- [ ] GPU enabled (top right: "Accelerator: GPU")
- [ ] At least 5 images in each folder
- [ ] JSON files present: `train_coco.json`, `val_coco.json`
- [ ] All images are JPEG format

---

## 🎯 Which File to Use?

| Need | Use This |
|------|----------|
| Step-by-step walkthrough | **FSOD_Kaggle_Training.ipynb** |
| Different methods explained | **KAGGLE_GUIDE.md** |
| Quick lookup | **KAGGLE_QUICK_REFERENCE.md** |
| Copy-paste code | **KAGGLE_COPY_PASTE.md** |
| Python script | **kaggle_notebook.py** |

---

## ⏱️ Time Estimates

| Episodes | Training Time | GPU Memory | Quality |
|----------|--------------|-----------|---------|
| 10 | 1-2 min | 2-3 GB | Debug only |
| 100 | 10-15 min | 2-3 GB | Test |
| 500 | 50 min | 2-3 GB | Low |
| 1000 | 100 min | 2-3 GB | Medium |
| 5000 | 8+ hours | 2-3 GB | Good |

⚠️ Kaggle notebooks timeout after 9 hours!

---

## 🔥 Power User Tips

1. **Start with 100 episodes** → Test everything works
2. **Monitor GPU usage** → Right side panel shows memory
3. **Download frequently** → In case of timeout
4. **Use batch inference** → Faster than single images
5. **Split large training** → Train 5000 episodes, download, continue locally

---

## 🐛 Troubleshooting

### "Dataset not found"
- Check dataset name matches Kaggle dataset exactly
- Verify dataset added via "Add Data" button

### "CUDA out of memory"
- Use fewer episodes: `NUM_EPISODES = 50`
- Reduce model size in `config.py`

### "GPU not available"
- Top right: Select "Accelerator: GPU"
- Restart notebook

### "Notebook timeout after 9 hours"
- Download checkpoint after ~8 hours
- Continue training locally
- Or train fewer episodes

### "Image format error"
- Ensure all images are JPEG (.jpg or .jpeg)
- No PNG, BMP, or other formats

---

## 📚 Related Documentation

See main README.md for:
- General FSOD documentation
- Model architecture details
- Inference examples
- Troubleshooting

See documentation/ folder for:
- `INFERENCE_GUIDE.md` - Detailed inference
- `DOCKER_SETUP.md` - Docker usage
- `ARCHITECTURE_VERIFICATION.md` - Technical details

---

## 💾 After Training: Next Steps

### 1. Download Model
- Click "Output" tab
- Download `fsod/checkpoints/best_model.pth`

### 2. Use Locally
```bash
python inference.py --mode batch \
  --model_path best_model.pth \
  --support_img support1.jpg support2.jpg \
  --query_dir test_images/ \
  --output_csv results.csv
```

### 3. Continue Training
- Upload model back to Kaggle
- Use as pretrained checkpoint
- Train more episodes

---

## 🎉 You're Ready!

Pick a guide above and start training on Kaggle GPU! 🚀

---

## Need Help?

1. Check **KAGGLE_QUICK_REFERENCE.md** for quick answers
2. See **KAGGLE_GUIDE.md** for detailed explanations
3. Use **KAGGLE_COPY_PASTE.md** if copy-paste fails
4. Try **kaggle_notebook.py** as alternative approach
