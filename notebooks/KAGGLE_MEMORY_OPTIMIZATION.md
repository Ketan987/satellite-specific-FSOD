# Kaggle GPU Memory Optimization Guide

## Problem
**Error:** `torch.cuda.OutOfMemoryError: CUDA out of memory`

This happens when the GPU runs out of memory during training. The Kaggle free GPU (typically P100/T4 with 16GB) isn't enough for 512×512 images with ResNet-50.

---

## ✅ Solutions Implemented

### 1. **Image Size Reduced** (Already Done)
- **Before:** `IMAGE_SIZE = 512` → **After:** `IMAGE_SIZE = 384`
- **Memory saved:** ~35% reduction
- **Impact:** Slightly lower accuracy but training completes

### 2. **CUDA Memory Optimization** (Already Done)
Added to `train.py` and `inference.py`:
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```
This prevents GPU memory fragmentation.

### 3. **Gradient Checkpointing** (Already Done)
Enabled in the model to trade compute for memory usage during backprop.

---

## 🎯 If You Still Get Memory Errors

### Option A: Reduce Image Size Further
Edit `/home/workspace/workspace/v1/fsod/config.py`:
```python
IMAGE_SIZE = 320  # Even smaller
```

### Option B: Reduce K_SHOT (Support Samples)
```python
K_SHOT = 2  # Instead of 3
```

### Option C: Use Mixed Precision
Add to `train.py` after imports:
```python
from torch.cuda.amp import autocast, GradScaler

# In training loop:
scaler = GradScaler()
with autocast():
    predictions = model(...)
    loss = compute_detection_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Option D: Clear Cache Between Episodes
```python
if torch.cuda.is_available() and (episode + 1) % 10 == 0:
    torch.cuda.empty_cache()
```

---

## 📊 Memory Usage by Image Size

| Image Size | Est. Memory | Speed | Quality |
|-----------|-----------|-------|---------|
| 512 | ~14.5 GB | Slow | Best |
| 384 | ~8.5 GB | Fast | Good ✅ |
| 320 | ~6 GB | Faster | OK |
| 256 | ~3.5 GB | Very Fast | Lower |

---

## ✨ Recommended Settings for Kaggle

```python
# config.py - Kaggle Optimized
IMAGE_SIZE = 384
N_WAY = 5
K_SHOT = 3  # Or 2 if still OOM
QUERY_SAMPLES = 10
NUM_EPISODES = 100  # Start small, increase after testing
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
```

---

## 🔍 Debug Steps

1. **Check GPU Memory:**
```python
import torch
print(torch.cuda.get_device_properties(0))
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

2. **Monitor During Training:**
```python
if (episode + 1) % 10 == 0:
    memory_used = torch.cuda.memory_allocated() / 1e9
    print(f"Memory used: {memory_used:.2f} GB")
```

3. **Profile Memory:**
```python
from torch.utils.checkpoint import checkpoint
# Use checkpoint to save memory on large models
```

---

## 🚀 When to Increase

Once training works with 100 episodes:
- ✅ Increase `NUM_EPISODES` to 500-1000
- ✅ Keep `IMAGE_SIZE = 384`
- ✅ Monitor GPU memory in first few episodes

---

## 📝 Notes

- **Default config already optimized** for Kaggle GPU
- **384×384 is the sweet spot** - good accuracy without OOM
- **ResNet-50 is heavy** - consider EfficientNet if needed
- **Batch size = 1** - episodic training doesn't benefit from batching
- **Kaggle timeout = 9 hours** - start with 100-500 episodes

---

## 💡 Quick Checklist

- ✅ Image size: 384 (not 512)
- ✅ CUDA memory optimization enabled
- ✅ K_SHOT = 3 (reasonable for Kaggle)
- ✅ NUM_EPISODES = 100 (start small)
- ✅ Gradient checkpointing enabled

If you still hit OOM errors after these steps, reduce `IMAGE_SIZE` to 320 or `K_SHOT` to 2.
