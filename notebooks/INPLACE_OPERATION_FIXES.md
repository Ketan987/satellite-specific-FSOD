# In-Place Operation Fixes

## Problem
Error: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

This occurs when PyTorch detects that a tensor needed for backpropagation was modified in-place, breaking the computation graph.

## Root Causes & Fixes

### 1. **Box Delta Application** ✅ FIXED
**File:** `models/detector.py` - `apply_box_deltas()`

**Problem:**
```python
refined = boxes.clone().float()
refined[:, 0] = refined[:, 0] + ...  # In-place assignment
```

**Solution:**
```python
refined_x = boxes[:, 0] + deltas_clamped[:, 0] * boxes[:, 2] * 0.1
refined_y = boxes[:, 1] + deltas_clamped[:, 1] * boxes[:, 3] * 0.1
refined_w = boxes[:, 2] * torch.exp(...)
refined_h = boxes[:, 3] * torch.exp(...)
# Stack to create new tensor (non-in-place)
refined = torch.stack([refined_x, refined_y, refined_w, refined_h], dim=1)
```

### 2. **Target Assignment** ✅ FIXED
**File:** `models/detector.py` - `compute_detection_loss()`

**Problem:**
```python
targets = torch.zeros(P, dtype=torch.long, device=device)
targets[pos_mask] = shifted_labels  # In-place modification
```

**Solution:**
```python
targets = torch.where(pos_mask, shifted_labels.long(), targets)
```

### 3. **ReLU In-Place Operations** ✅ FIXED
**Files:** `models/backbone.py`, `models/detector.py`

**Problem:**
```python
nn.ReLU(inplace=True)  # Modifies input in-place
```

**Solution:**
```python
nn.ReLU(inplace=False)  # Creates new tensor
```

## Key Changes

| File | Location | Change |
|------|----------|--------|
| detector.py | apply_box_deltas() | Replaced in-place assignments with torch.stack() |
| detector.py | compute_detection_loss() | Replaced targets[pos_mask] = with torch.where() |
| detector.py | __init__ | Changed ReLU(inplace=True) → False |
| backbone.py | FeatureEmbedding | Changed ReLU(inplace=True) → False |

## Why This Matters

PyTorch's autograd engine tracks all tensor operations. In-place operations can corrupt this record because:
1. They modify the original tensor
2. But the computation graph still references the old value
3. During backward pass, gradients can't be computed correctly

## Testing

After pulling these changes:
```bash
cd /kaggle/working
git pull
# Restart kernel
python train.py --device cuda --num_episodes 100
```

Training should now complete without in-place operation errors.

## Memory Impact

- **In-place=True:** Lower memory, faster (but breaks gradients)
- **In-place=False:** Slightly higher memory, but correct gradients ✅

On Kaggle GPU (16GB), the difference is negligible for this model.

## Prevention Tips

- ✅ Never use `tensor[:, i] = value` in forward pass
- ✅ Use `torch.stack()`, `torch.cat()`, `torch.where()` instead
- ✅ Avoid `inplace=True` in any layer used during training
- ✅ Test with small batches first
