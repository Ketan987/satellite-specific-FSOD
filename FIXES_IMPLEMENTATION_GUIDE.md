# FSOD Implementation Fixes - Quick Reference Guide

This document provides exact code fixes for the critical issues identified in the code review.

---

## FIX #1: Implement True Class Prototypes

**File:** `models/similarity.py`  
**Lines to Replace:** 40-60

**OLD CODE:**
```python
def compute_class_similarity(self, query_features, support_features, 
                            support_labels, n_way):
    M = query_features.shape[0]
    class_similarities = []
    
    for class_id in range(n_way):
        mask = support_labels == class_id
        class_support = support_features[mask]
        
        if class_support.shape[0] == 0:
            class_sim = torch.zeros(M, device=query_features.device)
        else:
            sim = self.forward(class_support, query_features)  # [M, K]
            class_sim = sim.mean(dim=1)  # Average similarities
        
        class_similarities.append(class_sim)
    
    class_similarity = torch.stack(class_similarities, dim=1)
    return class_similarity
```

**NEW CODE (Prototypical Networks approach):**
```python
def compute_class_similarity(self, query_features, support_features, 
                            support_labels, n_way):
    """
    Compute similarity to class prototypes.
    
    Prototypical Networks: Learn to classify based on prototypical representations.
    Prototype = mean of support features for each class.
    """
    M = query_features.shape[0]
    class_similarities = []
    
    # Normalize query features once
    query_norm = F.normalize(query_features, p=2, dim=1)  # [M, C]
    
    for class_id in range(n_way):
        mask = support_labels == class_id
        class_support = support_features[mask]  # [K, C]
        
        if class_support.shape[0] == 0:
            # No support examples for this class
            class_sim = torch.zeros(M, device=query_features.device)
        else:
            # Create class prototype: mean of support features
            prototype = class_support.mean(dim=0, keepdim=True)  # [1, C]
            
            # Normalize prototype
            proto_norm = F.normalize(prototype, p=2, dim=1)  # [1, C]
            
            # Compute cosine similarity: [M, 1] -> [M]
            class_sim = torch.mm(query_norm, proto_norm.t()).squeeze(1)
            
            # Scale by temperature
            class_sim = class_sim * self.temperature
        
        class_similarities.append(class_sim)
    
    # Stack to [M, n_way]
    class_similarity = torch.stack(class_similarities, dim=1)
    return class_similarity
```

**Why This Works:**
- Replaces averaging of raw similarities with averaging of feature representations
- Creates proper prototype = centroid of support examples
- Follows Prototypical Networks paper (Snell et al., 2017)
- Better generalization to novel classes

---

## FIX #2: Add Hard Negative Mining to Loss Function

**File:** `models/detector.py`  
**Location:** In `compute_detection_loss()`, around line 301-330

**REPLACEMENT:** Update the main loss computation loop:

```python
def compute_detection_loss(predictions, target_boxes, target_labels, 
                          iou_threshold=0.3, box_loss_weight=1.0, use_focal=True):
    """
    Compute detection loss with hard negative mining.
    
    Hard negative mining: Select hard negatives (high objectness but low IoU)
    to focus training on discriminative regions.
    """
    total_loss = 0.0
    device = None
    num_imgs = 0

    for pred, gt_boxes, gt_labels in zip(predictions, target_boxes, target_labels):
        pred_boxes = pred['boxes']  # [P, 4]
        class_logits = pred.get('class_logits', None)  # [P, n_way]
        objectness = pred.get('objectness', torch.ones(len(pred_boxes)))  # [P]

        if device is None:
            device = pred_boxes.device

        if class_logits is None or len(pred_boxes) == 0:
            continue

        num_imgs += 1
        P = pred_boxes.shape[0]
        n_way = class_logits.shape[1]

        # Compute IoU matrix
        if len(gt_boxes) == 0:
            targets = torch.zeros(P, dtype=torch.long, device=device)
            if use_focal:
                cls_loss = focal_loss_ce(class_logits, targets, alpha=0.25, gamma=2.0)
            else:
                cls_loss = F.cross_entropy(class_logits, targets)
            total_loss += cls_loss
            continue

        ious = compute_iou_matrix(pred_boxes, gt_boxes)  # [P, G]
        max_ious, gt_indices = ious.max(dim=1)  # [P]

        # ===== HARD NEGATIVE MINING =====
        pos_mask = max_ious > iou_threshold  # [P] bool
        neg_mask = max_ious <= iou_threshold  # [P] bool
        
        # Identify hard negatives: high objectness but low IoU
        hard_neg_score = objectness.detach()
        hard_neg_mask = neg_mask.clone()
        
        # Select hard negatives: top scoring among negatives
        num_pos = pos_mask.sum().item()
        if num_pos > 0:
            # Target ratio: 3 negatives per positive (standard in object detection)
            target_num_neg = min(neg_mask.sum().item(), max(1, num_pos * 3))
            
            if target_num_neg > 0 and neg_mask.sum() > 0:
                # Get scores of negative proposals
                neg_scores = hard_neg_score[neg_mask]
                
                # Keep only top hard negatives
                if neg_scores.shape[0] > target_num_neg:
                    neg_indices = torch.topk(neg_scores, target_num_neg)[1]
                    # Map back to full indices
                    full_neg_indices = torch.where(neg_mask)[0][neg_indices]
                    hard_neg_mask[torch.where(neg_mask)[0] != full_neg_indices] = False
                # else: keep all negatives if fewer than target
        else:
            # No positive examples, keep top 5% negatives only
            hard_neg_mask = neg_mask.clone()
            if neg_mask.sum() > 20:
                neg_scores = hard_neg_score[neg_mask]
                top_k = max(1, neg_mask.sum().item() // 20)  # 5%
                _, top_indices = torch.topk(neg_scores, top_k)
                full_indices = torch.where(neg_mask)[0][top_indices]
                hard_neg_mask[torch.where(neg_mask)[0] != full_indices] = False
        
        # Select training examples: positive + hard negatives
        selected_mask = pos_mask | hard_neg_mask  # [P] bool
        
        if selected_mask.sum() == 0:
            # Fallback: use all proposals
            selected_mask = torch.ones(P, dtype=torch.bool, device=device)
        
        # ===== COMPUTE TARGETS =====
        targets = torch.zeros(P, dtype=torch.long, device=device)
        pos_indices = torch.where(pos_mask)[0]
        if len(pos_indices) > 0:
            matched_gt_idxs = gt_indices[pos_indices]
            matched_labels = gt_labels[matched_gt_idxs]
            matched_labels = torch.clamp(matched_labels, 0, n_way - 1)
            targets[pos_indices] = matched_labels.long() + 1  # Shift to 1-indexed
            targets = torch.clamp(targets, 0, n_way)

        # ===== COMPUTE LOSS ON SELECTED PROPOSALS ONLY =====
        selected_logits = class_logits[selected_mask]
        selected_targets = targets[selected_mask]
        
        if use_focal:
            cls_loss = focal_loss_ce(selected_logits, selected_targets, 
                                     alpha=0.25, gamma=2.0)
        else:
            cls_loss = F.cross_entropy(selected_logits, selected_targets)

        # Box regression loss
        box_loss = torch.tensor(0.0, device=device)
        selected_pos_mask = pos_mask[selected_mask]
        if selected_pos_mask.sum() > 0:
            selected_pred_pos = pred_boxes[selected_mask][selected_pos_mask]
            # Map back to gt indices
            gt_idx_selected = gt_indices[selected_mask][selected_pos_mask]
            gt_pos = gt_boxes[gt_idx_selected]
            box_loss = F.smooth_l1_loss(selected_pred_pos, gt_pos, reduction='mean')

        total_loss += cls_loss + box_loss * box_loss_weight

    if num_imgs == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / num_imgs
```

**Key Changes:**
- Identifies hard negatives: `neg_mask.sum() > 0 and objectness[neg_mask] > threshold`
- Maintains positive:negative ratio (3:1 standard)
- Focuses loss computation on important proposals only
- Fallback logic for edge cases (no positives, etc.)

---

## FIX #3: Joint Objectness and Classification Learning

**File:** `models/detector.py`  
**Location:** In `FSODDetector.__init__()` and `forward()`

**STEP 1: Modify detector head to output objectness as confidence gate**

```python
# In FSODDetector.__init__() around line 26-34:

# OLD:
self.objectness = nn.Linear(512, 1)

# NEW: Make objectness a confidence score modulated by class predictions
self.objectness = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
```

**STEP 2: Modify forward pass to condition objectness on class**

```python
# In forward() around line 128-145, REPLACE THIS:

# Old way (independent):
objectness = torch.sigmoid(self.objectness(query_roi_features).squeeze(1))
class_probs = torch.softmax(class_sim, dim=1)
max_class_probs, pred_classes = class_probs.max(dim=1)
final_scores = max_class_probs * objectness

# NEW way (joint/conditioned):
# Concatenate class logits with features for objectness prediction
class_logits_detached = class_sim.detach()  # [P, n_way]
augmented_features = torch.cat([query_roi_features, class_logits_detached], dim=1)

# Predict objectness conditioned on class
objectness_raw = self.objectness(augmented_features).squeeze(1)  # [P]
objectness_prob = torch.sigmoid(objectness_raw)

# Combine with class probability
class_probs = torch.softmax(class_sim, dim=1)  # [P, n_way]
max_class_probs, pred_classes = class_probs.max(dim=1)  # [P]

# Use objectness as modulation factor (0-1 range)
final_scores = max_class_probs * objectness_prob
```

**Why This Works:**
- Objectness now depends on predicted class (via concatenation)
- Prevents high objectness for wrong classes
- More natural score combination
- Reduces spurious detections

---

## FIX #4: Immediate Box Coordinate Validation

**File:** `models/detector.py`  
**Location:** In `forward()` right after box refinement, around line 125

**ADD THIS CODE** after `refined_boxes = self.apply_box_deltas(proposal_boxes, box_deltas)`:

```python
# Immediately validate refined boxes (don't wait for ROI pooling)
refined_boxes = self.apply_box_deltas(proposal_boxes, box_deltas)

# ===== IMMEDIATE BOX VALIDATION =====
# Ensure boxes are within image bounds and have positive dimensions
MIN_BOX_SIZE = 2.0

# Clamp center coordinates
refined_boxes[:, 0] = torch.clamp(refined_boxes[:, 0], 0.0, 
                                 float(self.image_size) - MIN_BOX_SIZE)
refined_boxes[:, 1] = torch.clamp(refined_boxes[:, 1], 0.0, 
                                 float(self.image_size) - MIN_BOX_SIZE)

# Clamp dimensions
refined_boxes[:, 2] = torch.clamp(refined_boxes[:, 2], MIN_BOX_SIZE, 
                                 float(self.image_size))
refined_boxes[:, 3] = torch.clamp(refined_boxes[:, 3], MIN_BOX_SIZE, 
                                 float(self.image_size))

# Additional safety: ensure bottom-right corner is in bounds
max_x = refined_boxes[:, 0] + refined_boxes[:, 2]
max_y = refined_boxes[:, 1] + refined_boxes[:, 3]

# If extends beyond image, shrink box
out_of_bounds_x = max_x > self.image_size
out_of_bounds_y = max_y > self.image_size

refined_boxes[out_of_bounds_x, 2] -= (max_x[out_of_bounds_x] - self.image_size)
refined_boxes[out_of_bounds_y, 3] -= (max_y[out_of_bounds_y] - self.image_size)

# Final safety clamp
refined_boxes[:, 0] = torch.clamp(refined_boxes[:, 0], 0.0, 
                                 float(self.image_size) - MIN_BOX_SIZE)
refined_boxes[:, 1] = torch.clamp(refined_boxes[:, 1], 0.0, 
                                 float(self.image_size) - MIN_BOX_SIZE)
refined_boxes[:, 2] = torch.clamp(refined_boxes[:, 2], MIN_BOX_SIZE, 
                                 float(self.image_size))
refined_boxes[:, 3] = torch.clamp(refined_boxes[:, 3], MIN_BOX_SIZE, 
                                 float(self.image_size))
```

---

## FIX #5: Add Support Set Diversity Constraints

**File:** `utils/data_loader.py`  
**Location:** In `FSODDataset.__getitem__()`, modify support sampling

**ADD THIS FUNCTION** to `FSODDataset` class:

```python
def _compute_box_statistics(self, boxes):
    """Compute statistics to measure diversity"""
    if len(boxes) == 0:
        return None
    
    statistics = []
    for box in boxes:
        x, y, w, h = box
        area = w * h
        aspect_ratio = w / (h + 1e-6)
        center = (x + w/2, y + h/2)
        statistics.append({
            'area': area,
            'aspect_ratio': aspect_ratio,
            'center': center,
            'box': box
        })
    return statistics

def _select_diverse_support(self, all_examples_for_class, k_shot):
    """Select k_shot diverse examples from class"""
    if len(all_examples_for_class) <= k_shot:
        return all_examples_for_class
    
    # Compute statistics for each example
    stats = [self._compute_box_statistics(ex.get('boxes', [])) 
             for ex in all_examples_for_class]
    
    # Simple diversity heuristic: max/min spread in area
    areas = [s[0]['area'] if s else 0 for s in stats]
    
    # Sort by area to get diversity
    sorted_indices = np.argsort(areas)
    
    # Select evenly spaced examples
    step = len(all_examples_for_class) // k_shot
    selected_indices = sorted_indices[::step][:k_shot]
    
    # If not enough, add more
    if len(selected_indices) < k_shot:
        remaining = [i for i in range(len(all_examples_for_class)) 
                    if i not in selected_indices]
        selected_indices = list(selected_indices) + remaining[:k_shot - len(selected_indices)]
    
    return [all_examples_for_class[i] for i in selected_indices[:k_shot]]
```

---

## FIX #6: Fix Focal Loss Target Validation

**File:** `models/detector.py`  
**Location:** In `focal_loss_ce()`, around line 368-370

**OLD CODE:**
```python
def focal_loss_ce(class_logits, targets, alpha=0.25, gamma=2.0):
    num_classes = class_logits.shape[1]
    if targets.max() >= num_classes or targets.min() < 0:
        targets = torch.clamp(targets, 0, num_classes - 1)  # Silent corruption!
```

**NEW CODE:**
```python
def focal_loss_ce(class_logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss with proper validation.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    num_classes = class_logits.shape[1]
    
    # Validate targets - don't silently corrupt
    if targets.max() >= num_classes:
        raise ValueError(f"Invalid target label {targets.max()} >= num_classes {num_classes}")
    if targets.min() < 0:
        raise ValueError(f"Invalid target label {targets.min()} < 0")
    
    # Compute log softmax
    log_probs = F.log_softmax(class_logits, dim=-1)  # [N, C]
    
    # Gather log probabilities for target classes
    log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    p_t = torch.exp(log_p_t)  # [N]
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1.0 - p_t) ** gamma  # [N]
    
    # Apply alpha weighting for class imbalance (optional)
    if alpha is not None:
        alpha_t = torch.where(targets == 0, alpha, 1.0 - alpha)
        focal_weight = alpha_t * focal_weight
    
    # Focal loss
    loss = -focal_weight * log_p_t  # [N]
    
    return loss.mean()
```

---

## FIX #7: Freeze Support Features During Training

**File:** `models/detector.py`  
**Location:** In `forward()`, around line 85

**MODIFY THIS:**
```python
# OLD:
support_roi_features = self.extract_roi_features(
    support_features, support_boxes
)

# NEW:
with torch.no_grad():
    support_roi_features = self.extract_roi_features(
        support_features, support_boxes
    ).detach()
```

**Rationale:**
- Prevents support features from changing during training
- Focuses gradients on query features and similarity matching
- Reduces variance in training

---

## Implementation Checklist

- [ ] Apply FIX #1: True class prototypes
- [ ] Apply FIX #2: Hard negative mining
- [ ] Apply FIX #3: Joint objectness/classification
- [ ] Apply FIX #4: Box validation
- [ ] Apply FIX #5: Support diversity (optional, for next phase)
- [ ] Apply FIX #6: Focal loss validation
- [ ] Apply FIX #7: Freeze support features
- [ ] Test training runs with small dataset (100 episodes)
- [ ] Verify detections spread across image
- [ ] Verify mAP increases with training

---

## Testing After Fixes

Run this quick test:
```bash
python train.py --num_episodes 100 --device cuda
```

**Success Criteria:**
- Loss decreases smoothly (no spikes)
- mAP increases from 0.0 to > 0.1
- Detections appear scattered across query images (not clustered)
- No errors about invalid targets
- Training completes without OOM

