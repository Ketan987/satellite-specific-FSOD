"""
Main FSOD detector model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNet50Backbone, FeatureEmbedding, ROIPooling
from models.similarity import SimilarityMatcher, ProposalGenerator, non_max_suppression


class FSODDetector(nn.Module):
    """Few-Shot Object Detector"""
    
    def __init__(self, feature_dim=2048, embed_dim=512, image_size=512, pretrained=True):
        super(FSODDetector, self).__init__()
        
        # Feature extraction
        self.backbone = ResNet50Backbone(pretrained=pretrained, feature_dim=feature_dim)
        self.embedding = FeatureEmbedding(in_dim=feature_dim, embed_dim=embed_dim)
        
        # ROI pooling
        self.roi_pooling = ROIPooling(output_size=7)
        
        # Similarity matcher
        self.similarity_matcher = SimilarityMatcher(embed_dim=embed_dim)
        
        # Proposal generator
        self.proposal_generator = ProposalGenerator()
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim * 7 * 7, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
        )
        
        # Box refinement
        self.box_regressor = nn.Linear(512, 4)
        
        # Objectness: predicts "is this an object?" conditioned on class
        # Concatenate class logits with features for objectness prediction
        # This makes objectness depend on predicted class (joint learning)
        self.objectness = nn.Sequential(
            nn.Linear(512 + 512, 256),  # 512 features + 512 from class context
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.image_size = image_size
        self.embed_dim = embed_dim
    
    def extract_features(self, images):
        """Extract features from images"""
        # Backbone features
        features = self.backbone(images)  # [B, 2048, H, W]
        
        # Embed features
        embedded = self.embedding(features)  # [B, embed_dim, H, W]
        
        return embedded
    
    def extract_roi_features(self, features, boxes):
        """Extract ROI features and flatten"""
        # ROI pooling
        roi_features = self.roi_pooling(features, boxes, self.image_size)
        # [N, embed_dim, 7, 7]
        
        # Flatten
        N = roi_features.shape[0]
        roi_flat = roi_features.view(N, -1)  # [N, embed_dim * 7 * 7]
        
        # Detection head
        det_features = self.detection_head(roi_flat)  # [N, 512]
        
        return det_features
    
    def forward(self, support_images, support_boxes, support_labels, query_images, n_way=None):
        """
        Forward pass
        
        Args:
            support_images: [N*K, 3, H, W]
            support_boxes: List of [num_boxes, 4]
            query_images: [Q, 3, H, W]
        
        Returns:
            predictions: Dict with boxes, scores, features
        """
        # Extract features
        support_features = self.extract_features(support_images)
        query_features = self.extract_features(query_images)
        
        # ===== CRITICAL FIX #6: Freeze Support Features =====
        # Support features should not change during training (they're the reference)
        # This prevents gradient explosion and keeps support representations stable
        with torch.no_grad():
            support_roi_features = self.extract_roi_features(
                support_features, support_boxes
            ).detach()
        
        # Query features are NOT frozen - they should adapt during training
        query_roi_features_all = []  # Will be filled per query image
        
        # Expand support_labels to match number of boxes
        # (one label per support image, but ROI pooling may produce multiple features per image)
        support_box_labels = []
        for img_idx, boxes in enumerate(support_boxes):
            # boxes is a tensor, boxes.shape[0] is number of boxes
            num_boxes = boxes.shape[0] if isinstance(boxes, torch.Tensor) else len(boxes)
            label_val = support_labels[img_idx].item() if isinstance(support_labels[img_idx], torch.Tensor) else support_labels[img_idx]
            support_box_labels.extend([label_val] * num_boxes)
        support_box_labels = torch.tensor(support_box_labels, dtype=support_labels.dtype, device=support_labels.device)        # Generate proposals for query images
        Q, _, H, W = query_features.shape
        feature_map_size = (query_features.shape[2], query_features.shape[3])
        
        all_predictions = []
        
        for q in range(Q):
            # Generate proposals on correct device
            proposals = self.proposal_generator.generate_proposals(
                feature_map_size, self.image_size, device=query_features.device
            )
            
            # Convert to boxes
            proposal_boxes = self.proposal_generator.proposals_to_boxes(proposals, device=query_features.device)
            
            # Extract features for proposals
            query_roi_features = self.extract_roi_features(
                query_features[q:q+1], [proposal_boxes]
            )

            # Compute similarity to support examples -> [num_proposals, N*K]
            similarity = self.similarity_matcher(
                support_roi_features, query_roi_features
            )

            # Compute class-level similarity logits [num_proposals, n_way]
            if n_way is None:
                # try to infer n_way from support_labels
                n_way = int(torch.max(support_labels).item()) + 1

            class_sim = self.similarity_matcher.compute_class_similarity(
                query_roi_features, support_roi_features, support_box_labels, n_way
            )  # [num_proposals, n_way]

            # Refine boxes
            box_deltas = self.box_regressor(query_roi_features)
            refined_boxes = self.apply_box_deltas(proposal_boxes, box_deltas)

            # ===== CRITICAL FIX #4: Immediate Box Validation =====
            # Validate boxes right after refinement (don't wait for ROI pooling)
            # This prevents clustering from invalid box coordinates
            MIN_BOX_SIZE = 2.0
            
            # Clamp center coordinates to valid range
            refined_boxes[:, 0] = torch.clamp(refined_boxes[:, 0], 0.0, 
                                             float(self.image_size) - MIN_BOX_SIZE)
            refined_boxes[:, 1] = torch.clamp(refined_boxes[:, 1], 0.0, 
                                             float(self.image_size) - MIN_BOX_SIZE)
            
            # Clamp dimensions to positive values
            refined_boxes[:, 2] = torch.clamp(refined_boxes[:, 2], MIN_BOX_SIZE, 
                                             float(self.image_size))
            refined_boxes[:, 3] = torch.clamp(refined_boxes[:, 3], MIN_BOX_SIZE, 
                                             float(self.image_size))
            
            # Additional safety: ensure bottom-right corner stays in bounds
            max_x = refined_boxes[:, 0] + refined_boxes[:, 2]
            max_y = refined_boxes[:, 1] + refined_boxes[:, 3]
            
            # If extends beyond image, shrink box
            out_of_bounds_x = max_x > self.image_size
            out_of_bounds_y = max_y > self.image_size
            
            if out_of_bounds_x.sum() > 0:
                refined_boxes[out_of_bounds_x, 2] -= (max_x[out_of_bounds_x] - self.image_size)
            if out_of_bounds_y.sum() > 0:
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

            # ===== CRITICAL FIX #3: Joint Objectness/Classification =====
            # Objectness is CONDITIONED on predicted class (not independent)
            # This prevents high objectness scores for wrong classes
            
            # Detach class logits to prevent circular gradients during objectness prediction
            class_logits_for_objectness = class_sim.detach()  # [num_proposals, n_way]
            
            # Create augmented feature vector: original features + class context
            # This makes objectness depend on what class the proposal predicts
            augmented_features = torch.cat([
                query_roi_features,  # [P, 512] - visual features
                class_logits_for_objectness  # [P, n_way] 
            ], dim=1)  # [P, 512 + n_way]
            
            # Predict objectness conditioned on both visual features AND class prediction
            objectness_raw = self.objectness(augmented_features).squeeze(1)  # [P]
            objectness = torch.sigmoid(objectness_raw)  # [P], range [0, 1]
            
            # Convert class_sim to probabilities
            class_probs = torch.softmax(class_sim, dim=1)
            final_scores = max_class_probs * objectness
            
            # Critical Fix #5: Pre-filter proposals by objectness before NMS
            # Reduces computation by filtering obvious negatives early
            objectness_threshold = 0.5  # Increased from 0.3 to filter more aggressively
            objectness_keep = objectness >= objectness_threshold
            if objectness_keep.sum() == 0:
                # No proposals pass threshold, keep top 5% (stricter)
                top_k = max(1, len(objectness) // 20)
                _, top_indices = torch.topk(objectness, min(top_k, len(objectness)))
                objectness_keep = torch.zeros_like(objectness_keep)
                objectness_keep[top_indices] = True
            
            # Filter all predictions by objectness
            refined_boxes = refined_boxes[objectness_keep]
            final_scores = final_scores[objectness_keep]
            class_logits = class_sim[objectness_keep]
            pred_classes = pred_classes[objectness_keep]

            all_predictions.append({
                'boxes': refined_boxes,
                'scores': final_scores,
                'class_logits': class_sim,
                'pred_classes': pred_classes,
                'objectness_scores': objectness,
                'similarity': similarity
            })
        
        return all_predictions
    
    def apply_box_deltas(self, boxes, deltas):
        """Apply box regression deltas with constrained refinement."""
        # Don't clone here - create new tensor to avoid in-place ops
        # Constrain box regression deltas to avoid explosion
        deltas_clamped = torch.clamp(deltas, -0.5, 0.5)
        
        # Apply delta: dx, dy are relative to box size; dw, dh are log-ratios
        # Use torch.stack to create new tensor instead of in-place assignment
        refined_x = boxes[:, 0] + deltas_clamped[:, 0] * boxes[:, 2] * 0.1  # x
        refined_y = boxes[:, 1] + deltas_clamped[:, 1] * boxes[:, 3] * 0.1  # y
        refined_w = boxes[:, 2] * torch.exp(torch.clamp(deltas_clamped[:, 2], -0.2, 0.2))  # w
        refined_h = boxes[:, 3] * torch.exp(torch.clamp(deltas_clamped[:, 3], -0.2, 0.2))  # h
        
        # Enforce minimum size and image bounds
        refined_w = torch.clamp(refined_w, 1.0, float(self.image_size))
        refined_h = torch.clamp(refined_h, 1.0, float(self.image_size))
        
        # Clamp center coordinates to valid region
        refined_x = torch.clamp(refined_x, 0.0, float(self.image_size) - 1.0)
        refined_y = torch.clamp(refined_y, 0.0, float(self.image_size) - 1.0)
        
        # Stack to create refined boxes [N, 4]
        refined = torch.stack([refined_x, refined_y, refined_w, refined_h], dim=1)
        
        return refined
    
    def predict(self, support_images, support_boxes, support_labels, query_image, 
                score_threshold=0.3, nms_threshold=0.4, max_detections=100, n_way=None):
        """
        Inference method
        
        Args:
            support_images: [N*K, 3, H, W]
            support_boxes: List of [num_boxes, 4]
            query_image: [1, 3, H, W]
            score_threshold: Minimum score threshold
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
        
        Returns:
            boxes: [N, 4] detected boxes
            scores: [N] similarity scores
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(support_images, support_boxes, support_labels, query_image, n_way=n_way)
            pred = predictions[0]

            boxes = pred['boxes']
            scores = pred['scores']
            pred_classes = pred.get('pred_classes', None)
            
            # Filter by score threshold
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            
            if len(boxes) == 0:
                return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
            
            # Apply NMS
            keep_indices = non_max_suppression(boxes, scores, nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            if pred_classes is not None:
                pred_classes = pred_classes[keep_indices]
            
            # Limit to max detections
            if len(boxes) > max_detections:
                boxes = boxes[:max_detections]
                scores = scores[:max_detections]
                if pred_classes is not None:
                    pred_classes = pred_classes[:max_detections]

            return boxes, scores, pred_classes


def compute_detection_loss(predictions, target_boxes, target_labels, iou_threshold=0.3, box_loss_weight=1.0, use_focal=True):
    """
    Compute detection loss with hard negative mining.
    
    CRITICAL FIX: Hard negative mining focuses training on discriminative regions.
    Positives: IoU > threshold
    Hard Negatives: High objectness but low IoU (proposals near objects)
    Easy Negatives: Ignored (waste of training signal)
    
    Ratio: 3 negatives per positive (standard in object detection)

    Args:
        predictions: List of prediction dicts (per query image)
        target_boxes: List of [num_boxes, 4] (per query image)
        target_labels: List of [num_boxes] (per query image) -- labels in [0..n_way-1]
        iou_threshold: IoU threshold for positive/negative
        box_loss_weight: Weight for box regression loss
        use_focal: Use focal loss for class imbalance

    Returns:
        loss: scalar tensor
    """
    total_loss = 0.0
    device = None
    num_imgs = 0

    for pred, gt_boxes, gt_labels in zip(predictions, target_boxes, target_labels):
        pred_boxes = pred['boxes']  # [P, 4]
        class_logits = pred.get('class_logits', None)  # [P, n_way]
        objectness_scores = pred.get('objectness_scores', torch.ones(len(pred_boxes)))  # [P]

        if device is None:
            device = pred_boxes.device

        if class_logits is None or len(pred_boxes) == 0:
            continue

        num_imgs += 1
        P = pred_boxes.shape[0]
        n_way = class_logits.shape[1]

        # Compute IoU matrix between predictions and GTs
        if len(gt_boxes) == 0:
            # No GT -> all proposals treated as hard negatives
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
        pos_mask = max_ious > iou_threshold  # [P] - positive proposals
        neg_mask = ~pos_mask  # [P] - negative proposals
        
        # Select hard negatives: high objectness but low IoU
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        
        hard_neg_mask = torch.zeros(P, dtype=torch.bool, device=device)
        
        if num_pos > 0 and num_neg > 0:
            # Target: 3 negatives per positive (standard ratio)
            target_num_hard_neg = min(num_neg, max(1, num_pos * 3))
            
            # Get objectness scores of negative proposals
            neg_objectness = objectness_scores[neg_mask].detach()
            
            # Select top hard negatives by objectness
            if target_num_hard_neg < num_neg:
                _, top_hard_indices = torch.topk(neg_objectness, target_num_hard_neg)
                # Map back to full indices
                neg_indices = torch.where(neg_mask)[0]
                hard_neg_indices = neg_indices[top_hard_indices]
                hard_neg_mask[hard_neg_indices] = True
            else:
                # Keep all negatives if fewer than target
                hard_neg_mask[neg_mask] = True
        elif num_pos == 0 and num_neg > 0:
            # No positives: keep top 5% of negatives
            neg_objectness = objectness_scores[neg_mask].detach()
            top_k = max(1, num_neg // 20)  # 5% of negatives
            _, top_indices = torch.topk(neg_objectness, min(top_k, num_neg))
            neg_indices = torch.where(neg_mask)[0]
            hard_neg_mask[neg_indices[top_indices]] = True
        
        # Selected proposals: positives + hard negatives only
        selected_mask = pos_mask | hard_neg_mask
        
        if selected_mask.sum() == 0:
            # Safety: if no proposals selected, use positives
            selected_mask = pos_mask
        
        if selected_mask.sum() == 0:
            # Still no proposals, skip this image
            continue

        # ===== ASSIGN TARGETS =====
        targets = torch.zeros(P, dtype=torch.long, device=device)
        pos_indices = torch.where(pos_mask)[0]
        
        if len(pos_indices) > 0:
            matched_gt_idxs = gt_indices[pos_indices]
            matched_labels = gt_labels[matched_gt_idxs]
            
            # Ensure matched_labels are valid (should be 0..n_way-1)
            if isinstance(matched_labels, torch.Tensor):
                matched_labels = torch.clamp(matched_labels, 0, n_way - 1)
            
            # Shift to 1-indexed (0 is background), but ensure it doesn't exceed n_way
            shifted_labels = matched_labels + 1
            shifted_labels = torch.clamp(shifted_labels, 0, n_way)
            
            # Set targets for positive proposals
            targets[pos_indices] = shifted_labels.long()

        # ===== COMPUTE LOSS ONLY ON SELECTED PROPOSALS =====
        selected_logits = class_logits[selected_mask]
        selected_targets = targets[selected_mask]
        
        # Classification loss
        if use_focal:
            cls_loss = focal_loss_ce(selected_logits, selected_targets, alpha=0.25, gamma=2.0)
        else:
            cls_loss = F.cross_entropy(selected_logits, selected_targets)

        # Box regression loss for positive proposals only
        box_loss = torch.tensor(0.0, device=device)
        selected_pos_mask = pos_mask[selected_mask]
        
        if selected_pos_mask.sum() > 0:
            # Get positive proposals from selected proposals
            selected_pred_boxes = pred_boxes[selected_mask]
            pred_pos = selected_pred_boxes[selected_pos_mask]
            
            # Map back to gt indices
            selected_pos_indices = torch.where(selected_mask)[0][torch.where(selected_pos_mask)[0]]
            gt_idx_for_pos = gt_indices[selected_pos_indices]
            gt_pos = gt_boxes[gt_idx_for_pos]
            
            box_loss = F.smooth_l1_loss(pred_pos, gt_pos, reduction='mean')

        total_loss += cls_loss + box_loss * box_loss_weight

    if num_imgs == 0:
        return torch.tensor(0.0, device=device if device is not None else 'cpu')

    return total_loss / num_imgs


def focal_loss_ce(class_logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Addresses class imbalance and hard example mining.
    Properly validates targets instead of silently corrupting them.
    
    Args:
        class_logits: [N, C] logits
        targets: [N] class indices (0..C-1)
        alpha: weighting for background (0.25)
        gamma: focusing parameter (2.0)
    """
    num_classes = class_logits.shape[1]
    
    # VALIDATE targets - don't silently corrupt data
    if targets.max() >= num_classes:
        raise ValueError(f"Invalid target label {int(targets.max())} >= num_classes {num_classes}")
    if targets.min() < 0:
        raise ValueError(f"Invalid target label {int(targets.min())} < 0")
    
    # Compute log softmax
    log_probs = F.log_softmax(class_logits, dim=-1)  # [N, C]
    
    # Gather log probabilities for target classes
    log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    p_t = torch.exp(log_p_t)  # [N]
    
    # Focal weight: (1 - p_t)^gamma focuses on hard examples
    focal_weight = (1.0 - p_t) ** gamma  # [N]
    
    # Focal loss
    loss = -focal_weight * log_p_t  # [N]
    
    return loss.mean()


def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU matrix between two sets of boxes with safety checks"""
    N, M = len(boxes1), len(boxes2)
    device = boxes1.device if isinstance(boxes1, torch.Tensor) else 'cpu'
    ious = torch.zeros(N, M, device=device, dtype=torch.float32)
    
    if N == 0 or M == 0:
        return ious
    
    for i in range(N):
        for j in range(M):
            try:
                iou = compute_box_iou(boxes1[i], boxes2[j])
                ious[i, j] = max(0.0, min(1.0, float(iou)))  # Clamp to [0, 1]
            except Exception as e:
                # On error, assign 0 IoU
                ious[i, j] = 0.0
    
    return ious


def compute_box_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format with safety checks"""
    # Handle tensor conversion with validation
    try:
        x1 = box1[0].item() if isinstance(box1[0], torch.Tensor) else float(box1[0])
        y1 = box1[1].item() if isinstance(box1[1], torch.Tensor) else float(box1[1])
        w1 = box1[2].item() if isinstance(box1[2], torch.Tensor) else float(box1[2])
        h1 = box1[3].item() if isinstance(box1[3], torch.Tensor) else float(box1[3])
        
        x2 = box2[0].item() if isinstance(box2[0], torch.Tensor) else float(box2[0])
        y2 = box2[1].item() if isinstance(box2[1], torch.Tensor) else float(box2[1])
        w2 = box2[2].item() if isinstance(box2[2], torch.Tensor) else float(box2[2])
        h2 = box2[3].item() if isinstance(box2[3], torch.Tensor) else float(box2[3])
        
        # Validate boxes have positive dimensions
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0
        
        # Convert to [x1, y1, x2, y2]
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        # Intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        iou = inter_area / union_area
        return max(0.0, min(1.0, iou))  # Clamp to [0, 1]
    
    except Exception as e:
        return 0.0