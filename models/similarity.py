"""
Similarity computation for FSOD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityMatcher(nn.Module):
    """Compute similarity between support and query features"""
    
    def __init__(self, embed_dim=512, temperature=5.0):
        super(SimilarityMatcher, self).__init__()
        self.embed_dim = embed_dim
        # Critical Fix #7: Lower initial temperature (5.0 vs 10.0) 
        # Prevents over-sharpening of similarity early in training
        self.temperature = temperature
    
    def forward(self, support_features, query_features):
        """
        Compute cosine similarity
        
        Args:
            support_features: [N*K, C] - Support set features
            query_features: [M, C] - Query features (M proposal features)
        
        Returns:
            similarity: [M, N*K] - Similarity scores
        """
        # Normalize features
        support_norm = F.normalize(support_features, p=2, dim=1)
        query_norm = F.normalize(query_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(query_norm, support_norm.t())
        
        # Scale by temperature
        similarity = similarity * self.temperature
        
        return similarity
    
    def compute_class_similarity(self, query_features, support_features, 
                                support_labels, n_way):
        """
        Compute similarity to each class (average over support examples)
        
        Args:
            query_features: [M, C]
            support_features: [N*K, C]
            support_labels: [N*K]
            n_way: Number of classes
        
        Returns:
            class_similarity: [M, N_way]
        """
        M = query_features.shape[0]
        class_similarities = []
        
        # Compute similarity for each class
        for class_id in range(n_way):
            # Get support features for this class
            mask = support_labels == class_id
            class_support = support_features[mask]
            
            if class_support.shape[0] == 0:
                # No support examples for this class
                class_sim = torch.zeros(M, device=query_features.device)
            else:
                # Compute similarity and average
                sim = self.forward(class_support, query_features)  # [M, K]
                class_sim = sim.mean(dim=1)  # [M]
            
            class_similarities.append(class_sim)
        
        # Stack to [M, N_way]
        class_similarity = torch.stack(class_similarities, dim=1)
        
        return class_similarity


class ProposalGenerator(nn.Module):
    """Generate object proposals using simple sliding window"""
    
    def __init__(self, feature_size=16, scales=[16, 32, 64, 128, 256], ratios=[0.5, 1.0, 2.0]):
        super(ProposalGenerator, self).__init__()
        self.feature_size = feature_size
        self.scales = scales  # Anchor sizes covering different object scales
        self.ratios = ratios
        
        # Generate anchor boxes
        self.anchors = self._generate_anchors()
    
    def _generate_anchors(self):
        """Generate anchor boxes for each feature map location"""
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                w = scale * (ratio ** 0.5)
                h = scale / (ratio ** 0.5)
                anchors.append([w, h])
        return torch.tensor(anchors, dtype=torch.float32)
    
    def generate_proposals(self, feature_map_size, image_size):
        """
        Generate proposal boxes with subsampling for efficiency
        
        Args:
            feature_map_size: (H, W) of feature map
            image_size: Original image size
        
        Returns:
            proposals: [N, 4] boxes in [x_center, y_center, w, h]
        """
        H, W = feature_map_size
        stride_h = image_size / float(H)
        stride_w = image_size / float(W)
        
        proposals = []
        
        # Critical Fix #2: Subsample to reduce proposal density
        # Reduces computation by 4-16x while maintaining coverage
        step = max(1, int(stride_w / 16.0))  # Sample every ~16 pixels at image scale
        
        # For each subsampled feature map location
        for i in range(0, H, step):
            for j in range(0, W, step):
                # Center of the cell in image coordinates
                cx = (j + 0.5) * stride_w
                cy = (i + 0.5) * stride_h
                
                # Generate anchors at this location
                for anchor_w, anchor_h in self.anchors:
                    proposals.append([cx, cy, anchor_w, anchor_h])
        
        return torch.tensor(proposals, dtype=torch.float32)
    
    def proposals_to_boxes(self, proposals):
        """
        Convert proposals [cx, cy, w, h] to boxes [x, y, w, h]
        """
        boxes = proposals.clone()
        boxes[:, 0] = proposals[:, 0] - proposals[:, 2] / 2  # x
        boxes[:, 1] = proposals[:, 1] - proposals[:, 3] / 2  # y
        boxes[:, 2] = proposals[:, 2]  # w
        boxes[:, 3] = proposals[:, 3]  # h
        return boxes


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply NMS to proposals
    
    Args:
        boxes: [N, 4] in [x, y, w, h] format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Convert to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        # Compute IoU with remaining boxes
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)