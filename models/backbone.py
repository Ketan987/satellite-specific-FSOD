"""
ResNet-50 backbone for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor"""
    
    def __init__(self, pretrained=True, feature_dim=2048):
        super(ResNet50Backbone, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final FC layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        self.feature_dim = feature_dim
        
        # Freeze early layers for faster training (optional)
        # self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze first few layers"""
        for name, param in self.features.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, 2048, H/32, W/32]
        """
        return self.features(x)


class FeatureEmbedding(nn.Module):
    """Project features to embedding space"""
    
    def __init__(self, in_dim=2048, embed_dim=512):
        super(FeatureEmbedding, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, embed_dim, kernel_size=1),
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: [B, in_dim, H, W]
        Returns:
            embedded: [B, embed_dim, H, W]
        """
        return self.projection(x)


class ROIPooling(nn.Module):
    """Simple ROI pooling for bounding boxes"""
    
    def __init__(self, output_size=7):
        super(ROIPooling, self).__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, features, boxes, image_size):
        """
        Extract features for each bounding box
        
        Args:
            features: [B, C, H, W]
            boxes: List of [num_boxes, 4] in [x, y, w, h] format
            image_size: Original image size
        
        Returns:
            roi_features: [total_boxes, C, output_size, output_size]
        """
        B, C, H, W = features.shape
        scale_h = H / image_size
        scale_w = W / image_size
        
        all_roi_features = []
        
        for b in range(B):
            if len(boxes) > b and len(boxes[b]) > 0:
                box_tensor = boxes[b]
                
                for box in box_tensor:
                    x, y, w, h = box
                    
                    # Ensure box has non-zero dimensions
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Convert to feature map coordinates with proper rounding
                    x1_f = x * scale_w
                    y1_f = y * scale_h
                    x2_f = (x + w) * scale_w
                    y2_f = (y + h) * scale_h
                    
                    # Convert tensors to float before rounding
                    x1_f = x1_f.item() if isinstance(x1_f, torch.Tensor) else x1_f
                    y1_f = y1_f.item() if isinstance(y1_f, torch.Tensor) else y1_f
                    x2_f = x2_f.item() if isinstance(x2_f, torch.Tensor) else x2_f
                    y2_f = y2_f.item() if isinstance(y2_f, torch.Tensor) else y2_f
                    
                    # Use round to nearest, then clamp
                    x1 = max(0, int(round(x1_f)))
                    y1 = max(0, int(round(y1_f)))
                    x2 = min(W, int(round(x2_f)) + 1)
                    y2 = min(H, int(round(y2_f)) + 1)
                    
                    # Ensure we have at least a 1x1 region
                    if x2 <= x1:
                        x2 = min(x1 + 1, W)
                    if y2 <= y1:
                        y2 = min(y1 + 1, H)
                    
                    # Extract and pool ROI
                    roi = features[b:b+1, :, y1:y2, x1:x2]
                    roi_pooled = self.adaptive_pool(roi)
                    all_roi_features.append(roi_pooled)
        
        if len(all_roi_features) == 0:
            # Return dummy feature if no boxes
            return torch.zeros(1, C, self.output_size, self.output_size, 
                             device=features.device)
        
        return torch.cat(all_roi_features, dim=0)