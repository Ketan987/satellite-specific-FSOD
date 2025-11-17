"""
ResNet-50 backbone for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor with multi-band support"""
    
    def __init__(self, pretrained=True, feature_dim=2048, input_channels=3):
        super(ResNet50Backbone, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Adapt first conv layer for different channel inputs
        if input_channels != 3:
            # Replace first conv layer to accept multi-band input
            original_conv = resnet.conv1
            self.input_adapter = nn.Conv2d(
                input_channels, 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize with mean of pretrained weights
            with torch.no_grad():
                # Average pretrained weights across input channels
                pretrained_weights = original_conv.weight  # [64, 3, 7, 7]
                if input_channels == 4:
                    # For 4-band: repeat the 3rd channel or average
                    adapted_weights = torch.cat([
                        pretrained_weights,
                        pretrained_weights[:, :1, :, :].mean(dim=1, keepdim=True)  # Average for 4th band
                    ], dim=1)
                    self.input_adapter.weight.copy_(adapted_weights)
            resnet.conv1 = self.input_adapter
        
        # Remove the final FC layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        self.feature_dim = feature_dim
        self.input_channels = input_channels
        
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
            x: [B, C, H, W] where C can be 3 or 4
        Returns:
            features: [B, 2048, H/32, W/32]
        """
        # Validate input channels match expected
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {x.shape[1]}")
        
        return self.features(x)


class FeatureEmbedding(nn.Module):
    """Project features to embedding space"""
    
    def __init__(self, in_dim=2048, embed_dim=512):
        super(FeatureEmbedding, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, 1024, kernel_size=1),
            nn.ReLU(inplace=False),
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
                        # Use center patch as fallback
                        x1_f = (image_size * 0.25) * scale_w
                        y1_f = (image_size * 0.25) * scale_h
                        x2_f = (image_size * 0.75) * scale_w
                        y2_f = (image_size * 0.75) * scale_h
                    else:
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
                    
                    # Use ceil/floor to ensure valid regions
                    x1 = max(0, int(x1_f))
                    y1 = max(0, int(y1_f))
                    x2 = min(W, int(x2_f) + 1)
                    y2 = min(H, int(y2_f) + 1)
                    
                    # Ensure we have at least a 2x2 region
                    if x2 <= x1 + 1:
                        x2 = min(x1 + 2, W)
                    if y2 <= y1 + 1:
                        y2 = min(y1 + 2, H)
                    
                    # Final safety check
                    x1 = max(0, min(x1, W - 2))
                    y1 = max(0, min(y1, H - 2))
                    x2 = max(x1 + 1, min(x2, W))
                    y2 = max(y1 + 1, min(y2, H))
                    
                    # Extract and pool ROI
                    roi = features[b:b+1, :, y1:y2, x1:x2]
                    
                    # Skip if region is somehow invalid
                    if roi.shape[2] < 1 or roi.shape[3] < 1:
                        continue
                    
                    roi_pooled = self.adaptive_pool(roi)
                    all_roi_features.append(roi_pooled)
        
        if len(all_roi_features) == 0:
            # Return dummy feature if no boxes
            return torch.zeros(1, C, self.output_size, self.output_size, 
                             device=features.device)
        
        return torch.cat(all_roi_features, dim=0)