"""
Data loader for FSOD training
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image


class FSODDataset(Dataset):
    """Few-Shot Object Detection Dataset"""
    
    def __init__(self, coco_dataset, n_way, k_shot, query_samples, 
                 image_size=512, num_episodes=1000):
        self.coco_dataset = coco_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_samples = query_samples
        self.image_size = image_size
        self.num_episodes = num_episodes
        
        # Image transforms with augmentation for training
        # Augmentation helps with few-shot learning and overfitting
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Deterministic transform for validation
        self.transform_val = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.num_episodes
    
    def __getitem__(self, idx):
        """
        Sample an episode and return support and query sets
        """
        # Sample episode
        support_data, query_data, selected_cats = self.coco_dataset.sample_episode(
            self.n_way, self.k_shot, self.query_samples
        )
        
        # Process support set
        support_images = []
        support_boxes = []
        support_labels = []
        
        cat_to_label = {cat_id: i for i, cat_id in enumerate(selected_cats)}
        
        for item in support_data:
            img, img_info = self.coco_dataset.get_image(item['image_id'])
            orig_w, orig_h = img.size
            
            # Transform image
            img_tensor = self.transform(img)
            support_images.append(img_tensor)
            
            # Get boxes for the target category
            boxes = []
            for ann in item['annotations']:
                if ann['category_id'] == item['category_id']:
                    bbox = ann['bbox']  # [x, y, w, h]
                    # Skip invalid boxes (width or height < 1 pixel)
                    if bbox[2] < 1 or bbox[3] < 1:
                        continue
                    # Scale to new size
                    x = bbox[0] * self.image_size / orig_w
                    y = bbox[1] * self.image_size / orig_h
                    w = bbox[2] * self.image_size / orig_w
                    h = bbox[3] * self.image_size / orig_h
                    # Ensure minimum size after scaling
                    w = max(2.0, w)
                    h = max(2.0, h)
                    boxes.append([x, y, w, h])
            
            if len(boxes) == 0:
                boxes = [[self.image_size // 4, self.image_size // 4, self.image_size // 2, self.image_size // 2]]  # Center box
            
            support_boxes.append(torch.tensor(boxes, dtype=torch.float32))
            support_labels.append(cat_to_label[item['category_id']])
        
        # Process query set
        query_images = []
        query_boxes = []
        query_labels = []
        
        for item in query_data:
            img, img_info = self.coco_dataset.get_image(item['image_id'])
            orig_w, orig_h = img.size
            
            # Transform image
            img_tensor = self.transform(img)
            query_images.append(img_tensor)
            
            # Get all boxes and labels
            boxes = []
            labels = []
            for ann in item['annotations']:
                if ann['category_id'] in selected_cats:
                    bbox = ann['bbox']
                    # Skip invalid boxes (width or height < 1 pixel)
                    if bbox[2] < 1 or bbox[3] < 1:
                        continue
                    # Scale to new size
                    x = bbox[0] * self.image_size / orig_w
                    y = bbox[1] * self.image_size / orig_h
                    w = bbox[2] * self.image_size / orig_w
                    h = bbox[3] * self.image_size / orig_h
                    # Ensure minimum size after scaling
                    w = max(2.0, w)
                    h = max(2.0, h)
                    boxes.append([x, y, w, h])
                    labels.append(cat_to_label[ann['category_id']])
            
            if len(boxes) == 0:
                boxes = [[self.image_size // 4, self.image_size // 4, self.image_size // 2, self.image_size // 2]]
                labels = [0]
            
            query_boxes.append(torch.tensor(boxes, dtype=torch.float32))
            query_labels.append(torch.tensor(labels, dtype=torch.long))
        
        return {
            'support_images': torch.stack(support_images),  # [N*K, 3, H, W]
            'support_boxes': support_boxes,  # List of [num_boxes, 4]
            'support_labels': torch.tensor(support_labels, dtype=torch.long),  # [N*K]
            'query_images': torch.stack(query_images),  # [Q, 3, H, W]
            'query_boxes': query_boxes,  # List of [num_boxes, 4]
            'query_labels': query_labels,  # List of [num_boxes]
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }


def collate_fn(batch):
    """Custom collate function for batching episodes"""
    # For simplicity, return single episode (batch_size=1 for episodes)
    return batch[0]


def prepare_inference_data(support_images, query_image, image_size=512):
    """
    Prepare data for inference
    
    Args:
        support_images: List of PIL Images or paths
        query_image: PIL Image or path
        image_size: Target size
    
    Returns:
        support_tensors, query_tensor
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform support images
    support_tensors = []
    for img in support_images:
        if isinstance(img, str):
            # Validate JPEG
            if not img.lower().endswith(('.jpg', '.jpeg')):
                raise ValueError(f"Only JPEG images allowed. Got: {img}")
            img = Image.open(img).convert('RGB')
        support_tensors.append(transform(img))
    
    # Load and transform query image
    if isinstance(query_image, str):
        if not query_image.lower().endswith(('.jpg', '.jpeg')):
            raise ValueError(f"Only JPEG images allowed. Got: {query_image}")
        query_image = Image.open(query_image).convert('RGB')
    
    query_tensor = transform(query_image)
    
    return torch.stack(support_tensors), query_tensor.unsqueeze(0)