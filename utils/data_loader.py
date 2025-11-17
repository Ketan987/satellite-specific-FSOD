"""
Data loader for FSOD training with multi-band image support (3-band RGB and 4-band TIFF)
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
import rasterio
from typing import Union, Tuple, List


class FSODDataset(Dataset):
    """Few-Shot Object Detection Dataset with multi-band support"""
    
    def __init__(self, coco_dataset, n_way, k_shot, query_samples, 
                 image_size=512, num_episodes=1000):
        self.coco_dataset = coco_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_samples = query_samples
        self.image_size = image_size
        self.num_episodes = num_episodes
    
    def _get_image_transforms(self, num_channels):
        """Get appropriate transforms based on number of channels"""
        if num_channels == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif num_channels == 4:
            mean = [0.485, 0.456, 0.406, 0.406]
            std = [0.229, 0.224, 0.225, 0.225]
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")
        
        return {
            'train': T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]),
            'val': T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        }
    
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
        
        # Determine number of channels from first image
        first_img_path = self.coco_dataset.get_image_path(support_data[0]['image_id'])
        num_channels = self.coco_dataset._get_num_channels(first_img_path)
        
        transforms = self._get_image_transforms(num_channels)
        
        # Process support set
        support_images = []
        support_boxes = []
        support_labels = []
        
        cat_to_label = {cat_id: i for i, cat_id in enumerate(selected_cats)}
        
        for item in support_data:
            img, img_info = self.coco_dataset.get_image(item['image_id'])
            orig_w, orig_h = img.size
            
            # Transform image
            img_tensor = transforms['train'](img)
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
            img_tensor = transforms['val'](img)
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
            'support_images': torch.stack(support_images),  # [N*K, C, H, W]
            'support_boxes': support_boxes,  # List of [num_boxes, 4]
            'support_labels': torch.tensor(support_labels, dtype=torch.long),  # [N*K]
            'query_images': torch.stack(query_images),  # [Q, C, H, W]
            'query_boxes': query_boxes,  # List of [num_boxes, 4]
            'query_labels': query_labels,  # List of [num_boxes]
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'num_channels': num_channels
        }


def collate_fn(batch):
    """Custom collate function for batching episodes"""
    # For simplicity, return single episode (batch_size=1 for episodes)
    return batch[0]


def prepare_inference_data(support_images, query_image, image_size=512):
    """
    Prepare data for inference with multi-band support
    
    Args:
        support_images: List of PIL Images, paths, or tuples of (image, num_channels)
        query_image: PIL Image, path, or tuple of (image, num_channels)
        image_size: Target size
    
    Returns:
        support_tensors, query_tensor, num_channels
    """
    
    def load_and_normalize_image(img_input):
        """Load image and return tensor with proper normalization based on channels"""
        # Load image if it's a path
        if isinstance(img_input, str):
            img, num_channels = _load_image_from_path(img_input)
        elif isinstance(img_input, tuple):
            img, num_channels = img_input
        else:
            img = img_input
            num_channels = 3 if img.mode == 'RGB' else 4
        
        # Get transforms based on channels
        if num_channels == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif num_channels == 4:
            mean = [0.485, 0.456, 0.406, 0.406]
            std = [0.229, 0.224, 0.225, 0.225]
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")
        
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        return transform(img), num_channels
    
    # Load and transform support images
    support_tensors = []
    num_channels = 3  # Default
    
    for img_input in support_images:
        tensor, nc = load_and_normalize_image(img_input)
        support_tensors.append(tensor)
        num_channels = nc  # Use channels from data
    
    # Load and transform query image
    query_tensor, num_channels = load_and_normalize_image(query_image)
    
    return torch.stack(support_tensors), query_tensor.unsqueeze(0), num_channels


def _load_image_from_path(image_path: str) -> Tuple[Image.Image, int]:
    """
    Load image from path, automatically detecting 3-band or 4-band
    
    Args:
        image_path: Path to image file (.jpg/.png for 3-band, .tif/.tiff for 4-band)
    
    Returns:
        Tuple of (PIL Image, num_channels)
    """
    file_ext = image_path.lower().split('.')[-1]
    
    if file_ext in ['tif', 'tiff']:
        # Load 4-band TIF
        with rasterio.open(image_path) as src:
            data = src.read()  # [bands, height, width]
            
            if data.shape[0] == 4:
                # RGBN format
                img_array = np.transpose(data[:4], (1, 2, 0))  # [height, width, 4]
            else:
                raise ValueError(f"TIF must have 4 bands, got {data.shape[0]}")
            
            # Convert to 0-255 range if needed
            if img_array.max() > 255:
                # Assuming 16-bit or higher - normalize to 0-255
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            # Create PIL Image from RGBA
            img = Image.fromarray(img_array, mode='RGBA')
            return img, 4
    else:
        # Load 3-band JPG/PNG
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise ValueError(f"Supported formats: jpg, png, tif. Got: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        return img, 3