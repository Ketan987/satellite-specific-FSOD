"""
COCO dataset utilities for FSOD
"""

import json
import os
from PIL import Image
import numpy as np
from collections import defaultdict


class COCODataset:
    """COCO-style dataset loader for FSOD"""
    
    def __init__(self, json_path, image_dir, allowed_formats=['.jpg', '.jpeg']):
        self.json_path = json_path
        self.image_dir = image_dir
        self.allowed_formats = [fmt.lower() for fmt in allowed_formats]
        
        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build indices
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id and category_id
        self.img_to_anns = defaultdict(list)
        self.cat_to_anns = defaultdict(list)
        
        for ann in self.coco_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
            self.cat_to_anns[ann['category_id']].append(ann)
        
        # Filter images with valid format
        self.valid_images = self._filter_valid_images()
        
        print(f"Loaded {len(self.valid_images)} valid images")
        print(f"Total categories: {len(self.categories)}")
    
    def _filter_valid_images(self):
        """Filter images with allowed formats"""
        valid = []
        for img_id, img_info in self.images.items():
            file_name = img_info['file_name']
            ext = os.path.splitext(file_name)[1].lower()
            if ext in self.allowed_formats:
                img_path = os.path.join(self.image_dir, file_name)
                if os.path.exists(img_path):
                    valid.append(img_id)
        return valid
    
    def get_image(self, image_id, num_channels=3):
        """Load image by ID and convert to the requested number of channels"""
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        if not self._is_valid_format(img_path):
            raise ValueError(f"Unsupported image format: {img_path}")

        image = Image.open(img_path)
        desired_mode = self._pil_mode_from_channels(num_channels)
        if desired_mode is not None:
            image = image.convert(desired_mode)
        return image, img_info
    
    def _is_valid_format(self, img_path):
        ext = os.path.splitext(img_path)[1].lower()
        return ext in self.allowed_formats

    @staticmethod
    def _pil_mode_from_channels(num_channels):
        if num_channels == 1:
            return 'L'
        if num_channels == 3:
            return 'RGB'
        if num_channels == 4:
            return 'RGBA'
        return None
    
    def get_annotations(self, image_id):
        """Get annotations for an image"""
        return self.img_to_anns[image_id]
    
    def get_category_images(self, category_id, num_samples=None):
        """Get images containing a specific category"""
        anns = self.cat_to_anns[category_id]
        
        # Get unique image IDs
        img_ids = list(set([ann['image_id'] for ann in anns]))
        
        # Filter valid images
        img_ids = [iid for iid in img_ids if iid in self.valid_images]
        
        if num_samples and len(img_ids) > num_samples:
            img_ids = np.random.choice(img_ids, num_samples, replace=False)
        
        return img_ids
    
    def sample_episode(self, n_way, k_shot, query_samples):
        """
        Sample an N-way K-shot episode
        Returns: support_set, query_set
        """
        # Select categories that have at least one image
        available_cats = [cat_id for cat_id in self.cat_to_anns.keys() 
                         if len(self.get_category_images(cat_id)) >= 1]

        if len(available_cats) == 0:
            raise ValueError("No categories with valid images found in dataset.")

        # If there are fewer categories than n_way, allow sampling with replacement
        replace_cats = len(available_cats) < n_way
        selected_cats = np.random.choice(available_cats, n_way, replace=replace_cats)

        support_set = []
        query_set = []

        for cat_id in selected_cats:
            # Get images for this category
            img_ids = self.get_category_images(cat_id)

            # If not enough images for unique sampling, allow sampling with replacement
            need = k_shot + query_samples
            replace_images = len(img_ids) < need
            sampled_ids = np.random.choice(img_ids, need, replace=replace_images)

            # Split into support and query
            support_ids = sampled_ids[:k_shot]
            query_ids = sampled_ids[k_shot:]

            for img_id in support_ids:
                support_set.append({
                    'image_id': int(img_id),
                    'category_id': int(cat_id),
                    'annotations': self.get_annotations(img_id)
                })

            for img_id in query_ids:
                query_set.append({
                    'image_id': int(img_id),
                    'category_id': int(cat_id),
                    'annotations': self.get_annotations(img_id)
                })

        return support_set, query_set, selected_cats


def coco_to_xyxy(bbox):
    """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_coco(bbox):
    """Convert [x1, y1, x2, y2] to COCO bbox [x, y, w, h]"""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area