"""
Inference script for FSOD
"""

import torch
import json
import os
import argparse
from PIL import Image
import numpy as np

from config import Config
from models.detector import FSODDetector
from utils.data_loader import prepare_inference_data


class FSODInference:
    """FSOD Inference wrapper"""
    
    def __init__(self, model_path, device='cuda'):
        self.config = Config()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = FSODDetector(
            feature_dim=self.config.FEATURE_DIM,
            embed_dim=self.config.EMBEDDING_DIM,
            image_size=self.config.IMAGE_SIZE
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def validate_images(self, image_paths):
        """Validate that all images are JPEG format"""
        for path in image_paths:
            if not path.lower().endswith(('.jpg', '.jpeg')):
                raise ValueError(f"Only JPEG images are allowed. Invalid file: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
    
    def detect(self, support_set, query_image, 
               score_threshold=None, nms_threshold=None, max_detections=None):
        """
        Run detection on query image given support set
        
        Args:
            support_set: List of dicts with 'image_path', 'bbox', 'class_name'
                Example: [
                    {'image_path': 'cat1.jpg', 'bbox': [x, y, w, h], 'class_name': 'cat'},
                    {'image_path': 'cat2.jpg', 'bbox': [x, y, w, h], 'class_name': 'cat'},
                    ...
                ]
            query_image: Path to query image (JPEG)
            score_threshold: Minimum similarity score (default from config)
            nms_threshold: NMS threshold (default from config)
            max_detections: Maximum detections (default from config)
        
        Returns:
            detections: List of detection dicts
        """
        # Set defaults
        if score_threshold is None:
            score_threshold = self.config.SCORE_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.config.NMS_THRESHOLD
        if max_detections is None:
            max_detections = self.config.MAX_DETECTIONS
        
        # Validate images
        support_paths = [item['image_path'] for item in support_set]
        self.validate_images(support_paths + [query_image])
        
        # Load support images
        support_images = []
        support_boxes = []
        support_classes = []
        
        for item in support_set:
            img = Image.open(item['image_path']).convert('RGB')
            support_images.append(img)
            support_boxes.append(item['bbox'])
            support_classes.append(item['class_name'])
        
        # Prepare data
        support_tensors, query_tensor = prepare_inference_data(
            support_images,
            query_image,
            self.config.IMAGE_SIZE
        )
        
        support_tensors = support_tensors.to(self.device)
        query_tensor = query_tensor.to(self.device)
        
        # Prepare support boxes as list
        support_boxes_list = [torch.tensor([support_boxes], dtype=torch.float32).to(self.device)]
        
        # Run detection
        with torch.no_grad():
            pred_boxes, pred_scores = self.model.predict(
                support_tensors,
                support_boxes_list,
                query_tensor,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections
            )
        
        # Convert to output format
        detections = []
        for box, score in zip(pred_boxes, pred_scores):
            detection = {
                'bbox': box.cpu().numpy().tolist(),  # [x, y, w, h]
                'similarity_score': float(score.cpu().numpy()),
                'class_name': support_classes[0] if support_classes else 'object'
            }
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, support_set, query_images, **kwargs):
        """Run detection on multiple query images"""
        results = []
        for query_img in query_images:
            detections = self.detect(support_set, query_img, **kwargs)
            results.append({
                'image_path': query_img,
                'detections': detections
            })
        return results
    
    def save_results(self, results, output_path):
        """Save detection results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


def main(args):
    # Initialize inference
    inferencer = FSODInference(args.model_path, args.device)
    
    # Example support set (you need to modify this)
    support_set = [
        {
            'image_path': args.support_image1,
            'bbox': args.support_bbox1,  # [x, y, w, h]
            'class_name': args.class_name
        },
        # Add more support examples...
    ]
    
    # Run detection
    print(f"Running detection on {args.query_image}...")
    detections = inferencer.detect(
        support_set,
        args.query_image,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections
    )
    
    # Print results
    print(f"\nDetected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. bbox: {det['bbox']}, score: {det['similarity_score']:.3f}")
    
    # Save results
    if args.output:
        results = {
            'image_path': args.query_image,
            'detections': detections
        }
        inferencer.save_results(results, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSOD Inference")
    parser.add_argument('--model_path', type=str, required=True, 
                       help="Path to trained model")
    parser.add_argument('--query_image', type=str, required=True,
                       help="Path to query image (JPEG)")
    parser.add_argument('--support_image1', type=str, required=True,
                       help="Path to support image 1 (JPEG)")
    parser.add_argument('--support_bbox1', type=float, nargs=4, required=True,
                       help="Support bbox 1 [x y w h]")
    parser.add_argument('--class_name', type=str, default='object',
                       help="Class name")
    parser.add_argument('--score_threshold', type=float, default=0.3,
                       help="Score threshold")
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                       help="NMS threshold")
    parser.add_argument('--max_detections', type=int, default=100,
                       help="Maximum detections")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device (cuda/cpu)")
    parser.add_argument('--output', type=str, default='results.json',
                       help="Output JSON file")
    
    args = parser.parse_args()
    main(args)