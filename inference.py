"""
Inference script for FSOD with batch and single image modes
Direct support images via CLI - no JSON config required
"""

import torch
import os
import argparse
import csv
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Memory optimization for Kaggle GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

from config import Config
from models.detector import FSODDetector
from utils.data_loader import prepare_inference_data


class FSODInference:
    """FSOD Inference wrapper with batch and single modes"""
    
    def __init__(self, model_path, device='cpu', multi_gpu=True):
        self.config = Config()
        
        # Determine device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')  # Primary GPU
            self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        else:
            self.device = torch.device('cpu')
            self.multi_gpu = False
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = FSODDetector(
            feature_dim=self.config.FEATURE_DIM,
            embed_dim=self.config.EMBEDDING_DIM,
            image_size=self.config.IMAGE_SIZE,
            pretrained=False
        ).to(self.device)
        
        # Load weights - handle both checkpoint dicts and raw state dicts
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Note: Multi-GPU with DataParallel doesn't work well with mixed tensor/list data structures
        # Used in FSOD (support_boxes is a list). Use single GPU for stability.
        if self.multi_gpu:
            print(f"⚠️  Found {torch.cuda.device_count()} GPUs but DataParallel has issues with mixed data types")
            print(f"   Using single GPU inference for stability (GPU 0)")
            self.multi_gpu = False
        
        self.model.eval()
        print(f"Model loaded successfully! Device: {self.device}")
    
    def validate_images(self, image_paths):
        """Validate that all images exist and are readable"""
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            try:
                Image.open(path)
            except Exception as e:
                raise ValueError(f"Cannot open image {path}: {e}")
    
    def visualize_detections(self, image_path, detections, output_path):
        """
        Draw bounding boxes with similarity scores on image
        """
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
        ]
        
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            score = det['similarity_score']
            class_name = det.get('class_name', 'object')
            
            # bbox format: [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox
            
            # Draw box
            color = colors[idx % len(colors)]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            
            # Draw label
            label = f"{class_name}: {score:.3f}"
            draw.text((x_min, y_min - 20), label, fill=color, font=font)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    def detect(self, support_images, query_image, 
               score_threshold=None, nms_threshold=None, max_detections=None):
        """
        Run detection on query image given support images
        
        Args:
            support_images: List of dicts with 'image_path' and 'class_name' keys
                           OR list of string paths (defaults to 'object' class)
            query_image: Path to query image
            score_threshold: Detection confidence threshold
            nms_threshold: NMS threshold
            max_detections: Maximum detections per image
            
        Returns:
            List of detections with bbox in [x_min, y_min, x_max, y_max] format and class_name
        """
        # Set defaults
        if score_threshold is None:
            score_threshold = self.config.SCORE_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.config.NMS_THRESHOLD
        if max_detections is None:
            max_detections = self.config.MAX_DETECTIONS

        # Parse support images - handle both list of paths and list of dicts
        support_img_paths = []
        support_classes = []
        
        for support_item in support_images:
            if isinstance(support_item, dict):
                # Dict format: {'image_path': '...', 'class_name': '...'}
                support_img_paths.append(support_item['image_path'])
                support_classes.append(support_item.get('class_name', 'object'))
            elif isinstance(support_item, str):
                # Simple path string
                support_img_paths.append(support_item)
                support_classes.append('object')  # Default class name
            else:
                raise ValueError(f"Invalid support image format: {support_item}")

        # Validate images
        self.validate_images(support_img_paths + [query_image])

        # Load support images
        support_imgs = []
        support_bboxes = []  # Auto-detected full image as bbox
        
        for img_path in support_img_paths:
            img = Image.open(img_path).convert('RGB')
            support_imgs.append(img)
            # Use full image as bounding box [x, y, w, h]
            w, h = img.size
            support_bboxes.append([0, 0, w, h])
        
        # Prepare data
        support_tensors, query_tensor = prepare_inference_data(
            support_imgs,
            query_image,
            self.config.IMAGE_SIZE
        )
        
        support_tensors = support_tensors.to(self.device)
        query_tensor = query_tensor.to(self.device)

        # Prepare support boxes as list of tensors
        support_boxes_list = [
            torch.tensor([b], dtype=torch.float32, device=self.device)
            for b in support_bboxes
        ]

        # Map class names to integer labels
        class_to_idx = {}
        support_labels_list = []
        for class_name in support_classes:
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            support_labels_list.append(class_to_idx[class_name])
        
        support_labels = torch.tensor(support_labels_list, dtype=torch.long, device=self.device)
        n_way = len(class_to_idx)

        # Run detection
        with torch.no_grad():
            pred_boxes, pred_scores, pred_classes = self.model.predict(
                support_tensors,
                support_boxes_list,
                support_labels,
                query_tensor,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections,
                n_way=n_way
            )

        # Create reverse mapping from index to class name
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Convert to output format [x_min, y_min, x_max, y_max]
        detections = []
        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            score = pred_scores[i]
            pred_class_idx = int(pred_classes[i].item()) if pred_classes is not None else 0
            
            # Safety: Validate class index is in valid range
            if pred_class_idx < 0 or pred_class_idx >= n_way:
                pred_class_idx = 0
            
            pred_class_name = idx_to_class.get(pred_class_idx, 'object')
            
            # Convert box format [x, y, w, h] to [x_min, y_min, x_max, y_max]
            box_np = box.cpu().numpy().astype(float)
            x, y, w, h = box_np
            detection = {
                'bbox': [float(x), float(y), float(x + w), float(y + h)],
                'similarity_score': float(score.cpu().numpy()),
                'class_name': pred_class_name
            }
            detections.append(detection)

        return detections
    
    def detect_batch(self, support_images, query_images_dir, score_threshold=None, 
                    nms_threshold=None, max_detections=None, output_csv='batch_results.csv'):
        """
        BATCH MODE: Run detection on multiple query images from a directory
        
        Args:
            support_images: List of support image dicts or paths
            query_images_dir: Directory with query images
            output_csv: Output CSV filename
        """
        print(f"\n{'='*60}")
        print("BATCH MODE: Running detection on multiple images")
        print(f"{'='*60}\n")
        
        # Get query images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        query_images = []
        for ext in image_extensions:
            query_images.extend(sorted(Path(query_images_dir).glob(f'*{ext}')))
        
        if not query_images:
            print(f"❌ No query images found in {query_images_dir}")
            return
        
        query_images = [str(p) for p in query_images]
        print(f"✅ Found {len(query_images)} query images")
        print(f"✅ Using {len(support_images)} support images\n")
        
        # Create output directory and CSV
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        
        # Prepare CSV file
        csv_rows = []
        
        # Run detection on each image
        for query_img_path in tqdm(query_images, desc="Batch Detection"):
            try:
                detections = self.detect(
                    support_images,
                    query_img_path,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections
                )
                
                # Extract filename
                filename = os.path.basename(query_img_path)
                
                # Add to CSV rows
                for det in detections:
                    x_min, y_min, x_max, y_max = det['bbox']
                    score = det['similarity_score']
                    class_name = det.get('class_name', 'object')
                    csv_rows.append({
                        'x_min': round(x_min, 2),
                        'y_min': round(y_min, 2),
                        'x_max': round(x_max, 2),
                        'y_max': round(y_max, 2),
                        'filename': filename,
                        'class_name': class_name,
                        'similarity_score': round(score, 4)
                    })
            except Exception as e:
                print(f"⚠️  Error processing {query_img_path}: {e}")
                continue
        
        # Write CSV
        if csv_rows:
            fieldnames = ['x_min', 'y_min', 'x_max', 'y_max', 'filename', 'class_name', 'similarity_score']
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"\n✅ Batch processing completed!")
            print(f"📊 Total detections: {len(csv_rows)}")
            print(f"💾 Results saved to: {output_csv}")
        else:
            print("⚠️  No detections found")
    
    def detect_single(self, support_images, query_image, output_dir='output', 
                     score_threshold=None, nms_threshold=None, max_detections=None):
        """
        SINGLE MODE: Run detection on a single query image with visualization
        
        Args:
            support_images: List of support image dicts or paths
            query_image: Path to single query image
            output_dir: Output directory for annotated image
        """
        print(f"\n{'='*60}")
        print("SINGLE MODE: Running detection on single query image")
        print(f"{'='*60}\n")
        
        print(f"✅ Using {len(support_images)} support images")
        
        # Run detection
        detections = self.detect(
            support_images,
            query_image,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections
        )
        
        # Print results
        print(f"\nQuery Image: {query_image}")
        print(f"Number of objects detected: {len(detections)}\n")
        
        if detections:
            print("Detection Results:")
            print("-" * 100)
            print(f"{'#':<4} {'x_min':<12} {'y_min':<12} {'x_max':<12} {'y_max':<12} {'Score':<8} {'Class':<15}")
            print("-" * 100)
            
            for idx, det in enumerate(detections, 1):
                x_min, y_min, x_max, y_max = det['bbox']
                score = det['similarity_score']
                class_name = det.get('class_name', 'object')
                print(f"{idx:<4} {x_min:<12.1f} {y_min:<12.1f} {x_max:<12.1f} {y_max:<12.1f} {score:<8.4f} {class_name:<15}")
            
            print("-" * 100)
        else:
            print("⚠️  No objects detected!")
        
        # Visualize and save
        output_filename = os.path.basename(query_image)
        output_filename = os.path.splitext(output_filename)[0] + '_detected.jpg'
        output_path = os.path.join(output_dir, output_filename)
        
        self.visualize_detections(query_image, detections, output_path)
        
        # Print summary
        print(f"\n✅ Detection completed!")
        print(f"📸 Annotated image saved to: {output_path}\n")


def build_support_set(args):
    """Build support set from either config JSON or CLI arguments"""
    
    # Method 1: Load from support config JSON
    if args.support_config:
        with open(args.support_config, 'r') as f:
            support_set = json.load(f)
        print(f"✅ Loaded support set from: {args.support_config}")
        return support_set
    
    # Method 2: Direct command-line support images
    if args.support_img:
        support_set = []
        
        # Handle multiple support images with optional classes
        for i, img_path in enumerate(args.support_img):
            # Check if we have corresponding class names
            if args.support_classes and i < len(args.support_classes):
                class_name = args.support_classes[i]
            else:
                class_name = 'object'
            
            support_set.append({
                'image_path': img_path,
                'class_name': class_name
            })
        
        print(f"✅ Using {len(support_set)} support images from CLI:")
        for item in support_set:
            print(f"   - {item['image_path']} (class: {item['class_name']})")
        return support_set
    
    raise ValueError("Either --support_config or --support_img is required!")


def main(args):
    """Main inference function"""
    
    # Initialize inference with multi-GPU support
    inferencer = FSODInference(args.model_path, args.device, multi_gpu=not args.no_multi_gpu)
    
    # Build support set
    support_set = build_support_set(args)
    
    # BATCH MODE
    if args.mode == 'batch':
        if not args.query_dir:
            print("❌ Error: --query_dir required for batch mode")
            return
        
        inferencer.detect_batch(
            support_set,
            args.query_dir,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
            output_csv=args.output_csv
        )
    
    # SINGLE MODE
    elif args.mode == 'single':
        if not args.query_image:
            print("❌ Error: --query_image required for single mode")
            return
        
        inferencer.detect_single(
            support_set,
            args.query_image,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections
        )
    
    else:
        print(f"❌ Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSOD Inference - Direct Support Images (No JSON Config Required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

1. BATCH MODE with direct support images (multiple classes):
   python inference.py --mode batch \\
     --model_path checkpoints/best_model.pth \\
     --support_img cat1.jpg cat2.jpg dog1.jpg dog2.jpg \\
     --support_classes cat cat dog dog \\
     --query_dir data/test_images/ \\
     --output_csv results.csv \\
     --device cpu

2. BATCH MODE with direct support images (single class):
   python inference.py --mode batch \\
     --model_path checkpoints/best_model.pth \\
     --support_img car1.jpg car2.jpg car3.jpg \\
     --query_dir data/test_images/ \\
     --output_csv car_detections.csv \\
     --device cpu

3. SINGLE MODE with direct support images:
   python inference.py --mode single \\
     --model_path checkpoints/best_model.pth \\
     --support_img bird1.jpg bird2.jpg bird3.jpg \\
     --query_image test_image.jpg \\
     --output_dir output/ \\
     --device cpu

4. Using support config JSON (alternative method):
   python inference.py --mode batch \\
     --model_path checkpoints/best_model.pth \\
     --support_config support_set.json \\
     --query_dir data/test_images/ \\
     --output_csv results.csv \\
     --device cpu
        """
    )
    
    # Common arguments
    parser.add_argument('--mode', type=str, choices=['batch', 'single'], required=True,
                       help="Inference mode: 'batch' for multiple images, 'single' for one image")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help="Device to use (default: cpu)")
    parser.add_argument('--no_multi_gpu', action='store_true',
                       help="Disable multi-GPU inference (default: enabled if available)")
    
    # Support set arguments - CHOOSE ONE METHOD
    support_group = parser.add_argument_group('Support Set (choose one method)')
    support_group.add_argument('--support_config', type=str, default=None,
                              help="[METHOD 1] Path to JSON file with support set config")
    support_group.add_argument('--support_img', type=str, nargs='+', default=None,
                              help="[METHOD 2] List of support image paths (e.g., img1.jpg img2.jpg img3.jpg)")
    support_group.add_argument('--support_classes', type=str, nargs='+', default=None,
                              help="[METHOD 2] Class names for support images (optional, defaults to 'object')")
    
    # Query arguments
    query_group = parser.add_argument_group('Query Images')
    query_group.add_argument('--query_dir', type=str, default=None,
                            help="Directory with query images (required for batch mode)")
    query_group.add_argument('--query_image', type=str, default=None,
                            help="Path to single query image (required for single mode)")
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_csv', type=str, default='batch_results.csv',
                             help="Output CSV file for batch mode (default: batch_results.csv)")
    output_group.add_argument('--output_dir', type=str, default='output',
                             help="Output directory for single mode visualizations (default: output)")
    
    # Detection parameters
    det_group = parser.add_argument_group('Detection Parameters')
    det_group.add_argument('--score_threshold', type=float, default=0.3,
                          help="Detection score threshold (default: 0.3)")
    det_group.add_argument('--nms_threshold', type=float, default=0.4,
                          help="NMS threshold (default: 0.4)")
    det_group.add_argument('--max_detections', type=int, default=100,
                          help="Maximum number of detections per image (default: 100)")
    
    args = parser.parse_args()
    main(args)
