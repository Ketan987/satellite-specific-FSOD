"""
Inference script for FSOD with batch and single image modes
Simplified version: Support set is just images of objects
"""

import torch
import os
import argparse
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import Config
from models.detector import FSODDetector
from utils.data_loader import prepare_inference_data


class FSODInference:
    """FSOD Inference wrapper with batch and single modes"""
    
    def __init__(self, model_path, device='cpu'):
        self.config = Config()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = FSODDetector(
            feature_dim=self.config.FEATURE_DIM,
            embed_dim=self.config.EMBEDDING_DIM,
            image_size=self.config.IMAGE_SIZE,
            pretrained=False
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
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
            
            # bbox format: [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox
            
            # Draw box
            color = colors[idx % len(colors)]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            
            # Draw label
            label = f"Score: {score:.3f}"
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
            support_images: List of paths to support/example images
            query_image: Path to query image
            score_threshold: Detection confidence threshold
            nms_threshold: NMS threshold
            max_detections: Maximum detections per image
            
        Returns:
            List of detections with bbox in [x_min, y_min, x_max, y_max] format
        """
        # Set defaults
        if score_threshold is None:
            score_threshold = self.config.SCORE_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.config.NMS_THRESHOLD
        if max_detections is None:
            max_detections = self.config.MAX_DETECTIONS

        # Validate images
        self.validate_images(support_images + [query_image])

        # Load support images
        support_imgs = []
        support_bboxes = []  # Auto-detected full image as bbox
        
        for img_path in support_images:
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
            torch.tensor([b], dtype=torch.float32).to(self.device)
            for b in support_bboxes
        ]

        # All support images are same class (class 0)
        support_labels = torch.zeros(len(support_images), dtype=torch.long).to(self.device)

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
                n_way=1  # Single class
            )

        # Convert to output format [x_min, y_min, x_max, y_max]
        detections = []
        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            score = pred_scores[i]
            
            # Convert box format [x, y, w, h] to [x_min, y_min, x_max, y_max]
            box_np = box.cpu().numpy().astype(float)
            x, y, w, h = box_np
            detection = {
                'bbox': [float(x), float(y), float(x + w), float(y + h)],
                'similarity_score': float(score.cpu().numpy())
            }
            detections.append(detection)

        return detections
    
    def detect_batch(self, support_images_dir, query_images_dir, score_threshold=None, 
                    nms_threshold=None, max_detections=None, output_csv='batch_results.csv'):
        """
        BATCH MODE: Run detection on multiple query images
        
        Args:
            support_images_dir: Directory with support/example images
            query_images_dir: Directory with query images
            output_csv: Output CSV filename
        """
        print(f"\n{'='*60}")
        print("BATCH MODE: Running detection on multiple images")
        print(f"{'='*60}\n")
        
        # Get support images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        support_images = []
        for ext in image_extensions:
            support_images.extend(sorted(Path(support_images_dir).glob(f'*{ext}')))
        
        if not support_images:
            print(f"❌ No support images found in {support_images_dir}")
            return
        
        support_images = [str(p) for p in support_images]
        print(f"✅ Loaded {len(support_images)} support images")
        
        # Get query images
        query_images = []
        for ext in image_extensions:
            query_images.extend(sorted(Path(query_images_dir).glob(f'*{ext}')))
        
        if not query_images:
            print(f"❌ No query images found in {query_images_dir}")
            return
        
        print(f"✅ Found {len(query_images)} query images")
        
        # Create output directory and CSV
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        
        # Prepare CSV file
        csv_rows = []
        
        # Run detection on each image
        for query_img_path in tqdm(query_images, desc="Batch Detection"):
            try:
                detections = self.detect(
                    support_images,
                    str(query_img_path),
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
                    csv_rows.append({
                        'x_min': round(x_min, 2),
                        'y_min': round(y_min, 2),
                        'x_max': round(x_max, 2),
                        'y_max': round(y_max, 2),
                        'filename': filename,
                        'similarity_score': round(score, 4)
                    })
            except Exception as e:
                print(f"⚠️  Error processing {query_img_path}: {e}")
                continue
        
        # Write CSV
        if csv_rows:
            fieldnames = ['x_min', 'y_min', 'x_max', 'y_max', 'filename', 'similarity_score']
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"\n✅ Batch processing completed!")
            print(f"📊 Total detections: {len(csv_rows)}")
            print(f"💾 Results saved to: {output_csv}")
        else:
            print("⚠️  No detections found")
    
    def detect_single(self, support_images_dir, query_image, output_dir='output', 
                     score_threshold=None, nms_threshold=None, max_detections=None):
        """
        SINGLE MODE: Run detection on a single query image with visualization
        
        Args:
            support_images_dir: Directory with support/example images
            query_image: Path to single query image
            output_dir: Output directory for annotated image
        """
        print(f"\n{'='*60}")
        print("SINGLE MODE: Running detection on single query image")
        print(f"{'='*60}\n")
        
        # Get support images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        support_images = []
        for ext in image_extensions:
            support_images.extend(sorted(Path(support_images_dir).glob(f'*{ext}')))
        
        if not support_images:
            print(f"❌ No support images found in {support_images_dir}")
            return
        
        support_images = [str(p) for p in support_images]
        print(f"✅ Loaded {len(support_images)} support images")
        
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
            print("-" * 80)
            print(f"{'#':<4} {'x_min':<10} {'y_min':<10} {'x_max':<10} {'y_max':<10} {'Score':<8}")
            print("-" * 80)
            
            for idx, det in enumerate(detections, 1):
                x_min, y_min, x_max, y_max = det['bbox']
                score = det['similarity_score']
                print(f"{idx:<4} {x_min:<10.1f} {y_min:<10.1f} {x_max:<10.1f} {y_max:<10.1f} {score:<8.4f}")
            
            print("-" * 80)
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


def main(args):
    """Main inference function"""
    
    # Initialize inference
    inferencer = FSODInference(args.model_path, args.device)
    
    # BATCH MODE
    if args.mode == 'batch':
        if not args.support_dir or not args.query_dir:
            print("Error: --support_dir and --query_dir required for batch mode")
            return
        
        inferencer.detect_batch(
            args.support_dir,
            args.query_dir,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
            output_csv=args.output_csv
        )
    
    # SINGLE MODE
    elif args.mode == 'single':
        if not args.support_dir or not args.query_image:
            print("Error: --support_dir and --query_image required for single mode")
            return
        
        inferencer.detect_single(
            args.support_dir,
            args.query_image,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections
        )
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSOD Inference with Batch and Single Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

BATCH MODE (process multiple images, save to CSV):
  python inference.py --mode batch --model_path checkpoints/best_model.pth \\
    --support_dir data/support_images/ --query_dir data/test_images/ \\
    --output_csv results.csv --device cpu

SINGLE MODE (process one image, visualize with bboxes):
  python inference.py --mode single --model_path checkpoints/best_model.pth \\
    --support_dir data/support_images/ --query_image data/test.jpg \\
    --output_dir output/ --device cpu
        """
    )
    
    # Common arguments
    parser.add_argument('--mode', type=str, choices=['batch', 'single'], required=True,
                       help="Inference mode: 'batch' for multiple images, 'single' for one image")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help="Device to use")
    
    # Support set argument
    support_group = parser.add_argument_group('Support Set')
    support_group.add_argument('--support_dir', type=str, required=True,
                              help="Directory with support/example images (object images)")
    
    # Query arguments
    query_group = parser.add_argument_group('Query Images')
    query_group.add_argument('--query_dir', type=str, default=None,
                            help="Directory with query images (required for batch mode)")
    query_group.add_argument('--query_image', type=str, default=None,
                            help="Path to single query image (required for single mode)")
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_csv', type=str, default='batch_results.csv',
                             help="Output CSV file for batch mode")
    output_group.add_argument('--output_dir', type=str, default='output',
                             help="Output directory for single mode visualizations")
    
    # Detection parameters
    det_group = parser.add_argument_group('Detection Parameters')
    det_group.add_argument('--score_threshold', type=float, default=0.3,
                          help="Detection score threshold")
    det_group.add_argument('--nms_threshold', type=float, default=0.4,
                          help="NMS threshold")
    det_group.add_argument('--max_detections', type=int, default=100,
                          help="Maximum number of detections")
    
    args = parser.parse_args()
    main(args)
    
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    def detect(self, support_set, query_image, 
               score_threshold=None, nms_threshold=None, max_detections=None):
        """
        Run detection on query image given support set
        Returns: list of detections with bbox in [x_min, y_min, x_max, y_max] format
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

        # Prepare support boxes as list of tensors, one per support image
        import torch as _torch
        support_boxes_list = [
            _torch.tensor([b], dtype=_torch.float32).to(self.device)
            for b in support_boxes
        ]

        # Map support class names to integer labels
        class_to_idx = {}
        support_labels_list = []
        for cname in support_classes:
            if cname not in class_to_idx:
                class_to_idx[cname] = len(class_to_idx)
            support_labels_list.append(class_to_idx[cname])

        support_labels = _torch.tensor(support_labels_list, dtype=_torch.long).to(self.device)

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
                n_way=len(class_to_idx)
            )

        # Convert to output format [x_min, y_min, x_max, y_max]
        detections = []
        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            score = pred_scores[i]
            cls_idx = int(pred_classes[i].item()) if pred_classes is not None else 0
            class_name = None
            
            # reverse mapping
            for k, v in class_to_idx.items():
                if v == cls_idx:
                    class_name = k
                    break

            # Convert box format [x, y, w, h] to [x_min, y_min, x_max, y_max]
            box_np = box.cpu().numpy().astype(float)
            x, y, w, h = box_np
            detection = {
                'bbox': [float(x), float(y), float(x + w), float(y + h)],  # [x_min, y_min, x_max, y_max]
                'similarity_score': float(score.cpu().numpy()),
                'class_name': class_name if class_name is not None else support_classes[0] if support_classes else 'object'
            }
            detections.append(detection)

        return detections
    
    def detect_batch(self, support_set, query_images_dir, score_threshold=None, 
                    nms_threshold=None, max_detections=None, output_csv='batch_results.csv'):
        """
        BATCH MODE: Run detection on multiple query images in a directory
        Saves results to CSV: x_min, y_min, x_max, y_max, filename, similarity_score
        """
        print(f"\n{'='*60}")
        print("BATCH MODE: Running detection on multiple images")
        print(f"{'='*60}")
        
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        query_images = []
        for ext in image_extensions:
            query_images.extend(sorted(Path(query_images_dir).glob(f'*{ext}')))
        
        if not query_images:
            print(f"No images found in {query_images_dir}")
            return
        
        print(f"Found {len(query_images)} query images")
        
        # Create output directory and CSV
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        
        # Prepare CSV file
        csv_rows = []
        
        # Run detection on each image
        for query_img_path in tqdm(query_images, desc="Batch Detection"):
            try:
                detections = self.detect(
                    support_set,
                    str(query_img_path),
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
                    csv_rows.append({
                        'x_min': round(x_min, 2),
                        'y_min': round(y_min, 2),
                        'x_max': round(x_max, 2),
                        'y_max': round(y_max, 2),
                        'filename': filename,
                        'similarity_score': round(score, 4)
                    })
            except Exception as e:
                print(f"Error processing {query_img_path}: {e}")
                continue
        
        # Write CSV
        if csv_rows:
            fieldnames = ['x_min', 'y_min', 'x_max', 'y_max', 'filename', 'similarity_score']
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"\n✅ Batch processing completed!")
            print(f"📊 Total detections: {len(csv_rows)}")
            print(f"💾 Results saved to: {output_csv}")
        else:
            print("No detections found")
    
    def detect_single(self, support_set, query_image, output_dir='output', 
                     score_threshold=None, nms_threshold=None, max_detections=None):
        """
        SINGLE MODE: Run detection on a single query image with visualization
        Prints results and saves annotated image to output folder
        """
        print(f"\n{'='*60}")
        print("SINGLE MODE: Running detection on single query image")
        print(f"{'='*60}")
        
        # Run detection
        detections = self.detect(
            support_set,
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
            print("-" * 80)
            print(f"{'#':<4} {'x_min':<10} {'y_min':<10} {'x_max':<10} {'y_max':<10} {'Score':<8} {'Class':<15}")
            print("-" * 80)
            
            for idx, det in enumerate(detections, 1):
                x_min, y_min, x_max, y_max = det['bbox']
                score = det['similarity_score']
                class_name = det['class_name']
                print(f"{idx:<4} {x_min:<10.1f} {y_min:<10.1f} {x_max:<10.1f} {y_max:<10.1f} {score:<8.4f} {class_name:<15}")
            
            print("-" * 80)
        else:
            print("No objects detected!")
        
        # Visualize and save
        output_filename = os.path.basename(query_image)
        output_filename = os.path.splitext(output_filename)[0] + '_detected.jpg'
        output_path = os.path.join(output_dir, output_filename)
        
        self.visualize_detections(query_image, detections, output_path)
        
        # Print summary
        print(f"\n✅ Detection completed!")
        print(f"📸 Annotated image saved to: {output_path}\n")


def main(args):
    """Main inference function"""
    
    # Initialize inference
    inferencer = FSODInference(args.model_path, args.device)
    
    # Prepare support set from JSON config
    if args.support_config:
        with open(args.support_config, 'r') as f:
            support_set = json.load(f)
        print(f"Loaded support set from {args.support_config}")
    else:
        # Example support set from command line
        support_set = [
            {
                'image_path': args.support_image,
                'bbox': args.support_bbox,  # [x, y, w, h]
                'class_name': args.class_name
            }
        ]
    
    # BATCH MODE
    if args.mode == 'batch':
        if not args.query_dir:
            print("Error: --query_dir required for batch mode")
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
            print("Error: --query_image required for single mode")
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
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSOD Inference with Batch and Single Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

BATCH MODE (process multiple images, save to CSV):
  python inference.py --mode batch --model_path checkpoints/best_model.pth \\
    --query_dir data/test_images/ --support_config support_set.json \\
    --output_csv results.csv --device cpu

SINGLE MODE (process one image, visualize with bboxes):
  python inference.py --mode single --model_path checkpoints/best_model.pth \\
    --query_image data/test_image.jpg --support_image data/support1.jpg \\
    --support_bbox 10 20 50 80 --class_name cat --output_dir output/ --device cpu
        """
    )
    
    # Common arguments
    parser.add_argument('--mode', type=str, choices=['batch', 'single'], required=True,
                       help="Inference mode: 'batch' for multiple images, 'single' for one image")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help="Device to use")
    
    # Support set arguments
    support_group = parser.add_argument_group('Support Set')
    support_group.add_argument('--support_config', type=str, default=None,
                              help="Path to JSON file with support set (recommended for batch mode)")
    support_group.add_argument('--support_image', type=str, default=None,
                              help="Path to support image (for single mode via CLI)")
    support_group.add_argument('--support_bbox', type=float, nargs=4, default=None,
                              help="Support bbox [x y w h] (for single mode via CLI)")
    support_group.add_argument('--class_name', type=str, default='object',
                              help="Class name for support set")
    
    # Query arguments
    query_group = parser.add_argument_group('Query Images')
    query_group.add_argument('--query_dir', type=str, default=None,
                            help="Directory with query images (required for batch mode)")
    query_group.add_argument('--query_image', type=str, default=None,
                            help="Path to single query image (required for single mode)")
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_csv', type=str, default='batch_results.csv',
                             help="Output CSV file for batch mode")
    output_group.add_argument('--output_dir', type=str, default='output',
                             help="Output directory for single mode visualizations")
    
    # Detection parameters
    det_group = parser.add_argument_group('Detection Parameters')
    det_group.add_argument('--score_threshold', type=float, default=0.3,
                          help="Detection score threshold")
    det_group.add_argument('--nms_threshold', type=float, default=0.4,
                          help="NMS threshold")
    det_group.add_argument('--max_detections', type=int, default=100,
                          help="Maximum number of detections")
    
    args = parser.parse_args()
    main(args)
