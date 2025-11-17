# FSOD (Few-Shot Object Detection) Project

A lightweight Few-Shot Object Detection system using ResNet-50 backbone with COCO-style annotations.

## Features

- ✅ **ResNet-50 backbone** for robust feature extraction
- ✅ **COCO-style annotations** for training and validation
- ✅ **JPEG-only** image support with validation
- ✅ **Similarity scores** (0-1 range) for each detection
- ✅ **Bounding box predictions** in [x, y, width, height] format
- ✅ **N-way K-shot** episodic training
- ✅ **Simple inference API** for easy deployment

## Project Structure

```
fsod_project/
├── config.py                 # Configuration parameters
├── train.py                  # Training script
├── inference.py              # Inference script
├── example_usage.py          # Usage examples
├── models/
│   ├── backbone.py          # ResNet-50 feature extractor
│   ├── detector.py          # Main FSOD detector
│   └── similarity.py        # Similarity matching
├── utils/
│   ├── coco_utils.py        # COCO dataset utilities
│   └── data_loader.py       # Data loading and preprocessing
├── data/
│   ├── train_coco.json      # Training annotations
│   ├── val_coco.json        # Validation annotations
│   ├── train_images/        # Training images (JPEG)
│   └── val_images/          # Validation images (JPEG)
├── checkpoints/             # Model checkpoints
└── README.md
```

## Installation

### Requirements

```bash
pip install torch torchvision pillow numpy tqdm
```

### Python Version
- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU support)

## Data Format

### COCO-style Annotation Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 50000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "cat",
      "supercategory": "animal"
    }
  ]
}
```

**Important Notes:**
- Bounding boxes are in `[x, y, width, height]` format
- All images must be in **JPEG format** (.jpg or .jpeg)
- Images should be placed in the corresponding directories

## Training

### Basic Training

```bash
python train.py
```

### Configuration

Edit `config.py` to adjust training parameters:

```python
# Few-shot parameters
N_WAY = 5           # Number of classes per episode
K_SHOT = 5          # Number of support examples per class
QUERY_SAMPLES = 10  # Number of query samples per episode

# Training parameters
NUM_EPISODES = 10000
LEARNING_RATE = 1e-4
BATCH_SIZE = 4

# Image parameters
IMAGE_SIZE = 512
```

### Monitor Training

The training script will:
- Log loss every 100 episodes
- Validate every 1000 episodes
- Save checkpoints every 1000 episodes
- Save the best model based on validation loss

## Inference

### Method 1: Using the Inference Class

```python
from inference import FSODInference

# Initialize
inferencer = FSODInference(
    model_path='checkpoints/best_model.pth',
    device='cuda'  # or 'cpu'
)

# Define support set (5-shot example)
support_set = [
    {
        'image_path': 'support/cat1.jpg',
        'bbox': [50, 60, 120, 150],  # [x, y, width, height]
        'class_name': 'cat'
    },
    {
        'image_path': 'support/cat2.jpg',
        'bbox': [30, 40, 110, 140],
        'class_name': 'cat'
    },
    # Add more support examples...
]

# Run detection
detections = inferencer.detect(
    support_set=support_set,
    query_image='query/test.jpg',
    score_threshold=0.3,
    nms_threshold=0.4,
    max_detections=100
)

# Print results
for det in detections:
    print(f"BBox: {det['bbox']}")
    print(f"Score: {det['similarity_score']:.4f}")
    print(f"Class: {det['class_name']}")
```

### Method 2: Command Line

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --query_image query/test.jpg \
    --support_image1 support/cat1.jpg \
    --support_bbox1 50 60 120 150 \
    --class_name cat \
    --score_threshold 0.3 \
    --output results.json
```

## Output Format

The detection output follows this format:

```json
{
  "image_path": "query/test_image.jpg",
  "detections": [
    {
      "bbox": [150.5, 200.3, 250.8, 350.6],
      "similarity_score": 0.8523,
      "class_name": "cat"
    },
    {
      "bbox": [400.2, 150.7, 200.5, 280.4],
      "similarity_score": 0.7891,
      "class_name": "cat"
    }
  ]
}
```

**Field Descriptions:**
- `bbox`: Bounding box in `[x, y, width, height]` format
- `similarity_score`: Similarity score between 0 and 1 (higher is better)
- `class_name`: Predicted class name from support set

## Advanced Usage

### Multi-Class Detection

```python
# Support set with multiple classes (N-way)
support_set = [
    # Cat examples
    {'image_path': 'cat1.jpg', 'bbox': [50, 60, 120, 150], 'class_name': 'cat'},
    {'image_path': 'cat2.jpg', 'bbox': [30, 40, 110, 140], 'class_name': 'cat'},
    
    # Dog examples
    {'image_path': 'dog1.jpg', 'bbox': [40, 50, 130, 160], 'class_name': 'dog'},
    {'image_path': 'dog2.jpg', 'bbox': [55, 65, 125, 155], 'class_name': 'dog'},
]

detections = inferencer.detect(support_set, 'query.jpg')
```

### Batch Processing

```python
query_images = ['img1.jpg', 'img2.jpg', 'img3.jpg']

results = inferencer.detect_batch(
    support_set=support_set,
    query_images=query_images,
    score_threshold=0.35
)

# Save results
inferencer.save_results(results, 'batch_results.json')
```

## Model Architecture

```
Input Images (Support & Query)
    ↓
ResNet-50 Backbone
    ↓
Feature Embedding (512-dim)
    ↓
ROI Pooling
    ↓
Detection Head
    ↓
Similarity Matching ←→ Support Features
    ↓
Bounding Box Regression + Objectness
    ↓
NMS & Post-processing
    ↓
Final Detections (bbox + similarity score)
```

## Key Components

1. **Backbone**: ResNet-50 pretrained on ImageNet
2. **Feature Embedding**: Projects 2048-dim features to 512-dim
3. **ROI Pooling**: Extracts fixed-size features from bounding boxes
4. **Similarity Matcher**: Computes cosine similarity between support and query
5. **Detection Head**: Predicts objectness and refines bounding boxes
6. **NMS**: Removes duplicate detections

## Performance Tips

1. **GPU Memory**: Reduce `IMAGE_SIZE` if running out of memory
2. **Training Speed**: Use smaller `NUM_EPISODES` for faster experimentation
3. **Detection Quality**: 
   - Increase `K_SHOT` for better accuracy
   - Adjust `score_threshold` based on your needs
   - Use diverse support examples

## Limitations

- **JPEG only**: Only JPEG/JPG images are supported
- **Single scale**: Detection at single scale (no multi-scale)
- **Memory**: Requires GPU with at least 6GB VRAM for training
- **Speed**: ~2-5 FPS on GPU for inference (can be optimized)

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce `IMAGE_SIZE` or `BATCH_SIZE` in config

### Issue: Poor detection results
**Solution**: 
- Check support set quality (clear objects, good bounding boxes)
- Increase `K_SHOT` (more support examples)
- Adjust `score_threshold`
- Train for more episodes

### Issue: Invalid image format error
**Solution**: Ensure all images are JPEG format (.jpg or .jpeg)

## Example Workflow

```bash
# 1. Prepare data
# - Create train_coco.json and val_coco.json
# - Place JPEG images in train_images/ and val_images/

# 2. Train model
python train.py

# 3. Run inference
python example_usage.py

# 4. Evaluate results
# Check results.json for detection outputs
```

## Citation

If you use this code, please cite:

```
@misc{fsod-lightweight,
  title={Lightweight Few-Shot Object Detection with ResNet-50},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue.