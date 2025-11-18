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

# FSOD - Few-Shot Object Detection

A production-ready Few-Shot Object Detection system using ResNet-50 backbone with COCO-style annotations. Designed for efficient learning from minimal training examples.

## 🎯 Features

- ✅ **ResNet-50 Backbone** - Pretrained ImageNet feature extractor
- ✅ **COCO Format Support** - Standard JSON annotations with JPEG images
- ✅ **N-way K-shot Learning** - Episodic training for generalization (default: 5-way, 3-shot)
- ✅ **Similarity Matching** - Cosine similarity with temperature scaling
- ✅ **Efficient ROI Pooling** - Fixed-size region feature extraction
- ✅ **Robust Detection Pipeline** - Focal loss for class imbalance, NMS post-processing
- ✅ **Two Inference Modes** - Single image with visualization, batch with CSV output
- ✅ **Docker Support** - Containerized training and inference
- ✅ **Comprehensive Architecture Verification** - Tested edge cases and numerical stability

## 📦 Project Structure

```
fsod/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration parameters
├── train.py                       # Training script
├── inference.py                   # Inference script (single/batch modes)
├── example_usage.py               # Usage examples
│
├── models/
│   ├── backbone.py               # ResNet-50 + FeatureEmbedding + ROIPooling
│   ├── detector.py               # Main FSODDetector orchestrator
│   └── similarity.py             # SimilarityMatcher + ProposalGenerator + NMS
│
├── utils/
│   ├── coco_utils.py             # COCO dataset loading
│   ├── data_loader.py            # Data loading and episodic sampling
│   └── metrics.py                # mAP computation
│
├── tests/
│   ├── comprehensive_test.py     # Full pipeline tests
│   └── smoke_test.py             # Basic sanity checks
│
├── data/
│   ├── train_coco.json           # Training COCO annotations
│   ├── val_coco.json             # Validation COCO annotations
│   ├── train_images/             # Training images (JPEG)
│   └── val_images/               # Validation images (JPEG)
│
├── checkpoints/                   # Model checkpoints
├── documentation/
│   ├── INFERENCE_GUIDE.md        # Detailed inference documentation
│   ├── DOCKER_SETUP.md           # Docker usage guide
│   └── ARCHITECTURE_VERIFICATION.md  # Architecture validation report
│
├── Docker files
│   ├── Dockerfile                # Docker image definition
│   ├── docker-compose.yml        # Docker Compose configuration
│   ├── docker.sh                 # Docker helper script
│   └── entrypoint.sh             # Container entrypoint
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager
- GPU with CUDA 11.8+ (optional, CPU works fine for testing)

### Installation

1. **Clone and navigate to the repository**
```bash
cd /path/to/fsod
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## 📊 Data Format

### COCO Annotation Format

FSOD requires COCO-style JSON annotations. Each dataset consists of:

```
data/
├── train_coco.json              # Training annotations
├── val_coco.json                # Validation annotations
├── train_images/                # Training images (JPEG only)
└── val_images/                  # Validation images (JPEG only)
```

**COCO JSON Structure:**
```json
{
  "images": [
    {"id": 1, "file_name": "image001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 50000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "cat", "supercategory": "animal"}
  ]
}
```

**Important Notes:**
- Bounding boxes are in `[x, y, width, height]` format (top-left corner + dimensions)
- **Only JPEG images are supported** (.jpg or .jpeg extensions)
- Image must exist in the corresponding directory
- At least N_WAY unique classes recommended (default 5)
- K_SHOT examples per class recommended (default 3)

## 🏋️ Training

### Basic Training (CPU - Good for Testing)

```bash
# Activate virtual environment
source venv/bin/activate

# Train for 100 episodes on CPU
python train.py --device cpu --num_episodes 100 --pretrained
```

**Estimated time:** 5-10 minutes on CPU

### Full Training (GPU - Recommended)

```bash
source venv/bin/activate

# Train for 10000 episodes on GPU
python train.py --device cuda --num_episodes 10000 --pretrained
```

**Estimated time:** 2-5 hours on GPU

### Configuration

Edit `config.py` to adjust training parameters:

```python
# Few-shot parameters
N_WAY = 5               # Number of classes per episode
K_SHOT = 3              # Support examples per class
QUERY_SAMPLES = 10      # Query examples per class

# Training parameters
NUM_EPISODES = 10000
LEARNING_RATE = 1e-4
BATCH_SIZE = 1          # Per-episode

# Image parameters
IMAGE_SIZE = 512
```

### Monitoring Training

The training script automatically:
- Logs loss every 100 episodes
- Validates every 1000 episodes
- Saves checkpoints every 1000 episodes
- Saves best model based on validation mAP

Output: `checkpoints/best_model.pth`

## 🎯 Inference

### Two Inference Modes

FSOD provides two inference modes optimized for different workflows:

#### Mode 1: Single Image (With Visualization)

Process one query image and automatically annotate detections:

```bash
source venv/bin/activate

python inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --support_img cat1.jpg cat2.jpg cat3.jpg \
  --support_classes cat cat cat \
  --query_image test_image.jpg \
  --output_dir output/ \
  --device cpu
```

**Output:**
- `output/test_image_detected.jpg` - Annotated image with bounding boxes
- Console table with detection coordinates and scores

#### Mode 2: Batch Processing (CSV Output)

Process multiple images and save results to CSV:

```bash
source venv/bin/activate

python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_img dog1.jpg dog2.jpg \
  --support_classes dog dog \
  --query_dir data/test_images/ \
  --output_csv results.csv \
  --device cpu
```

**Output CSV Format:**
```
x_min,y_min,x_max,y_max,filename,class_name,similarity_score
100.5,150.2,250.8,300.1,image001.jpg,dog,0.8723
50.1,75.3,180.9,200.5,image001.jpg,dog,0.7654
```

### Multi-Class Detection

Detect multiple object classes simultaneously:

```bash
python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_img cat1.jpg cat2.jpg dog1.jpg dog2.jpg bird1.jpg \
  --support_classes cat cat dog dog bird \
  --query_dir data/test_images/ \
  --output_csv multi_class_results.csv
```

### Detection Parameters

Fine-tune detection behavior with these parameters:

- `--score_threshold` (default: 0.3) - Confidence threshold (0-1)
  - Lower = more detections (may include false positives)
  - Higher = only confident detections
- `--nms_threshold` (default: 0.4) - NMS IoU threshold
- `--max_detections` (default: 100) - Maximum detections per image
- `--device` - 'cpu' or 'cuda'

**For detailed usage examples, see:** [documentation/INFERENCE_GUIDE.md](documentation/INFERENCE_GUIDE.md)

## 🐳 Docker Support

FSOD includes Docker support for easy containerized training and inference.

### Build Docker Image

```bash
docker build -t fsod-inference .

# Or with docker-compose
docker-compose build
```

### Run with Docker

```bash
# Single image inference
docker-compose run --rm fsod python3 inference.py \
  --mode single \
  --model_path checkpoints/best_model.pth \
  --support_img data/support.jpg \
  --query_image data/query.jpg

# Batch inference
docker-compose run --rm fsod python3 inference.py \
  --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_img data/cat1.jpg data/cat2.jpg \
  --query_dir data/images/ \
  --output_csv results.csv

# Training
docker-compose run --rm fsod python3 train.py \
  --device cpu --num_episodes 100
```

**For detailed Docker setup:** [documentation/DOCKER_SETUP.md](documentation/DOCKER_SETUP.md)

## 🏗️ Model Architecture

```
Input Images (Support & Query)
    ↓
ResNet-50 Backbone [pretrained, stride 32]
    ↓
Feature Embedding [2048-dim → 512-dim via 1×1 convolutions]
    ↓
ROI Pooling [extract 7×7 regions from each bounding box]
    ↓
Detection Head [FC layers: 25088 → 1024 → 512]
    ↓
Three Output Heads:
  ├─ Box Regressor      [512 → 4 deltas]
  ├─ Objectness Head    [512 → 1 score]
  └─ Similarity Matcher [cosine similarity to support features]
    ↓
Post-processing [NMS + score thresholding]
    ↓
Final Detections [bboxes + similarity scores + class names]
```

**Key Components:**
- **Backbone**: ResNet-50 pretrained on ImageNet
- **Feature Embedding**: Reduces feature dimensionality for efficiency
- **ROI Pooling**: Extracts fixed-size features from proposals
- **Similarity Matching**: Computes cosine similarity with temperature scaling
- **NMS**: Non-maximum suppression to remove duplicates
- **Training Loss**: Focal loss (class imbalance) + Smooth L1 (box regression)

## ⚙️ Performance Optimization

### Memory Usage
- **Training**: ~4-6 GB GPU VRAM for default settings
- **Inference**: ~2-3 GB GPU VRAM

If out of memory, adjust in `config.py`:
```python
IMAGE_SIZE = 384      # Reduce from 512
K_SHOT = 2            # Reduce from 3
NUM_EPISODES = 5000   # Reduce from 10000
```

### Speed
- **CPU Training**: ~5-10 sec/episode (100 episodes ≈ 10 min)
- **GPU Training**: ~0.5-1 sec/episode (10000 episodes ≈ 2-5 hours)
- **CPU Inference**: ~2-5 sec/image
- **GPU Inference**: ~0.1-0.3 sec/image

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `IMAGE_SIZE` or use CPU mode |
| **Poor detection results** | Increase `K_SHOT`, use better support images, lower `score_threshold` |
| **Image format error** | Ensure all images are JPEG (.jpg or .jpeg) |
| **No detections found** | Lower `score_threshold` (try 0.1-0.2), check model training |
| **Slow training** | Use GPU (`--device cuda`), reduce `NUM_EPISODES` |
| **Docker permission errors** | Run `chmod 777 output/` |

## 📚 Additional Resources

- **Detailed Inference Guide**: [documentation/INFERENCE_GUIDE.md](documentation/INFERENCE_GUIDE.md)
- **Docker Setup Guide**: [documentation/DOCKER_SETUP.md](documentation/DOCKER_SETUP.md)
- **Architecture Verification Report**: [documentation/ARCHITECTURE_VERIFICATION.md](documentation/ARCHITECTURE_VERIFICATION.md)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchVision 0.15+
- NumPy 1.24+
- Pillow 10.0+
- tqdm 4.65+

See `requirements.txt` for exact versions.

## 📝 Example Workflow

```bash
# 1. Setup
source venv/bin/activate

# 2. Prepare COCO data
# - Create data/train_coco.json and data/val_coco.json
# - Place JPEG images in data/train_images/ and data/val_images/

# 3. Train model
python train.py --device cuda --num_episodes 10000 --pretrained

# 4. Run inference
python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_img support/cat1.jpg support/cat2.jpg \
  --query_dir data/test_images/ \
  --output_csv results.csv

# 5. Evaluate results (results.csv contains all detections)
```



docker build -t fsod-inference .

docker run --rm fsod-inference single --help
docker-compose up

docker compose run --rm fsod train --num_episodes 10 --device cpu

