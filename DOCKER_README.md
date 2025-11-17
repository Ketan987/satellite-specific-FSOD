# FSOD Docker Setup

This directory contains Docker configuration for the FSOD (Few-Shot Object Detection) application.

## Quick Start

### Prerequisites
- Docker installed
- Docker Compose installed (optional but recommended)

### Build the Docker Image

```bash
docker build -t fsod-inference .
```

Or with docker-compose:

```bash
docker-compose build
```

## Usage

### Single Mode Inference

Run inference on a single query image:

```bash
docker run --rm \
  -v $(pwd)/data/support_images:/app/data/support_images:ro \
  -v $(pwd)/data/query_images:/app/data/query_images:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  fsod-inference python3 inference.py \
  --mode single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg \
  --output_dir output
```

Or with docker-compose:

```bash
docker-compose run --rm fsod python3 inference.py \
  --mode single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg \
  --output_dir output
```

### Batch Mode Inference

Run inference on multiple query images:

```bash
docker run --rm \
  -v $(pwd)/data/support_images:/app/data/support_images:ro \
  -v $(pwd)/data/query_images:/app/data/query_images:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  fsod-inference python3 inference.py \
  --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --output_csv output/results.csv
```

Or with docker-compose:

```bash
docker-compose run --rm fsod python3 inference.py \
  --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --output_csv output/results.csv
```

### Training (CPU)

Train the model for 100 episodes on CPU:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference python3 train.py \
  --device cpu \
  --num_episodes 100 \
  --pretrained
```

Or with docker-compose:

```bash
docker-compose run --rm fsod python3 train.py \
  --device cpu \
  --num_episodes 100 \
  --pretrained
```

### Interactive Shell

Run an interactive bash shell in the container:

```bash
docker run -it --rm \
  -v $(pwd):/app \
  fsod-inference bash
```

Or with docker-compose:

```bash
docker-compose run --rm fsod bash
```

## Directory Structure

```
.
├── data/
│   ├── support_images/    # Place support/example images here
│   ├── query_images/      # Place query images here for single/batch mode
│   ├── train_images/      # Training images (if training)
│   └── val_images/        # Validation images (if training)
├── checkpoints/           # Model checkpoints (output)
├── output/                # Inference results and visualizations
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
└── .dockerignore          # Files to exclude from Docker build
```

## Environment Variables

- `DEVICE`: Device to use (cpu/cuda), default: cpu
- `PYTHONUNBUFFERED`: Set to 1 to see Python output in real-time

## GPU Support

To enable GPU support, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then build and use with GPU:

```bash
docker-compose build
docker-compose run --rm fsod python3 inference.py --mode single --device cuda ...
```

## Volumes

The container mounts the following volumes:

- `/app/data/support_images` - Support images (read-only)
- `/app/data/query_images` - Query images (read-only)
- `/app/data/test_images` - Test images (read-only)
- `/app/output` - Output directory (read-write)
- `/app/checkpoints` - Model checkpoints (read-write)

## Building with Custom Tags

```bash
docker build -t fsod-inference:v1.0 .
docker build -t myregistry/fsod-inference:latest .
```

## Pushing to Registry

```bash
docker tag fsod-inference:latest myregistry/fsod-inference:latest
docker push myregistry/fsod-inference:latest
```

## Troubleshooting

### Permission Denied Errors
If you get permission errors writing to output, ensure the output directory is writable:

```bash
chmod 777 output
```

### Out of Memory
For CPU inference on memory-constrained systems:
- Reduce batch size (already set to 1)
- Use smaller images
- Increase swap space

### GPU Not Detected
Ensure NVIDIA Docker runtime is installed:

```bash
# Install NVIDIA Docker
curl https://get.docker.com | sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Performance Notes

- CPU inference: ~10-30 seconds per image
- GPU inference: ~0.5-1 second per image
- Batch processing 100 images on CPU: ~1-3 minutes

## Support Set Format

Support images should be:
- Clean photos of the object class you want to detect
- In JPG, PNG, or similar formats
- Ideally 2-5 diverse examples
- Images will be auto-cropped using full image as bounding box

## Output Format

**Single Mode:**
- Annotated image with bounding boxes (JPG)
- Terminal output with detection coordinates and scores

**Batch Mode:**
- CSV file with columns: x_min, y_min, x_max, y_max, filename, similarity_score

## Examples

### Detect cats in images

```bash
# Prepare directories
mkdir -p data/support_images data/query_images output

# Add support images (e.g., cat.jpg, cat2.jpg)
cp cat_examples/* data/support_images/

# Run detection
docker-compose run --rm fsod python3 inference.py \
  --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --output_csv output/cat_detections.csv
```

### Interactive exploration

```bash
docker-compose run --rm fsod bash

# Inside container
python3 inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg \
  --output_dir output
```

## License

See LICENSE file in the repository.
