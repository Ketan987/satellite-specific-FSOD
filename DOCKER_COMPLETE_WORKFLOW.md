# 🐳 Docker Complete Workflow: Training & Inference

Complete end-to-end guide for running FSOD training and inference in Docker with GPU/CPU support. Checkpoints are automatically saved outside the container for persistent use.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Building Docker Image](#building-docker-image)
4. [Training with Docker](#training-with-docker)
5. [Running Inference](#running-inference)
6. [Checkpoint Management](#checkpoint-management)
7. [All Docker Commands Reference](#all-docker-commands-reference)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Docker** installed ([Install Guide](https://docs.docker.com/get-docker/))
- **Docker Compose** (optional but recommended)
- **NVIDIA Docker Runtime** (for GPU support - [Install](https://github.com/NVIDIA/nvidia-docker))
- **At least 8GB RAM** (16GB recommended for GPU)
- **50GB disk space** (for image + models + data)

### Check Installation
```bash
# Verify Docker
docker --version
docker-compose --version

# Verify GPU support (if using GPU)
nvidia-smi
```

---

## Initial Setup

### 1. Prepare Data Structure

```bash
# From project root, create required directories
mkdir -p data/train_images data/val_images data/support_images data/query_images
mkdir -p checkpoints output

# Verify structure
tree -L 2 data/
tree -L 1 checkpoints/
```

Expected structure:
```
fsod/
├── data/
│   ├── train_images/          # Training images
│   ├── val_images/            # Validation images
│   ├── support_images/        # Support set for inference
│   ├── query_images/          # Query images for inference
│   ├── train_coco.json        # Training annotations
│   └── val_coco.json          # Validation annotations
├── checkpoints/               # Model checkpoints (persists outside container)
└── output/                    # Inference results
```

### 2. Prepare Training Data

If you have COCO format annotations:
```bash
# Copy your training data
cp /path/to/train_images/* data/train_images/
cp /path/to/val_images/* data/val_images/
cp /path/to/train_coco.json data/
cp /path/to/val_coco.json data/
```

For inference, prepare support and query images:
```bash
# Support images (few examples of object class)
cp /path/to/support_images/* data/support_images/

# Query images (images to run inference on)
cp /path/to/query_images/* data/query_images/
```

---

## Building Docker Image

### 1. Build Image from Dockerfile

```bash
# Build with default settings (CPU)
docker build -t fsod-inference .

# Build with custom tag for versioning
docker build -t fsod-inference:v1.0 .

# Force rebuild without cache
docker build -t fsod-inference . --no-cache
```

### 2. Using Docker Compose (Recommended)

```bash
# Build using docker-compose
docker-compose build

# Build and pull latest images
docker-compose build --pull
```

### 3. Verify Image Built Successfully

```bash
docker images | grep fsod

# Expected output:
# REPOSITORY          TAG       IMAGE ID      CREATED      SIZE
# fsod-inference      latest    abc123def456  5 hours ago  7.5GB
```

### 4. View Image Layers

```bash
docker history fsod-inference
```

---

## Training with Docker

### ⚠️ IMPORTANT: Volume Mounting

Checkpoints trained in Docker are **automatically saved outside the container** to `./checkpoints/` directory. This means:
- ✅ Models persist after container stops
- ✅ You can use trained models for inference anytime
- ✅ Models available on host machine immediately

### 1. CPU Training

```bash
# Simple CPU training
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 100 \
  --device cpu

# With output visible
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 100 \
  --device cpu

# Long training in background
docker run -d \
  --name fsod-training-cpu \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 1000 \
  --device cpu
```

### 2. GPU Training (Recommended)

#### Prerequisites for GPU
```bash
# Verify NVIDIA Docker works
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Should show GPU info. If not, install nvidia-docker:
# Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Run Training on GPU

```bash
# Single GPU training
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 100 \
  --device cuda

# Specific GPU (if multiple available)
docker run --rm --gpus '"device=0"' \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 100 \
  --device cuda

# Multiple GPUs
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 100 \
  --device cuda

# GPU training with logging
docker run --rm -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 100 \
  --device cuda \
  2>&1 | tee training.log

# Long GPU training in background
docker run -d \
  --name fsod-training-gpu \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 1000 \
  --device cuda
```

### 3. Using Docker Compose for Training

```bash
# Edit docker-compose.yml to set device (cpu or cuda)
# Then run:
docker-compose run --rm fsod train --num_episodes 100

# GPU with docker-compose (requires GPU section in docker-compose.yml)
docker-compose run --rm fsod train --num_episodes 100
```

### 4. Monitor Training

```bash
# View logs from running container
docker logs -f fsod-training-gpu

# Check GPU usage during training
docker exec fsod-training-gpu nvidia-smi

# Check checkpoint files being created
ls -lh checkpoints/
```

### 5. Training Parameters

```bash
# Common training options
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 500 \      # Number of training episodes
  --device cuda \           # 'cuda' or 'cpu'
  --pretrained \            # Use pretrained ResNet-50 backbone
  --test_episodes 10        # Quick test run (overrides num_episodes)
```

### 6. Stop Running Training

```bash
# Stop background training container
docker stop fsod-training-gpu
docker stop fsod-training-cpu

# Remove container after stopping
docker rm fsod-training-gpu

# Kill all fsod containers
docker kill $(docker ps -q --filter "ancestor=fsod-inference")
```

---

## Checkpoint Management

### ✅ Checkpoints Auto-Save to Host

During training, checkpoints are automatically saved to `./checkpoints/` directory on your host machine:

```bash
# After training, check saved files
ls -lh checkpoints/

# Expected files:
# best_model.pth           # Best model by validation mAP
# final_model.pth          # Final model after all episodes
# checkpoint_ep_100.pth    # Episodic checkpoints
# checkpoint_ep_200.pth
# ...
```

### View Checkpoint Details

```bash
# Check checkpoint file size
du -sh checkpoints/*.pth

# List all checkpoints with timestamps
ls -lh --time-style=long-iso checkpoints/

# Count total checkpoints
ls -1 checkpoints/*.pth | wc -l
```

### Use Trained Model for Inference Anytime

```bash
# Checkpoint is accessible immediately after training
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg

# Or next week:
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/new_image.jpg
```

---

## Running Inference

### 1. Single Image Inference

```bash
# CPU inference
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg

# GPU inference
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg \
  --device cuda
```

### 2. Batch Inference (Multiple Images)

```bash
# CPU batch inference
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --output_csv output/detections.csv

# GPU batch inference
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --output_csv output/detections.csv \
  --device cuda
```

### 3. Check Inference Results

```bash
# View detection CSV
cat output/detections.csv

# List visualizations
ls -lh output/visualizations/

# View a result image
display output/visualizations/result_1.jpg

# Or with preview
file output/visualizations/result_1.jpg
```

### 4. Custom Output Paths

```bash
# Save results to specific location
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --output_dir output/batch_results_v1 \
  --output_csv output/batch_results_v1/detections.csv \
  --device cuda
```

---

## All Docker Commands Reference

### Image Management

```bash
# Build image
docker build -t fsod-inference .
docker build -t fsod-inference:v1.0 .
docker build -t fsod-inference . --no-cache

# List images
docker images | grep fsod
docker images -a

# Remove image
docker rmi fsod-inference
docker rmi fsod-inference:v1.0

# Inspect image
docker inspect fsod-inference
docker history fsod-inference
```

### Container Execution - Training

```bash
# CPU training - interactive
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train --num_episodes 100 --device cpu

# GPU training - interactive
docker run --rm -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train --num_episodes 100 --device cuda

# Background training
docker run -d --name fsod-train-job --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train --num_episodes 500 --device cuda
```

### Container Execution - Inference

```bash
# Single image - CPU
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg

# Batch - GPU
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --device cuda
```

### Container Execution - Interactive Shell

```bash
# CPU interactive shell
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference bash

# GPU interactive shell
docker run --rm -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference bash

# Inside container, you can:
cd /app
python3 train.py --help
python3 inference.py --help
pip install ipdb
python3 -m ipdb train.py
```

### Container Management

```bash
# List running containers
docker ps
docker ps -a

# View container logs
docker logs container_id
docker logs -f container_id          # Follow logs
docker logs --tail 50 container_id   # Last 50 lines

# Execute command in running container
docker exec container_id ls /app/checkpoints
docker exec -it container_id bash

# Stop container
docker stop container_id

# Kill container (force stop)
docker kill container_id

# Remove container
docker rm container_id

# Remove all stopped containers
docker container prune
```

### Docker Compose Commands

```bash
# Build services
docker-compose build
docker-compose build --no-cache
docker-compose build --pull

# Run services
docker-compose up                  # Start and attach
docker-compose up -d               # Start in background
docker-compose down                # Stop and remove

# Run single service command
docker-compose run --rm fsod train --num_episodes 100
docker-compose run --rm fsod single --help

# View logs
docker-compose logs
docker-compose logs -f fsod        # Follow
docker-compose logs --tail 50 fsod

# List running services
docker-compose ps

# Execute in running service
docker-compose exec fsod bash
docker-compose exec fsod ls /app/checkpoints

# Stop services
docker-compose stop
docker-compose kill
```

### Disk Space & Cleanup

```bash
# Check Docker disk usage
docker system df

# Clean up unused images
docker image prune
docker image prune -a            # Remove all unused

# Clean up unused containers
docker container prune

# Clean up unused volumes
docker volume prune

# Full cleanup (be careful!)
docker system prune -a
```

### GPU Commands

```bash
# Check GPU in container
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU during training
docker exec container_id nvidia-smi

# List available GPUs
docker run --rm --gpus all fsod-inference bash -c "python3 -c \"import torch; print(torch.cuda.device_count())\""

# Specify GPU
docker run --rm --gpus '"device=0"' fsod-inference ...
docker run --rm --gpus '"device=0,1"' fsod-inference ...
```

---

## Complete Workflow Example

### Step-by-Step Guide

```bash
# 1. Build Docker image (one-time)
cd fsod
docker build -t fsod-inference .
# Wait 15-20 minutes...

# 2. Verify image built
docker images | grep fsod

# 3. Prepare your data
mkdir -p data/{train_images,val_images,support_images,query_images}
# Copy your COCO dataset and annotations here

# 4. Train on GPU (long running)
docker run -d --name fsod-train \
  --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train --num_episodes 500 --device cuda

# 5. Monitor training
docker logs -f fsod-train
# Check checkpoints being saved
ls -lh checkpoints/

# 6. After training completes
docker stop fsod-train

# 7. Check final checkpoints on host
ls -lh checkpoints/best_model.pth
ls -lh checkpoints/final_model.pth

# 8. Run inference on GPU
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg

# 9. Check results
ls -lh output/

# 10. Later: Run batch inference with same trained model
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference batch \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_dir data/query_images \
  --device cuda
```

---

## Troubleshooting

### Training Issues

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce batch size in `config.py`: `IMAGE_SIZE = 256` |
| **GPU not detected** | Run `nvidia-smi` on host. Install nvidia-docker if needed |
| **Training very slow** | Use GPU instead of CPU: `--device cuda` |
| **Data not found** | Check volume mount paths with `docker run --rm -it fsod-inference bash` |
| **Checkpoints not saving** | Verify `-v $(pwd)/checkpoints:/app/checkpoints` mount |
| **Permission denied** | Run `chmod 755 checkpoints/` on host |

### Inference Issues

| Problem | Solution |
|---------|----------|
| **Model not found** | Use absolute path or verify checkpoint exists: `ls checkpoints/` |
| **Inference slow** | Use GPU: `--gpus all` and `--device cuda` |
| **Out of memory** | Reduce query image count or image size |
| **No results** | Check `output/` directory has write permissions |

### Docker Issues

| Problem | Solution |
|---------|----------|
| **Image too large** | Normal - PyTorch is ~7.5GB. Use SSD or external drive |
| **Port conflicts** | Change port mapping in `docker-compose.yml` |
| **Container exits immediately** | Check logs: `docker logs container_id` |
| **Disk space full** | Run `docker system prune -a` |
| **Permission denied** | Run with `sudo` or add user to docker group |

---

## Performance Expectations

### Training Time (100 episodes)
- **CPU (Intel i7)**: ~60-90 minutes
- **GPU (Tesla T4)**: ~50-60 minutes
- **GPU (RTX 3090)**: ~15-20 minutes

### Inference Time (Single Image)
- **CPU**: 5-10 seconds
- **GPU (T4)**: 1-2 seconds
- **GPU (RTX)**: 0.5-1 seconds

### Disk Usage
- **Docker image**: 7.5 GB
- **Trained models**: 200-500 MB
- **Dataset (1000 images)**: 2-5 GB
- **Results/Logs**: 500 MB - 2 GB

---

## Tips & Best Practices

### ✅ DO's
- ✅ Use GPU for faster training and inference
- ✅ Mount checkpoints volume for persistent models
- ✅ Run training in background with `-d` flag
- ✅ Use `docker logs -f` to monitor progress
- ✅ Keep old checkpoints for comparison
- ✅ Test inference immediately after training

### ❌ DON'Ts
- ❌ Don't use `--rm` when you need to preserve models
- ❌ Don't forget volume mounts (data will be lost)
- ❌ Don't use `--gpus all` if GPU not available
- ❌ Don't interrupt training without saving checkpoints
- ❌ Don't mix CPU and GPU training in same container

---

## Quick Start Commands

### For GPU System
```bash
# Build (first time only)
docker build -t fsod-inference .

# Train
docker run -d --name train --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train --num_episodes 500 --device cuda

# Monitor
docker logs -f train

# Inference (after training)
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg
```

### For CPU System
```bash
# Build
docker build -t fsod-inference .

# Train
docker run -d --name train \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train --num_episodes 100 --device cpu

# Inference
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference single \
  --model_path checkpoints/best_model.pth \
  --support_dir data/support_images \
  --query_image data/query_images/test.jpg
```

---

## Advanced Topics

### Custom Training Configuration

```bash
# With pretrained backbone
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --num_episodes 500 \
  --device cuda \
  --pretrained

# Quick test (10 episodes)
docker run --rm -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fsod-inference train \
  --test_episodes 10 \
  --device cuda
```

### Custom Inference Configuration

Edit `config.py` for inference parameters:
```python
# In config.py
IMAGE_SIZE = 384           # Change image size
CONFIDENCE_THRESHOLD = 0.5 # Detection threshold
```

### Building Multi-Stage Images (Advanced)

```dockerfile
# For production, create smaller image
FROM python:3.10-slim as builder
# ... build stage

FROM python:3.10-slim as runtime
# ... copy only runtime artifacts
```

---

## Support & Issues

For issues or questions:

1. **Check logs**: `docker logs container_id`
2. **Verify mounts**: `docker run -it fsod-inference bash`
3. **Check GPU**: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
4. **Test data**: Ensure COCO JSON is valid
5. **Disk space**: Run `docker system df`

---

**Last Updated**: November 18, 2025  
**Version**: 1.0  
**Docker Base Image**: Python 3.10 Slim (Bullseye)  
**PyTorch**: 2.0.1  
**CUDA Support**: Yes (with nvidia-docker)
