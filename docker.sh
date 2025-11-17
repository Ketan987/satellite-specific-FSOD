#!/bin/bash

# FSOD Docker Helper Script
# Simplifies common Docker operations

set -e

IMAGE_NAME="fsod-inference"
CONTAINER_NAME="fsod-app"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
print_help() {
    cat << EOF
${BLUE}FSOD Docker Helper${NC}

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  build              Build Docker image
  run-single         Run single mode inference
  run-batch          Run batch inference
  train              Train model
  shell              Interactive shell
  push               Push image to registry
  clean              Remove images and containers

Examples:
  $0 build
  $0 run-single --support_dir data/support_images --query_image data/query.jpg
  $0 run-batch --support_dir data/support_images --query_dir data/queries
  $0 train --num_episodes 100 --device cpu

Options for run-single:
  --support_dir DIR      Directory with support images (required)
  --query_image FILE     Path to query image (required)
  --output_dir DIR       Output directory (default: output)
  --device DEVICE        cpu or cuda (default: cpu)

Options for run-batch:
  --support_dir DIR      Directory with support images (required)
  --query_dir DIR        Directory with query images (required)
  --output_csv FILE      Output CSV file (default: batch_results.csv)
  --device DEVICE        cpu or cuda (default: cpu)

Options for train:
  --num_episodes N       Number of episodes (default: 100)
  --device DEVICE        cpu or cuda (default: cpu)
  --pretrained           Use pretrained weights

EOF
}

# Build image
build_image() {
    echo -e "${GREEN}Building Docker image: $IMAGE_NAME${NC}"
    docker build -t $IMAGE_NAME .
    echo -e "${GREEN}✅ Image built successfully${NC}"
}

# Run single mode
run_single() {
    echo -e "${GREEN}Running single mode inference...${NC}"
    docker run --rm \
        -v $(pwd)/data/support_images:/app/data/support_images:ro \
        -v $(pwd)/output:/app/output \
        -v $(pwd)/checkpoints:/app/checkpoints:ro \
        $IMAGE_NAME single "$@"
}

# Run batch mode
run_batch() {
    echo -e "${GREEN}Running batch inference...${NC}"
    docker run --rm \
        -v $(pwd)/data/support_images:/app/data/support_images:ro \
        -v $(pwd)/data/query_images:/app/data/query_images:ro \
        -v $(pwd)/output:/app/output \
        -v $(pwd)/checkpoints:/app/checkpoints:ro \
        $IMAGE_NAME batch "$@"
}

# Train
train_model() {
    echo -e "${GREEN}Starting training...${NC}"
    docker run --rm \
        -v $(pwd)/data:/app/data:ro \
        -v $(pwd)/checkpoints:/app/checkpoints \
        $IMAGE_NAME train "$@"
}

# Interactive shell
run_shell() {
    echo -e "${GREEN}Starting interactive shell...${NC}"
    docker run -it --rm \
        -v $(pwd):/app \
        $IMAGE_NAME bash
}

# Push to registry
push_image() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Registry name required${NC}"
        echo "Usage: $0 push [REGISTRY]/[IMAGE]:[TAG]"
        exit 1
    fi
    
    echo -e "${GREEN}Tagging image: $1${NC}"
    docker tag $IMAGE_NAME $1
    
    echo -e "${GREEN}Pushing image to registry...${NC}"
    docker push $1
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
}

# Clean up
clean() {
    echo -e "${YELLOW}Removing Docker resources...${NC}"
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Removing container: $CONTAINER_NAME"
        docker rm -f $CONTAINER_NAME
    fi
    
    if docker images --format '{{.Repository}}' | grep -q "^${IMAGE_NAME}$"; then
        echo "Removing image: $IMAGE_NAME"
        docker rmi -f $IMAGE_NAME
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Main
if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

case "$1" in
    build)
        build_image
        ;;
    run-single)
        shift
        run_single "$@"
        ;;
    run-batch)
        shift
        run_batch "$@"
        ;;
    train)
        shift
        train_model "$@"
        ;;
    shell)
        run_shell
        ;;
    push)
        push_image "$2"
        ;;
    clean)
        clean
        ;;
    help|-h|--help)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_help
        exit 1
        ;;
esac
