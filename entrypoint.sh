#!/bin/bash

# FSOD Docker Entrypoint Script
# This script helps with common FSOD operations

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
show_help() {
    echo -e "${BLUE}FSOD Docker Entrypoint${NC}"
    echo ""
    echo "Usage: docker run fsod-inference [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  single       Run single image inference"
    echo "  batch        Run batch inference on directory"
    echo "  train        Train the model"
    echo "  bash         Interactive shell"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  docker run fsod-inference single --help"
    echo "  docker run fsod-inference batch --help"
    echo "  docker run fsod-inference train --help"
    echo ""
}

# Default to help if no arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case "$COMMAND" in
    single)
        echo -e "${GREEN}Running single mode inference...${NC}"
        python3 inference.py --mode single "$@"
        ;;
    batch)
        echo -e "${GREEN}Running batch inference...${NC}"
        python3 inference.py --mode batch "$@"
        ;;
    train)
        echo -e "${GREEN}Starting training...${NC}"
        python3 train.py "$@"
        ;;
    bash)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        /bin/bash "$@"
        ;;
    help)
        show_help
        ;;
    *)
        # If it looks like a python script, run it
        if [[ "$COMMAND" == *.py ]]; then
            python3 "$COMMAND" "$@"
        else
            # Otherwise try to run as a command directly
            "$COMMAND" "$@"
        fi
        ;;
esac
