# Use official Python image as base
# PyTorch will be installed via pip from requirements.txt
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create directories for data and output
RUN mkdir -p /app/data/support_images \
    && mkdir -p /app/data/query_images \
    && mkdir -p /app/output \
    && mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["help"]
