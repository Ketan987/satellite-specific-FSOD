# Use PyTorch official image with CPU support
FROM pytorch/pytorch:2.0-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and output
RUN mkdir -p /app/data/support_images \
    && mkdir -p /app/data/query_images \
    && mkdir -p /app/output \
    && mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Expose port for potential API usage (optional)
EXPOSE 8000

# Default command: inference in single mode
# Users can override this with their own commands
CMD ["python3", "inference.py", "--help"]
