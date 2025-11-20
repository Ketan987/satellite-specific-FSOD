"""
Configuration file for FSOD project
"""

class Config:
    # Data paths
    TRAIN_COCO_JSON = "data/train_coco.json"
    VAL_COCO_JSON = "data/val_coco.json"
    TRAIN_IMAGE_DIR = "data/train_images/"
    VAL_IMAGE_DIR = "data/val_images/"
    
    # Model parameters
    BACKBONE = "resnet50"
    FEATURE_DIM = 2048  # ResNet-50 output
    EMBEDDING_DIM = 320  # Higher embedding width for better feature reuse on large GPUs
    
    # Few-shot parameters
    N_WAY = 3  # Fewer classes per episode keeps GPU iterations lean
    K_SHOT = 3  # Trim support set to reduce memory without big accuracy hit
    QUERY_SAMPLES = 10
    
    # Training parameters
    BATCH_SIZE = 1
    NUM_EPISODES = 20000
    LEARNING_RATE = 3e-4  # Slightly lower LR for stability with larger support/query batches
    WEIGHT_DECAY = 1e-4
    
    # Image parameters
    IMAGE_SIZE = 384  # Smaller resize speeds up training while keeping context
    INPUT_CHANNELS = 3
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    ALLOWED_FORMATS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    
    # Detection parameters
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.5  # Increased from 0.3 to filter noisy predictions
    NMS_THRESHOLD = 0.3  # Stricter NMS to remove overlapping boxes
    MAX_DETECTIONS = 20
    
    # Anchor boxes (scaled to IMAGE_SIZE)
    ANCHOR_SCALES = [32, 64, 128, 256, 512]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    
    # Device
    DEVICE = "cuda"  # or "cpu"
    
    # Checkpoint
    CHECKPOINT_DIR = "checkpoints/"
    SAVE_FREQUENCY = 250  # Save more frequently due to longer training runs
    
    # Logging
    LOG_FREQUENCY = 100  # Log every N episodes

    # Meta-learning (Reptile-style) parameters
    USE_MAML = False         # Disable Reptile-style adaptation by default for faster episodes
    MAML_INNER_STEPS = 1     # Support-set updates per episode
    MAML_INNER_LR = 1e-4     # Inner loop learning rate
    MAML_META_LR = 1e-4      # How aggressively to move base weights toward adapted weights