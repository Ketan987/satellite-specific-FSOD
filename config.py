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
    EMBEDDING_DIM = 512  # Reduced dimension for similarity
    
    # Few-shot parameters
    N_WAY = 5  # Number of classes per episode
    K_SHOT = 5  # Number of support examples per class (increased from 2 for better prototypes)
    QUERY_SAMPLES = 20  # Number of query samples per episode (increased from 10 for better statistics)
    
    # Training parameters
    BATCH_SIZE = 1
    NUM_EPISODES = 10000
    LEARNING_RATE = 5e-4  # Increased from 1e-4 for better convergence
    WEIGHT_DECAY = 5e-4
    
    # Image parameters
    IMAGE_SIZE = 384  # Increased from 256 for better satellite image detail
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    ALLOWED_FORMATS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    
    # Detection parameters
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.5  # Increased from 0.3 to filter noisy predictions
    NMS_THRESHOLD = 0.3  # Stricter NMS to remove overlapping boxes
    MAX_DETECTIONS = 100
    
    # Anchor boxes (scaled to IMAGE_SIZE)
    ANCHOR_SCALES = [32, 64, 128, 256]
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    
    # Device
    DEVICE = "cuda"  # or "cpu"
    
    # Checkpoint
    CHECKPOINT_DIR = "checkpoints/"
    SAVE_FREQUENCY = 1000  # Save every N episodes
    
    # Logging
    LOG_FREQUENCY = 100  # Log every N episodes