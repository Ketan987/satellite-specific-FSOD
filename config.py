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
    K_SHOT = 2  # Number of support examples per class
    QUERY_SAMPLES = 10  # Number of query samples per episode
    
    # Training parameters
    BATCH_SIZE = 1
    NUM_EPISODES = 10000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4
    
    # Image parameters
    IMAGE_SIZE = 256  # Reduced from 512 to save GPU memory (Kaggle GPU limitation)
    # Normalization stats for different channel configurations
    # These will be auto-selected based on image bands
    IMAGE_MEAN_3BAND = [0.485, 0.456, 0.406]  # RGB
    IMAGE_STD_3BAND = [0.229, 0.224, 0.225]
    IMAGE_MEAN_4BAND = [0.485, 0.456, 0.406, 0.406]  # RGBN (NIR normalized similarly)
    IMAGE_STD_4BAND = [0.229, 0.224, 0.225, 0.225]
    
    # Supported formats (3-band: jpg/png, 4-band: tif)
    ALLOWED_FORMATS_3BAND = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    ALLOWED_FORMATS_4BAND = ['.tif', '.tiff', '.TIF', '.TIFF']
    ALLOWED_FORMATS = ALLOWED_FORMATS_3BAND + ALLOWED_FORMATS_4BAND
    
    # Channel configuration
    ALLOWED_CHANNELS = [3, 4]  # Support both 3-band RGB and 4-band RGBN
    
    # Detection parameters
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4
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