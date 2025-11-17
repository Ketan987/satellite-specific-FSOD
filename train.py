"""
Training script for FSOD
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np

# Memory optimization for Kaggle GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

from config import Config
from utils.coco_utils import COCODataset
from utils.data_loader import FSODDataset, collate_fn
from models.detector import FSODDetector, compute_detection_loss


def setup_multi_gpu(model, device):
    """Setup multi-GPU training if available"""
    # Note: DataParallel doesn't work well with mixed tensor/list data structures
    # Used in FSOD (support_boxes is a list). Single GPU training is more stable.
    if torch.cuda.device_count() > 1:
        print(f"⚠️  Found {torch.cuda.device_count()} GPUs but DataParallel has issues with mixed data types")
        print(f"   Using single GPU training for stability (GPU 0)")
    return model


def train_episode(model, episode_data, optimizer, device):
    """Train on a single episode"""
    model.train()
    
    # Move data to device
    support_images = episode_data['support_images'].to(device)
    support_boxes = [boxes.to(device) for boxes in episode_data['support_boxes']]
    support_labels = episode_data['support_labels'].to(device)
    query_images = episode_data['query_images'].to(device)
    query_boxes = [boxes.to(device) for boxes in episode_data['query_boxes']]
    query_labels = [labels.to(device) for labels in episode_data['query_labels']]
    
    # Forward pass
    optimizer.zero_grad()
    predictions = model(support_images, support_boxes, support_labels, query_images, n_way=episode_data.get('n_way', None))

    # Compute loss
    loss = compute_detection_loss(predictions, query_boxes, query_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate(model, val_loader, device, num_episodes=100):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    maps = []

    with torch.no_grad():
        for i, episode_data in enumerate(val_loader):
            if i >= num_episodes:
                break

            support_images = episode_data['support_images'].to(device)
            support_boxes = [boxes.to(device) for boxes in episode_data['support_boxes']]
            support_labels = episode_data['support_labels'].to(device)
            query_images = episode_data['query_images'].to(device)
            query_boxes = [boxes.to(device) for boxes in episode_data['query_boxes']]
            query_labels = [labels.to(device) for labels in episode_data['query_labels']]

            predictions = model(support_images, support_boxes, support_labels, query_images, n_way=episode_data.get('n_way', None))
            loss = compute_detection_loss(predictions, query_boxes, query_labels)
            total_loss += loss.item()

            # Compute per-episode mAP
            try:
                from utils.metrics import map_per_episode
                ep_map = map_per_episode(predictions, query_boxes, query_labels, n_way=episode_data.get('n_way', None))
            except Exception:
                ep_map = 0.0

            maps.append(ep_map)

    avg_loss = total_loss / num_episodes
    avg_map = float(sum(maps) / len(maps)) if len(maps) > 0 else 0.0
    return avg_loss, avg_map


def save_checkpoint(model, optimizer, episode, loss, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Handle DataParallel wrapper - get the underlying model
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'episode': episode,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pth')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main(args):
    # Configuration
    config = Config()
    
    # Override number of episodes for quick testing
    if args.num_episodes:
        config.NUM_EPISODES = args.num_episodes
        print(f"Override: NUM_EPISODES = {args.num_episodes}")
    
    # Device
    cli_device = getattr(args, 'device', None)
    if cli_device:
        device = torch.device(cli_device)
    else:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load COCO datasets
    print("Loading COCO datasets...")
    train_coco = COCODataset(
        config.TRAIN_COCO_JSON,
        config.TRAIN_IMAGE_DIR,
        config.ALLOWED_FORMATS
    )
    
    val_coco = COCODataset(
        config.VAL_COCO_JSON,
        config.VAL_IMAGE_DIR,
        config.ALLOWED_FORMATS
    )
    
    # Auto-detect input channels from first image
    print("Detecting image format...")
    first_img_path = train_coco.get_image_path(list(train_coco.imgs.keys())[0])
    input_channels = train_coco._get_num_channels(first_img_path)
    print(f"✅ Detected {input_channels}-band images")
    
    # Create FSOD datasets
    train_dataset = FSODDataset(
        train_coco,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        query_samples=config.QUERY_SAMPLES,
        image_size=config.IMAGE_SIZE,
        num_episodes=config.NUM_EPISODES
    )
    
    val_dataset = FSODDataset(
        val_coco,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        query_samples=config.QUERY_SAMPLES,
        image_size=config.IMAGE_SIZE,
        num_episodes=1000
    )
    
    # Data loaders
    num_workers = 0
    if getattr(args, 'test_episodes', None) is not None:
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    # Create model with correct input channels
    print(f"Creating model with {input_channels}-band input...")
    model = FSODDetector(
        feature_dim=config.FEATURE_DIM,
        embed_dim=config.EMBEDDING_DIM,
        image_size=config.IMAGE_SIZE,
        pretrained=args.pretrained,
        input_channels=input_channels
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model.backbone.features, 'gradient_checkpointing'):
        try:
            model.backbone.features.gradient_checkpointing = True
        except:
            pass
    
    # Setup multi-GPU if available
    model = setup_multi_gpu(model, device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('-inf')
    
    for episode, episode_data in enumerate(tqdm(train_loader, desc="Training")):
        # Train
        loss = train_episode(model, episode_data, optimizer, device)
        scheduler.step()
        
        # Log
        if (episode + 1) % config.LOG_FREQUENCY == 0:
            print(f"Episode {episode + 1}/{config.NUM_EPISODES}, Loss: {loss:.4f}")
        
        # Validate
        if (episode + 1) % config.SAVE_FREQUENCY == 0:
            val_loss, val_map = validate(model, val_loader, device, num_episodes=100)
            print(f"Validation Loss: {val_loss:.4f}, Validation mAP: {val_map:.4f}")

            # Save checkpoint
            save_checkpoint(model, optimizer, episode + 1, loss, config.CHECKPOINT_DIR)

            # Save best model by mAP
            if val_map > best_val_loss:
                best_val_loss = val_map
                best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model with val mAP: {val_map:.4f}")
    
    # Save final model
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    final_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    
    # Compute final mAP@50 on validation set
    print("\n" + "="*70)
    print("📊 COMPUTING FINAL VALIDATION METRICS (mAP@50)")
    print("="*70)
    model.eval()
    final_val_loss = 0.0
    final_maps = []
    
    with torch.no_grad():
        for i, episode_data in enumerate(tqdm(val_loader, desc="Final Validation", total=min(500, len(val_loader)))):
            if i >= 500:
                break
            
            support_images = episode_data['support_images'].to(device)
            support_boxes = [boxes.to(device) for boxes in episode_data['support_boxes']]
            support_labels = episode_data['support_labels'].to(device)
            query_images = episode_data['query_images'].to(device)
            query_boxes = [boxes.to(device) for boxes in episode_data['query_boxes']]
            query_labels = [labels.to(device) for labels in episode_data['query_labels']]
            
            predictions = model(support_images, support_boxes, support_labels, query_images, n_way=episode_data.get('n_way', None))
            loss = compute_detection_loss(predictions, query_boxes, query_labels)
            final_val_loss += loss.item()
            
            # Compute mAP@50
            try:
                from utils.metrics import map_per_episode
                ep_map = map_per_episode(predictions, query_boxes, query_labels, n_way=episode_data.get('n_way', None), iou_thr=0.5)
                final_maps.append(ep_map)
            except Exception as e:
                final_maps.append(0.0)
    
    final_avg_loss = final_val_loss / min(500, len(val_loader))
    final_avg_map = float(np.mean(final_maps)) if len(final_maps) > 0 else 0.0
    
    print("\n" + "="*70)
    print("✅ FINAL VALIDATION RESULTS")
    print("="*70)
    print(f"Final Validation Loss: {final_avg_loss:.4f}")
    print(f"Final mAP@50 (Validation Set): {final_avg_map:.4f}")
    print(f"Evaluated on: {min(500, len(val_loader))} episodes")
    print(f"Image Format: {input_channels}-band")
    print("="*70)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FSOD model")
    parser.add_argument('--config', type=str, default=None, help="Path to config file")
    parser.add_argument('--num_episodes', type=int, default=None, 
                       help="Number of training episodes (overrides config)")
    parser.add_argument('--test_episodes', type=int, default=None, 
                       help="Run a small number of episodes for testing (overrides num_episodes for loader)" )
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained backbone weights (may download)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use: 'cpu' or 'cuda'")
    args = parser.parse_args()
    
    main(args)