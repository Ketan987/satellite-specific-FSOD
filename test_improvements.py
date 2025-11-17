#!/usr/bin/env python3
"""
Test script to verify improvements without running full training.
Tests:
  1. Model forward/backward pass with synthetic data
  2. Loss computation and focal loss
  3. Data loading with augmentation
  4. Metrics computation
  5. Inference pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add repo to path
sys.path.insert(0, '/home/workspace/workspace/v1/fsod')

from models.detector import FSODDetector, compute_detection_loss, focal_loss_ce
from utils.metrics import map_per_episode, iou_xywh
from config import Config


def test_forward_pass():
    """Test model forward pass with synthetic data"""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass with Synthetic Data")
    print("="*60)
    
    config = Config()
    device = torch.device('cpu')  # CPU for testing
    
    # Create model (no pretrained weights)
    model = FSODDetector(
        feature_dim=config.FEATURE_DIM,
        embed_dim=config.EMBEDDING_DIM,
        image_size=config.IMAGE_SIZE,
        pretrained=False
    ).to(device)
    
    model.train()
    
    # Create synthetic data
    batch_size = 2  # 2 query images
    n_way = config.N_WAY
    k_shot = config.K_SHOT
    
    support_images = torch.randn(n_way * k_shot, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    support_boxes = [torch.tensor([[50.0, 50.0, 100.0, 100.0]]) for _ in range(n_way * k_shot)]
    support_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
    
    query_images = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    
    # Forward pass
    try:
        with torch.no_grad():
            predictions = model.forward(
                support_images,
                support_boxes,
                support_labels,
                query_images,
                n_way=n_way
            )
        
        print(f"✓ Forward pass succeeded")
        print(f"  - Predictions: {len(predictions)} images")
        for i, pred in enumerate(predictions):
            print(f"    Image {i}: boxes={pred['boxes'].shape}, scores={pred['scores'].shape}")
        
        return True, predictions
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss_computation(predictions):
    """Test loss computation with focal loss"""
    print("\n" + "="*60)
    print("TEST 2: Loss Computation with Focal Loss")
    print("="*60)
    
    if predictions is None:
        print("✗ Skipping (no predictions)")
        return False
    
    device = torch.device('cpu')
    
    # Create synthetic GT data
    num_query = len(predictions)
    target_boxes = []
    target_labels = []
    
    for _ in range(num_query):
        # Add some ground truth boxes
        boxes = torch.tensor([
            [100.0, 100.0, 50.0, 50.0],
            [300.0, 300.0, 80.0, 80.0],
            [200.0, 150.0, 60.0, 70.0],
        ], device=device)
        labels = torch.tensor([0, 1, 2], device=device)
        target_boxes.append(boxes)
        target_labels.append(labels)
    
    try:
        loss = compute_detection_loss(
            predictions,
            target_boxes,
            target_labels,
            iou_threshold=0.3,
            box_loss_weight=1.0,
            use_focal=True
        )
        
        print(f"✓ Loss computation succeeded")
        print(f"  - Loss value: {loss.item():.4f}")
        print(f"  - Loss is finite: {torch.isfinite(loss).item()}")
        
        # Test focal loss directly
        logits = torch.randn(100, 5, device=device)
        targets = torch.randint(0, 5, (100,), device=device)
        focal_loss = focal_loss_ce(logits, targets, alpha=0.25, gamma=2.0)
        print(f"  - Focal loss test: {focal_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_computation():
    """Test mAP metric computation"""
    print("\n" + "="*60)
    print("TEST 3: Metrics Computation (mAP)")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create synthetic predictions and ground truth
    num_images = 3
    n_way = 5
    
    predictions = []
    target_boxes = []
    target_labels = []
    
    for img_id in range(num_images):
        # Predictions
        pred_boxes = torch.tensor([
            [50.0, 50.0, 100.0, 100.0],
            [300.0, 300.0, 80.0, 80.0],
            [200.0, 150.0, 60.0, 70.0],
        ], device=device)
        pred_scores = torch.tensor([0.9, 0.7, 0.5], device=device)
        pred_classes = torch.tensor([0, 1, 2], device=device)
        
        predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'pred_classes': pred_classes
        })
        
        # Ground truth
        boxes = torch.tensor([
            [55.0, 55.0, 95.0, 95.0],
            [310.0, 310.0, 70.0, 70.0],
        ], device=device)
        labels = torch.tensor([0, 1], device=device)
        
        target_boxes.append(boxes)
        target_labels.append(labels)
    
    try:
        mAP = map_per_episode(predictions, target_boxes, target_labels, n_way=n_way, iou_thr=0.5)
        print(f"✓ Metrics computation succeeded")
        print(f"  - mAP@50: {mAP:.4f}")
        print(f"  - mAP is in valid range: {0.0 <= mAP <= 1.0}")
        return True
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi_pooling():
    """Test improved ROI pooling"""
    print("\n" + "="*60)
    print("TEST 4: ROI Pooling Fix")
    print("="*60)
    
    from models.backbone import ROIPooling
    
    device = torch.device('cpu')
    
    try:
        roi_pool = ROIPooling(output_size=7)
        
        # Create synthetic feature map
        features = torch.randn(1, 256, 16, 16, device=device)
        image_size = 512
        
        # Create small boxes (edge case that would fail with truncation)
        boxes = [torch.tensor([
            [10.0, 10.0, 5.0, 5.0],     # Small box
            [100.0, 100.0, 50.0, 50.0],  # Normal box
        ], device=device)]
        
        roi_features = roi_pool(features, boxes, image_size)
        
        print(f"✓ ROI Pooling succeeded")
        print(f"  - Input features: {features.shape}")
        print(f"  - ROI features shape: {roi_features.shape}")
        print(f"  - Expected: [2, 256, 7, 7]")
        assert roi_features.shape == (2, 256, 7, 7), f"Unexpected shape: {roi_features.shape}"
        
        return True
    except Exception as e:
        print(f"✗ ROI Pooling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass():
    """Test backward pass and optimization step"""
    print("\n" + "="*60)
    print("TEST 5: Backward Pass & Optimization")
    print("="*60)
    
    config = Config()
    device = torch.device('cpu')
    
    model = FSODDetector(
        feature_dim=config.FEATURE_DIM,
        embed_dim=config.EMBEDDING_DIM,
        image_size=config.IMAGE_SIZE,
        pretrained=False
    ).to(device)
    
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    try:
        # Create synthetic data
        n_way = 3
        k_shot = 2
        support_images = torch.randn(n_way * k_shot, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
        support_boxes = [torch.tensor([[50.0, 50.0, 100.0, 100.0]]) for _ in range(n_way * k_shot)]
        support_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        
        query_images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
        
        # Forward
        predictions = model.forward(
            support_images,
            support_boxes,
            support_labels,
            query_images,
            n_way=n_way
        )
        
        # Create GT
        target_boxes = [torch.tensor([[100.0, 100.0, 50.0, 50.0]], device=device)]
        target_labels = [torch.tensor([0], device=device)]
        
        # Compute loss
        loss = compute_detection_loss(predictions, target_boxes, target_labels, use_focal=True)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Optimization step
        optimizer.step()
        
        print(f"✓ Backward pass succeeded")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Gradient norm: {grad_norm:.6f}")
        print(f"  - Gradient is non-zero: {grad_norm > 0.0}")
        
        return True
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("FSOD IMPROVEMENTS TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Forward pass
    success, predictions = test_forward_pass()
    results['forward_pass'] = success
    
    # Test 2: Loss computation
    if predictions is not None:
        success = test_loss_computation(predictions)
        results['loss_computation'] = success
    
    # Test 3: Metrics
    success = test_metrics_computation()
    results['metrics'] = success
    
    # Test 4: ROI Pooling
    success = test_roi_pooling()
    results['roi_pooling'] = success
    
    # Test 5: Backward pass
    success = test_backward_pass()
    results['backward_pass'] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*60)
    if passed == total:
        print("ALL TESTS PASSED! ✓")
        print("\nThe improvements are ready for training.")
        return 0
    else:
        print(f"SOME TESTS FAILED ({total - passed} failures)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
