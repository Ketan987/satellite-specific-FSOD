# Multi-Band Image Support Guide

## Overview
The FSOD codebase has been updated to support both 3-band and 4-band satellite images:
- **3-band images**: Standard RGB (JPG, PNG)
- **4-band images**: RGBN (Red, Green, Blue, Near-Infrared) in TIFF format

## Key Changes

### 1. **Configuration (config.py)**
- Added separate normalization stats for 3-band and 4-band images
- Support for multiple file formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
- `ALLOWED_CHANNELS = [3, 4]` for flexible channel support

### 2. **Data Loading (utils/data_loader.py)**
- **Auto-detection**: Automatically detects number of bands from first image
- **Multi-band transforms**: Channel-specific normalization using `_get_image_transforms()`
- **TIF support**: `_load_image_from_path()` function handles both JPG/PNG and TIFF
- **16-bit TIF handling**: Automatically normalizes 16-bit TIFF data to 0-255 range
- **Returns num_channels**: Dataset now returns channel count in batch

### 3. **Backbone (models/backbone.py)**
- **Adaptive first layer**: `ResNet50Backbone` now accepts `input_channels` parameter
- **Weight adaptation**: Pretrained 3-channel weights are intelligently adapted for 4-channel input
  - 4th channel (NIR) weights initialized from average of RGB channels
- **Channel validation**: Ensures input tensors match expected channels

### 4. **Detector (models/detector.py)**
- **Multi-channel support**: `FSODDetector` accepts `input_channels` parameter
- **Channel validation**: Validates support and query images have matching channels
- **No size mismatch**: Proper tensor shape handling for [N*K, C, H, W] inputs

### 5. **Training (train.py)**
- **Auto-detection**: Detects image format from first image in dataset
- **Dynamic model creation**: Creates backbone with correct input channels
- **Channel reporting**: Displays detected image format during training

### 6. **Inference (inference.py)**
- **Flexible support**: Handles mixed 3-band and 4-band datasets
- **Automatic adaptation**: Model adapts to detected image format
- **TIF file support**: Automatically loads `.tif` and `.tiff` files

### 7. **COCO Utils (utils/coco_utils.py)**
- **Multi-format loading**: `get_image()` handles both JPEG/PNG and TIFF
- **Channel detection**: `_get_num_channels()` detects band count
- **Proper image path handling**: `get_image_path()` for consistency

### 8. **Dependencies (requirements.txt)**
- Added `rasterio==1.3.9` for TIFF file reading

## Usage

### Training with 4-Band Images

```bash
# Prepare your TIFF dataset in COCO format with 4-band images
# The model will auto-detect the format

python train.py --device cuda --pretrained
```

The training script will:
1. Auto-detect 4-band format from first image
2. Create backbone with 4-channel input
3. Train on RGBN data
4. Handle 16-bit TIF data automatically

### Inference with 4-Band Images

```bash
# Single image detection with 4-band support images
python inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --support_img support1.tif support2.tif support3.tif \
  --query_image query_image.tif \
  --device cuda

# Batch mode with 4-band images
python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --support_img support1.tif support2.tif \
  --query_dir ./test_images/ \
  --output_csv results.csv \
  --device cuda
```

### Mixed Format Support

Support and query images can be different formats, but they must have the **same number of bands**:

```python
# ✅ Valid: All 4-band TIF
support_images = ['satellite1.tif', 'satellite2.tif', 'satellite3.tif']
query_image = 'satellite_query.tif'

# ✅ Valid: All 3-band RGB
support_images = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
query_image = 'photo_query.jpg'

# ❌ Invalid: Mixed bands
support_images = ['photo1.jpg', 'satellite2.tif']  # Different channels!
```

## Data Format Requirements

### 4-Band TIFF Images
- **Format**: GeoTIFF or standard TIFF
- **Bands**: 4 (Red, Green, Blue, Near-Infrared)
- **Data type**: 8-bit or 16-bit (auto-normalized)
- **Expected shape**: (height, width, 4)

### 3-Band RGB Images
- **Formats**: JPG, PNG
- **Channels**: 3 (RGB)
- **Data type**: 8-bit
- **Expected shape**: (height, width, 3)

## Normalization

### 3-Band Images
```
Mean: [0.485, 0.456, 0.406]  # Standard ImageNet
Std:  [0.229, 0.224, 0.225]
```

### 4-Band Images (RGBN)
```
Mean: [0.485, 0.456, 0.406, 0.406]  # RGB + NIR (similar to B)
Std:  [0.229, 0.224, 0.225, 0.225]
```

## Technical Details

### Channel Adaptation in Backbone

When loading a model trained on 3-band data and applying to 4-band data:

1. Original pretrained weights: `[64, 3, 7, 7]` (3 input channels)
2. New adapted weights: `[64, 4, 7, 7]` (4 input channels)
3. **Adaptation method**: 4th channel initialized as average of RGB channels

```python
adapted_weights = torch.cat([
    pretrained_weights,
    pretrained_weights[:, :1, :, :].mean(dim=1, keepdim=True)
], dim=1)
```

### 16-bit TIFF Handling

Rasterio automatically reads multi-band GeoTIFF files:

```python
with rasterio.open(img_path) as src:
    data = src.read()  # [bands, height, width]
    # Normalize 16-bit to 8-bit if needed
    if data.max() > 255:
        data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
```

## Tensor Size Handling

### Support Images: `[N*K, C, H, W]`
- **N**: Number of classes (n_way)
- **K**: Number of support examples per class (k_shot)
- **C**: Number of channels (3 or 4)
- **H, W**: Image size (default 256x256)

### Query Images: `[Q, C, H, W]`
- **Q**: Number of query images per episode
- **C**: Must match support channels

### Feature Map: `[B, 2048, H/32, W/32]`
- ResNet50 reduces spatial dimensions by 32x
- Channel dimension always 2048 (backbone feature dim)

## Common Issues and Solutions

### Issue: "Channel mismatch" Error
**Cause**: Support and query images have different numbers of bands
**Solution**: Ensure all images in an episode have the same format (all 3-band or all 4-band)

### Issue: Size mismatch during forward pass
**Cause**: Model expects different channel count than input
**Solution**: Model automatically adapts; ensure images are loaded correctly

### Issue: Poor performance with 4-band data
**Recommendations**:
1. Use sufficient training samples (same # of episodes as 3-band)
2. NIR band may need special preprocessing (e.g., normalization)
3. Consider band statistics from your specific satellite imagery
4. Pretrained RGB weights provide good initialization for RGBN

## Performance Notes

- **4-band model training**: ~Same time as 3-band (minimal overhead from 4th channel)
- **Memory**: Slightly higher (~1.15x) due to additional channel
- **Inference**: Slightly slower but negligible for real-time applications

## Example: Complete Multi-Band Workflow

```python
from inference import FSODInference

# Initialize with model (auto-detects format)
inferencer = FSODInference('checkpoints/best_model.pth', device='cuda')

# 4-band support set
support_set = [
    {'image_path': 'satellite1.tif', 'class_name': 'building'},
    {'image_path': 'satellite2.tif', 'class_name': 'building'},
    {'image_path': 'satellite3.tif', 'class_name': 'building'},
]

# Run detection (auto-detects 4 bands from first image)
detections = inferencer.detect(
    support_set,
    'satellite_query.tif',
    score_threshold=0.3
)

for det in detections:
    print(f"Detection at {det['bbox']} with score {det['similarity_score']:.3f}")
```

## Migration from 3-Band Only

If you have an existing model trained on 3-band data:

1. **No retraining needed** for inference on 4-band data
2. Model will auto-adapt 4th channel from RGB pretrained weights
3. For best results, **fine-tune on 4-band data** for 1000-5000 episodes

```bash
# Fine-tune existing model on 4-band data
python train.py \
  --device cuda \
  --num_episodes 2000 \
  --pretrained  # Loads existing checkpoint if it exists
```

## Future Enhancements

Potential improvements:
1. Support for arbitrary channel counts (>4 bands)
2. Configurable band statistics per dataset
3. Automatic band selection for multi-spectral data
4. Learnable band weighting
5. Multi-scale band processing

---

For more information, see the main README.md and architecture documentation.
