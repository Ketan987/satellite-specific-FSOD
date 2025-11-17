# FSOD Inference Guide

This document explains the two inference modes available in the FSOD system.

## Quick Start

### Option 1: Run Complete Test Pipeline (Recommended for CPU testing)

```bash
python test_training_and_inference.py
```

This script will:
1. ✅ Check for required data files
2. ✅ Train model for 100 episodes on CPU
3. ✅ Save trained model to `checkpoints/best_model.pth`
4. ✅ Test single image inference with visualization
5. ✅ Test batch inference with CSV output

## Two Inference Modes

### Mode 1: SINGLE IMAGE MODE

Process a single query image with visualization. Perfect for testing and debugging.

**Features:**
- Process one query image at a time
- Automatically annotate bounding boxes on the image
- Print detection results in table format
- Save annotated image to output folder

**Command:**
```bash
python inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --query_image path/to/query_image.jpg \
  --support_config support_set_example.json \
  --output_dir output \
  --device cpu
```

**Output:**
```
============================================================
SINGLE MODE: Running detection on single query image
============================================================

Query Image: path/to/query_image.jpg
Number of objects detected: 3

Detection Results:
────────────────────────────────────────────────────────────────────────────────
#    x_min       y_min       x_max       y_max       Score    Class          
────────────────────────────────────────────────────────────────────────────────
1    100.5       150.2       250.8       300.1       0.8723   car            
2    50.1        75.3        180.9       200.5       0.7654   car            
3    300.2       100.1       450.5       250.3       0.6234   car            
────────────────────────────────────────────────────────────────────────────────

✅ Detection completed!
📸 Annotated image saved to: output/query_image_detected.jpg
```

**Output Files:**
- `output/query_image_detected.jpg` - Annotated image with colored bounding boxes

---

### Mode 2: BATCH MODE

Process multiple query images from a directory and save results to CSV.

**Features:**
- Process entire directory of images
- Save results in standardized CSV format
- Fast processing with progress bar
- Handles missing/invalid images gracefully

**Command:**
```bash
python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --query_dir path/to/images/ \
  --support_config support_set_example.json \
  --output_csv batch_results.csv \
  --device cpu
```

**CSV Output Format:**
```csv
x_min,y_min,x_max,y_max,filename,similarity_score
100.5,150.2,250.8,300.1,image001.jpg,0.8723
50.1,75.3,180.9,200.5,image001.jpg,0.7654
300.2,100.1,450.5,250.3,image002.jpg,0.6234
...
```

**Output Files:**
- `batch_results.csv` - All detections in CSV format

**CSV Columns:**
| Column | Description | Type | Example |
|--------|-------------|------|---------|
| x_min | Minimum x coordinate | float | 100.5 |
| y_min | Minimum y coordinate | float | 150.2 |
| x_max | Maximum x coordinate | float | 250.8 |
| y_max | Maximum y coordinate | float | 300.1 |
| filename | Query image filename | string | image001.jpg |
| similarity_score | Detection confidence | float | 0.8723 |

---

## Support Set Configuration

Both modes require a support set (few-shot examples). Create a JSON file like this:

**File: `support_set_example.json`**
```json
[
    {
        "image_path": "path/to/support_image1.jpg",
        "bbox": [x, y, width, height],
        "class_name": "cat"
    },
    {
        "image_path": "path/to/support_image2.jpg",
        "bbox": [x, y, width, height],
        "class_name": "cat"
    },
    {
        "image_path": "path/to/support_image3.jpg",
        "bbox": [x, y, width, height],
        "class_name": "cat"
    }
]
```

**Notes:**
- `image_path`: Absolute or relative path to support image
- `bbox`: [x, y, width, height] format (top-left corner + dimensions)
- `class_name`: Category name (all in this example are "cat")
- You can have multiple classes by adding more entries

---

## Training for Testing

### Quick Test (100 episodes)
```bash
python train.py --device cpu --num_episodes 100 --pretrained
```
**Time:** ~5-10 minutes on CPU  
**Output:** `checkpoints/best_model.pth`

### Full Training (10000 episodes)
```bash
python train.py --device cuda --num_episodes 10000 --pretrained
```
**Time:** ~2-5 hours on GPU  
**Output:** `checkpoints/best_model.pth`

---

## Common Parameters

### All Modes
- `--model_path`: Path to trained model checkpoint (required)
- `--device`: Device to use - 'cpu' or 'cuda' (default: cpu)
- `--score_threshold`: Detection confidence threshold (default: 0.3)
- `--nms_threshold`: NMS threshold (default: 0.4)
- `--max_detections`: Maximum detections per image (default: 100)

### Single Mode
- `--query_image`: Path to single query image (required)
- `--support_config`: Path to support set JSON (recommended)
- `--output_dir`: Directory for output images (default: output)

### Batch Mode
- `--query_dir`: Directory with query images (required)
- `--support_config`: Path to support set JSON (required)
- `--output_csv`: Output CSV filename (default: batch_results.csv)

---

## Examples

### Example 1: Single Image with CLI Arguments
```bash
python inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --query_image data/test_image.jpg \
  --support_image data/support1.jpg \
  --support_bbox 10 20 100 150 \
  --class_name cat \
  --output_dir output \
  --device cpu
```

### Example 2: Batch Processing All Validation Images
```bash
python inference.py --mode batch \
  --model_path checkpoints/best_model.pth \
  --query_dir data/val_images/ \
  --support_config support_set_example.json \
  --output_csv validation_results.csv \
  --score_threshold 0.5 \
  --device cuda
```

### Example 3: Test Mode with Low Thresholds
```bash
python inference.py --mode single \
  --model_path checkpoints/best_model.pth \
  --query_image test.jpg \
  --support_config support_set.json \
  --score_threshold 0.1 \
  --nms_threshold 0.3 \
  --device cpu
```

---

## Output Formats

### Single Mode Output Structure
```
output/
├── query_image_detected.jpg      # Annotated image
├── query_image2_detected.jpg     # Another annotated image
└── ...
```

**Image Annotations:**
- Colored bounding boxes (unique color per detection)
- Class name and similarity score label
- Multiple colors: Red, Green, Blue, Yellow, Magenta, Cyan

### Batch Mode Output Structure
```
batch_results.csv                # CSV with all detections
```

**CSV Format:**
- Header row: x_min, y_min, x_max, y_max, filename, similarity_score
- One row per detection
- Same image can appear multiple times (multiple detections)

---

## Troubleshooting

### No detections found
- Lower `--score_threshold` (try 0.1 instead of 0.3)
- Check support set matches query images well
- Ensure model is properly trained

### Error: Image not found
- Verify paths are correct
- Use absolute paths if relative paths don't work
- Check image file exists and is readable

### Out of memory on GPU
- Use `--device cpu` for smaller batches
- Reduce image size or use fewer support examples

### Slow inference on CPU
- This is normal! CPU inference is much slower than GPU
- Consider using GPU if available
- Use batch mode for processing multiple images

---

## Advanced Usage

### Processing Results from CSV
```python
import pandas as pd

# Load results
df = pd.read_csv('batch_results.csv')

# Group by filename
grouped = df.groupby('filename')

for filename, detections in grouped:
    print(f"Image {filename}: {len(detections)} detections")
    for _, det in detections.iterrows():
        print(f"  Box: ({det['x_min']}, {det['y_min']}) - ({det['x_max']}, {det['y_max']})")
        print(f"  Score: {det['similarity_score']:.4f}")
```

### Filtering Results
```bash
# Extract only high-confidence detections
python -c "
import pandas as pd
df = pd.read_csv('batch_results.csv')
high_conf = df[df['similarity_score'] > 0.7]
high_conf.to_csv('high_confidence_results.csv', index=False)
print(f'Filtered to {len(high_conf)} high-confidence detections')
"
```

---

## Key Differences Summary

| Feature | Single Mode | Batch Mode |
|---------|-------------|-----------|
| Input | One image | Directory of images |
| Output | Annotated JPG + print | CSV file |
| Use Case | Testing/Debugging | Large-scale processing |
| Speed | Slower (interactive) | Fast (automated) |
| Visualization | Yes (automatic) | No (use output to visualize) |

---

## Configuration

All detection parameters can be customized:

- **score_threshold**: Lower = more detections (may include false positives)
- **nms_threshold**: Lower = more isolated boxes (less overlap)
- **max_detections**: Limit detections per image

Recommended starting values:
- `--score_threshold 0.3`
- `--nms_threshold 0.4`
- `--max_detections 100`

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify model is properly trained with `test_training_and_inference.py`
3. Test with example data first before using custom data
