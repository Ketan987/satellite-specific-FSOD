# Multi-Band Image Processing Pipeline

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMAGE LOADING STAGE                           │
└─────────────────────────────────────────────────────────────────────┘

    JPG/PNG Files                    TIFF Files
        │                                │
        ├─ [H, W, 3]                     ├─ [4, H, W] or [H, W, 4]
        │  RGB (8-bit)                   │  RGBN (8/16-bit)
        │                                │
        ▼                                ▼
    Image.open()                  rasterio.open()
    .convert('RGB')               .read() bands
        │                                │
        └────────────────┬───────────────┘
                         │
                    Auto-Detect Channels
                    _get_num_channels()
                         │
                    ┌────┴────┐
                    │          │
                 num_ch=3   num_ch=4
                    │          │
                    ▼          ▼
        ┌──────────────────────────────┐
        │   Image Transformation       │
        │  _get_image_transforms()     │
        └──────────────────────────────┘
                    │
            ┌───────┴────────┐
            │                │
        [3-band]         [4-band]
            │                │
            ▼                ▼
        Normalize        Normalize
        Mean=[0.485,     Mean=[0.485,
              0.456,           0.456,
              0.406]           0.406, 0.406]
        Std=[0.229,      Std=[0.229,
             0.224,           0.224,
             0.225]           0.225, 0.225]
            │                │
            └────────┬───────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │   Normalized Tensor          │
        │   [C, H, W]                  │
        │   C ∈ {3, 4}                 │
        └──────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION STAGE                        │
└─────────────────────────────────────────────────────────────────────┘

        Input: [N*K, C, H, W]  (C ∈ {3, 4})
                  │
                  ▼
        ┌──────────────────────────────┐
        │   ResNet50 Conv1             │
        │   Adaptive Input Layer       │
        │   [C, 7, 7] → [64, 7, 7]    │
        └──────────────────────────────┘
                  │
            ┌─────┴─────┐
            │           │
         3-band      4-band
            │           │
        [3 weights]  Weight Adaptation:
            │         [3 weights] +
            │         [Avg(weights)]
            ▼           │
        Conv1 →         ▼ Conv1 (4-ch)
                  │
                  ▼
        ResNet-50 Layers 1-3
        (Shared architecture)
                  │
                  ▼
        ┌──────────────────────────────┐
        │   Backbone Features          │
        │   [B, 2048, H/32, W/32]      │
        │   (Channel agnostic)         │
        └──────────────────────────────┘
                  │
                  ▼
        ┌──────────────────────────────┐
        │   Feature Embedding          │
        │   Conv projections           │
        │   [2048] → [512]            │
        └──────────────────────────────┘
                  │
                  ▼
        ┌──────────────────────────────┐
        │   Embedded Features          │
        │   [B, 512, H/32, W/32]       │
        │   (Channel agnostic)         │
        └──────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                     DETECTION STAGE (QUERY)                          │
└─────────────────────────────────────────────────────────────────────┘

    Support Features          Query Features
    [N*K, 512, H/32, W/32]    [Q, 512, H/32, W/32]
            │                        │
            ▼                        ▼
        ROI Pooling          Region Proposals
        [N*K, 512, 7, 7]    [num_proposals]
            │                        │
            │◄──────────────────────►│
            │                        │
            ▼                        ▼
        Similarity Matching    Similarity Matching
        Metric Learning        Metric Learning
            │                        │
            └────────────┬───────────┘
                         │
                         ▼
        ┌──────────────────────────────┐
        │   Class Similarity Scores    │
        │   [num_proposals, n_way]     │
        └──────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────┐
        │   Box Regression             │
        │   Box Refinement             │
        └──────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────┐
        │   Final Detections           │
        │   [boxes, scores, classes]   │
        └──────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                   VALIDATION & ERROR HANDLING                        │
└─────────────────────────────────────────────────────────────────────┘

Channel Validation Points:
├─ get_image(): Detects channel count
├─ _get_image_transforms(): Validates channel support (3 or 4)
├─ FSODDataset.__getitem__(): Returns num_channels in batch
├─ ResNet50Backbone.forward(): Validates input channels match
├─ FSODDetector.forward(): Validates support/query channel match
└─ prepare_inference_data(): Returns detected num_channels

Error Prevention:
├─ Unsupported channels: ValueError with clear message
├─ Mixed channels in episode: ValueError (channel mismatch)
├─ Invalid TIFF bands: ValueError (must be 4 bands)
└─ 16-bit data: Auto-normalized to 0-255 range
```

## Training Workflow

```
┌────────────────────────────────────────────────────────────┐
│                    START TRAINING                           │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Load COCO Dataset              │
        │  - train_coco.json              │
        │  - val_coco.json                │
        │  Allowed formats: [*.jpg, *.png,│
        │                    *.tif, *.tiff]
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Auto-detect Image Format       │
        │  first_img = dataset[0]         │
        │  num_ch = _get_num_channels(...)│
        │  ✓ Detected 3-band or 4-band    │
        └─────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │          │
               3-band=True  4-band=True
                    │          │
                    ▼          ▼
            Create ResNet50 with C=3   C=4
                    │          │
                    └────┬─────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Create FSODDetector            │
        │  input_channels = num_ch        │
        │  Model adapted automatically    │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Training Loop                  │
        │  for episode in dataset:        │
        │    - Get batch [N*K, C, H, W]   │
        │    - Forward pass               │
        │    - Compute loss               │
        │    - Backward pass              │
        │    - Update weights             │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Periodic Validation            │
        │  Compute mAP on validation set  │
        │  Save checkpoints               │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Training Complete              │
        │  Report: Final Loss, mAP, Format│
        └─────────────────────────────────┘
```

## Inference Workflow

```
┌────────────────────────────────────────────────────────────┐
│                    START INFERENCE                          │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Initialize FSODInference       │
        │  Load model checkpoint          │
        │  Default: input_channels = 3    │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Load Support Images            │
        │  Parse paths or PIL Images      │
        │  For each image:                │
        │    img, num_ch = _load_image()  │
        │    Detect channels from first   │
        └─────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │          │
            num_ch != 3    num_ch == 3
                    │          │
                    ▼          ▼
            Recreate model   Skip
            with C=4 or 3
                    │          │
                    └────┬─────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Prepare Query Image            │
        │  Load & normalize with same C   │
        │  Validate channel match         │
        └─────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │          │
              Match ✓      Mismatch ✗
                    │          │
                    ▼          ▼
              Continue    Raise Error
                    │      (clear msg)
                    │
                    ▼
        ┌─────────────────────────────────┐
        │  Create Tensors                 │
        │  [N*K, C, H, W]                 │
        │  [1, C, H, W]                   │
        │  C matches for both             │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Model Inference                │
        │  Forward pass                   │
        │  Generate detections            │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Post-process Results           │
        │  NMS filtering                  │
        │  Format conversion              │
        │  [x,y,w,h] → [x1,y1,x2,y2]    │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │  Return Detections              │
        │  Format:                        │
        │  [{bbox, score, class_name}]    │
        └─────────────────────────────────┘
```

## Memory Layout During Forward Pass

```
BATCH PROCESSING: [N*K, C, H, W] where N=5 ways, K=2 shot, C∈{3,4}

Support Images:
┌─────────────────────────────────────────────────┐
│ Cat1_Img1  Cat1_Img2  Cat2_Img1  Cat2_Img2 ... │
│ [3, 256, 256] or [4, 256, 256]  (each)          │
│                                                 │
│ Stacked: [10, 3, 256, 256] or [10, 4, 256, 256]│
└─────────────────────────────────────────────────┘
         │
         ▼
     ResNet50 Backbone
         │
         ▼
  [10, 2048, 8, 8]  ← Spatial reduction 32x
         │
         ▼
     Feature Embedding
         │
         ▼
  [10, 512, 8, 8]
         │
         ▼
     ROI Pooling (one feature per support image)
         │
         ▼
  [10, 512, 7, 7]
         │
         ▼
     Flatten & Detection Head
         │
         ▼
  [10, 512]  ← Support feature vectors


Query Image: [1, C, H, W]
         │
         ▼
    Same backbone as support
         │
         ▼
  [1, 2048, 8, 8]  ← Channel-invariant
         │
         ▼
  [1, 512, 8, 8]
         │
         ▼
 Proposal Generation on feature map
         │
         ▼
 Extract features for each proposal
         │
         ▼
 Similarity matching with support features
 [num_proposals, 512] vs [10, 512]
         │
         ▼
 Class-level similarity
 [num_proposals, 5]  ← 5 classes
```

## Channel Compatibility Matrix

```
                ┌──────────────────────────────────────┐
                │     QUERY IMAGE CHANNELS             │
                │    3-Band        4-Band              │
    ┌───────────┼──────────────────────────────────────┤
    │   3-Band  │     ✅ OK         ❌ MISMATCH       │
S   │           │                                      │
U   ├───────────┼──────────────────────────────────────┤
P   │   4-Band  │     ❌ MISMATCH    ✅ OK             │
P   │           │                                      │
O   └───────────┴──────────────────────────────────────┘
R   
T   ✅ Valid: All images same channels
    ❌ Error: Mixed channels detected
    
    Error message:
    "Channel mismatch: support images have 3 channels 
     but query image has 4 channels. 
     All images must have the same number of bands."
```

---

This pipeline ensures:
1. Automatic format detection
2. Proper tensor shape validation
3. No size mismatch errors
4. Flexible input support (3 or 4 bands)
5. Clear error messages
