# FSOD Training & Architecture Guide

This document is the single reference for preparing data, training the few-shot detector, and understanding the model architecture shipped in this repository. Use it together with `readme.MD`, which covers day-to-day inference usage.

---

## 1. Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA 11.8 (CPU training works for smoke tests)
- JPEG datasets following COCO annotation format
- Recommended GPU memory: 12 GB for full-resolution training with MAML + RPN

Install dependencies once per environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Data Preparation

Place datasets under `data/` using these paths (defaults defined in `config.py`):

```
data/
├── train_coco.json
├── val_coco.json
├── train_images/
├── val_images/
```

Each annotation entry must include `bbox` values in `[x, y, width, height]` format. Only `.jpg/.jpeg` files are supported.

---

## 3. Training Workflow

1. **Configure Episodes** – edit `config.py` for `N_WAY`, `K_SHOT`, `QUERY_SAMPLES`, image size, and learning rates. Leave `USE_MAML=True` to enable inner-loop adaptation.
2. **Launch Training** – choose CPU or GPU:

```bash
source venv/bin/activate
python3 train.py --device cuda --pretrained
```

Optional flags:
- `--num_episodes 500` for quick smoke tests
- `--device cpu` when CUDA is unavailable

3. **Monitoring** – loss prints every `LOG_FREQUENCY` episodes; checkpoints and validation happen every `SAVE_FREQUENCY` episodes under `checkpoints/`.
4. **Resume/Finetune** – pass `--pretrained` to reuse ImageNet weights or edit `train.py` to load a specific checkpoint.

---

## 4. Architecture Overview

The detector follows a meta-learning variant of Faster R-CNN tailored for few-shot satellite imagery:

1. **Backbone + Embedding** – `models/backbone.py` wraps ResNet-50 and a 1×1 projection to a 256-D feature map.
2. **Region Proposal Network (RPN)** – `models/rpn.py` generates context-aware proposals using configurable anchor scales/ratios (`config.ANCHOR_SCALES`, `config.ANCHOR_RATIOS`). It provides both proposals and an auxiliary loss that feeds into `compute_detection_loss`.
3. **ROI Pooling & Detection Head** – `models/detector.py` pools each proposal, passes it through a two-layer MLP, and predicts class-aware box refinements plus objectness scores.
4. **Similarity Matcher** – `models/similarity.py` compares query proposals to support prototypes with cosine similarity and class-level logits to score detections.
5. **MAML Inner Loop** – during training (enabled via `USE_MAML`), each episode performs a fast adaptation on the support set before computing the meta-loss on the query samples, improving generalization to novel classes.

---

## 5. Inference Notes

- Use `inference.py` (documented in `readme.MD`) with your trained checkpoint and support set for single-image or batch predictions.
- Predictions are automatically rescaled to the original query resolution, so annotations align with high-resolution satellite imagery.

---

Keep this file updated whenever training steps or architecture components change so that it remains the canonical documentation for the project.