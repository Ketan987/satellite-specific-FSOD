"""
Simple detection metrics: per-episode AP/mAP at IoU=0.5

This is a lightweight implementation suitable for episodic FSOD evaluation.
"""
import torch
import numpy as np


def iou_xywh(box1, box2):
    # boxes: [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_ap(gt_boxes, gt_labels, detections, iou_thr=0.5, n_gt=None):
    """
    Compute AP for a single class over an episode.

    gt_boxes: list of boxes (list of tensors) per image
    gt_labels: list of label tensors per image
    detections: list of dicts {'boxes': [x,y,w,h], 'scores':float, 'class':int, 'image_id': int}
    """
    if n_gt is None:
        # count total GT for this class
        n_gt = 0
        for b in gt_boxes:
            n_gt += len(b)

    if n_gt == 0:
        return 0.0

    # Sort detections by score desc
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    # For matching, keep track of matched GTs per image
    matched = {}

    for i, det in enumerate(detections):
        img_id = det['image_id']
        box = det['bbox']
        # Get GTs for this image
        gt_b = gt_boxes[img_id]
        if len(gt_b) == 0:
            fp[i] = 1
            continue

        # Compute IoUs
        ious = [iou_xywh(box, gt.cpu().numpy().tolist()) for gt in gt_b]
        best_iou = 0.0
        best_idx = -1
        for idx, val in enumerate(ious):
            if val > best_iou:
                best_iou = val
                best_idx = idx

        if best_iou >= iou_thr:
            # check if this GT already matched
            already = matched.get((img_id, best_idx), False)
            if not already:
                tp[i] = 1
                matched[(img_id, best_idx)] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recalls = tp_cum / float(n_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # AP as area under PR curve (Trapezoidal)
    # Append sentinels
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    inds = np.where(recalls[1:] != recalls[:-1])[0]
    ap = 0.0
    for i in inds:
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap


def map_per_episode(predictions, query_boxes, query_labels, n_way, iou_thr=0.5):
    """
    Compute mAP for a single episode.

    predictions: list of prediction dicts from model (per query image) each with keys 'boxes','scores','pred_classes'
    query_boxes: list of tensors per query image
    query_labels: list of tensors per query image (labels 0..n_way-1)
    n_way: number of classes in episode
    """
    # Collect GT per image per class
    num_images = len(query_boxes)
    gt_boxes_per_class = {c: [] for c in range(n_way)}
    n_gt_per_class = {c: 0 for c in range(n_way)}

    for img_id in range(num_images):
        boxes = query_boxes[img_id]
        labels = query_labels[img_id]
        for b, l in zip(boxes, labels):
            cls = int(l.item())
            gt_boxes_per_class[cls].append((img_id, b))
            n_gt_per_class[cls] += 1

    # Build detections per class
    dets_per_class = {c: [] for c in range(n_way)}

    for img_id, pred in enumerate(predictions):
        boxes = pred['boxes']
        scores = pred['scores']
        pred_classes = pred.get('pred_classes', None)
        if pred_classes is None:
            continue
        for i in range(len(boxes)):
            cls = int(pred_classes[i].item())
            dets_per_class[cls].append({'image_id': img_id, 'bbox': boxes[i].cpu().numpy().tolist(), 'score': float(scores[i].item())})

    aps = []
    # Prepare GT boxes per image per class in lists matching compute_ap signature
    for c in range(n_way):
        # Build gt_boxes list indexed by image id
        gt_boxes_by_image = [[] for _ in range(num_images)]
        for (img_id, b) in [x for x in gt_boxes_per_class[c]]:
            gt_boxes_by_image[img_id].append(b)

        # Format detections for compute_ap
        dets = []
        for d in dets_per_class[c]:
            dets.append({'image_id': d['image_id'], 'bbox': d['bbox'], 'score': d['score']})

        ap = compute_ap(gt_boxes_by_image, None, dets, iou_thr=iou_thr, n_gt=n_gt_per_class[c])
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))
