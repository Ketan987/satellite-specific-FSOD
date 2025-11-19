import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.similarity import non_max_suppression


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([x, y, w, h], dim=1)


def clamp_boxes_xyxy(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    x1 = boxes[:, 0].clamp(0.0, float(image_size - 1))
    y1 = boxes[:, 1].clamp(0.0, float(image_size - 1))
    x2 = boxes[:, 2].clamp(0.0, float(image_size))
    y2 = boxes[:, 3].clamp(0.0, float(image_size))
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)

    x11 = boxes1[:, 0].unsqueeze(1)
    y11 = boxes1[:, 1].unsqueeze(1)
    x12 = boxes1[:, 2].unsqueeze(1)
    y12 = boxes1[:, 3].unsqueeze(1)

    x21 = boxes2[:, 0].unsqueeze(0)
    y21 = boxes2[:, 1].unsqueeze(0)
    x22 = boxes2[:, 2].unsqueeze(0)
    y22 = boxes2[:, 3].unsqueeze(0)

    inter_x1 = torch.max(x11, x21)
    inter_y1 = torch.max(y11, y21)
    inter_x2 = torch.min(x12, x22)
    inter_y2 = torch.min(y12, y22)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1 + area2 - inter_area
    union = torch.clamp(union, min=1e-6)
    return inter_area / union


class AnchorGenerator:
    def __init__(self, scales: List[float], ratios: List[float]):
        self.scales = scales
        self.ratios = ratios

    def base_anchors(self, device: torch.device) -> torch.Tensor:
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                w = scale * math.sqrt(ratio)
                h = scale / math.sqrt(ratio)
                anchors.append([0.0, 0.0, w, h])
        return torch.tensor(anchors, device=device, dtype=torch.float32)

    def grid_anchors(self, feat_h: int, feat_w: int, stride: float, device: torch.device) -> torch.Tensor:
        base = self.base_anchors(device)
        num_anchors = base.size(0)

        shifts_x = torch.arange(0, feat_w, device=device, dtype=torch.float32) * stride + stride / 2.0
        shifts_y = torch.arange(0, feat_h, device=device, dtype=torch.float32) * stride + stride / 2.0
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        centers = torch.stack((shift_x, shift_y), dim=-1).view(-1, 2)

        centers = centers.unsqueeze(1).repeat(1, num_anchors, 1)
        base = base.unsqueeze(0).repeat(centers.size(0), 1, 1)

        anchors = torch.zeros_like(base)
        anchors[..., 0:2] = centers
        anchors[..., 2:] = base[..., 2:]
        return anchors.view(-1, 4)


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        anchor_scales: Optional[List[float]] = None,
        anchor_ratios: Optional[List[float]] = None,
        pre_nms_topk: int = 1000,
        post_nms_topk: int = 300,
        nms_threshold: float = 0.7,
        min_box_size: float = 4.0,
    ):
        super().__init__()
        anchor_scales = anchor_scales or [32, 64, 128, 256]
        anchor_ratios = anchor_ratios or [0.5, 1.0, 2.0]

        self.anchor_generator = AnchorGenerator(anchor_scales, anchor_ratios)
        num_anchors = len(anchor_scales) * len(anchor_ratios)

        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.objectness_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

        for layer in [self.conv, self.objectness_logits, self.bbox_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_threshold = nms_threshold
        self.min_box_size = min_box_size
        self.pos_iou_threshold = 0.7
        self.neg_iou_threshold = 0.3
        self.batch_size_per_image = 256
        self.positive_fraction = 0.5

    def forward(
        self,
        features: torch.Tensor,
        image_size: int,
        gt_boxes: Optional[List[torch.Tensor]] = None,
    ):
        device = features.device
        B, _, H, W = features.shape
        stride = float(image_size) / float(H)
        anchors = self.anchor_generator.grid_anchors(H, W, stride, device)

        t = F.relu(self.conv(features))
        objectness = self.objectness_logits(t)
        bbox_deltas = self.bbox_deltas(t)

        objectness = objectness.permute(0, 2, 3, 1).reshape(B, -1)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(B, -1, 4)

        proposals = []
        rpn_losses = []

        for b in range(B):
            objectness_scores = torch.sigmoid(objectness[b])
            box_deltas = bbox_deltas[b]

            decoded = self.decode_boxes(anchors, box_deltas, image_size)
            filtered_boxes, filtered_scores = self.select_top_proposals(decoded, objectness_scores)
            proposals.append(filtered_boxes)

            if gt_boxes is not None and len(gt_boxes) > b:
                gt = gt_boxes[b]
                loss = self.compute_loss(objectness[b], box_deltas, anchors, gt, image_size)
                rpn_losses.append(loss)

        loss_tensor = None
        if rpn_losses:
            loss_tensor = torch.stack(rpn_losses).mean()

        return {
            'proposals': proposals,
            'losses': rpn_losses if rpn_losses else None,
            'loss': loss_tensor
        }

    def decode_boxes(self, anchors: torch.Tensor, deltas: torch.Tensor, image_size: int) -> torch.Tensor:
        ctr_x = anchors[:, 0]
        ctr_y = anchors[:, 1]
        widths = anchors[:, 2]
        heights = anchors[:, 3]

        dx = deltas[:, 0].clamp(min=-1.0, max=1.0)
        dy = deltas[:, 1].clamp(min=-1.0, max=1.0)
        dw = deltas[:, 2].clamp(min=-1.0, max=1.0)
        dh = deltas[:, 3].clamp(min=-1.0, max=1.0)

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)

        pred_w = pred_w.clamp(min=self.min_box_size, max=float(image_size))
        pred_h = pred_h.clamp(min=self.min_box_size, max=float(image_size))

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        boxes_xyxy = clamp_boxes_xyxy(boxes_xyxy, image_size)
        boxes_xywh = xyxy_to_xywh(boxes_xyxy)
        return boxes_xywh

    def select_top_proposals(self, boxes: torch.Tensor, scores: torch.Tensor):
        num_proposals = min(self.pre_nms_topk, boxes.size(0))
        scores, topk_idx = scores.topk(num_proposals)
        boxes = boxes[topk_idx]

        keep = non_max_suppression(boxes, scores, self.nms_threshold)
        keep = keep[: self.post_nms_topk]
        return boxes[keep], scores[keep]

    def compute_loss(
        self,
        objectness_logits: torch.Tensor,
        bbox_deltas: torch.Tensor,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        if gt_boxes.numel() == 0:
            labels = torch.zeros_like(objectness_logits)
            return F.binary_cross_entropy_with_logits(objectness_logits, labels, reduction='mean')

        gt_xyxy = xywh_to_xyxy(gt_boxes)
        anchors_xyxy = xywh_to_xyxy(self.cxcywh_to_xywh(anchors))

        labels, bbox_targets = self.assign_targets(anchors_xyxy, gt_xyxy)

        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=objectness_logits.device)

        cls_loss = F.binary_cross_entropy_with_logits(
            objectness_logits[valid_mask], labels[valid_mask].float(), reduction='mean'
        )

        pos_mask = labels == 1
        if pos_mask.sum() == 0:
            return cls_loss

        bbox_loss = F.smooth_l1_loss(
            bbox_deltas[pos_mask], bbox_targets[pos_mask], reduction='mean'
        )
        return cls_loss + bbox_loss

    def assign_targets(self, anchors_xyxy: torch.Tensor, gt_xyxy: torch.Tensor):
        ious = compute_iou_xyxy(anchors_xyxy, gt_xyxy)
        max_iou, gt_indices = ious.max(dim=1)

        labels = torch.full_like(max_iou, -1, dtype=torch.long)
        labels[max_iou < self.neg_iou_threshold] = 0
        labels[max_iou >= self.pos_iou_threshold] = 1

        # Ensure every GT box has at least one positive anchor
        if gt_xyxy.numel() > 0:
            max_iou_per_gt, anchor_idx = ious.max(dim=0)
            labels[anchor_idx[max_iou_per_gt > 0]] = 1

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        pos_indices = torch.where(labels == 1)[0]
        if pos_indices.numel() > num_pos:
            disable = pos_indices[torch.randperm(pos_indices.numel(), device=anchors_xyxy.device)[: pos_indices.numel() - num_pos]]
            labels[disable] = -1

        num_neg = self.batch_size_per_image - torch.sum(labels == 1).item()
        neg_indices = torch.where(labels == 0)[0]
        if neg_indices.numel() > num_neg:
            disable = neg_indices[torch.randperm(neg_indices.numel(), device=anchors_xyxy.device)[: neg_indices.numel() - num_neg]]
            labels[disable] = -1

        bbox_targets = torch.zeros((anchors_xyxy.size(0), 4), device=anchors_xyxy.device)
        pos_mask = labels == 1
        if pos_mask.any():
            selected_anchors = self.xyxy_to_cxcywh(anchors_xyxy[pos_mask])
            selected_gt = self.xyxy_to_cxcywh(gt_xyxy[gt_indices[pos_mask]])
            bbox_targets[pos_mask] = self.encode_boxes(selected_anchors, selected_gt)

        return labels, bbox_targets

    @staticmethod
    def encode_boxes(anchors: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros_like(gt)
        targets[:, 0] = (gt[:, 0] - anchors[:, 0]) / anchors[:, 2]
        targets[:, 1] = (gt[:, 1] - anchors[:, 1]) / anchors[:, 3]
        targets[:, 2] = torch.log(gt[:, 2] / anchors[:, 2])
        targets[:, 3] = torch.log(gt[:, 3] / anchors[:, 3])
        return targets.clamp(min=-5.0, max=5.0)

    @staticmethod
    def cxcywh_to_xywh(anchors: torch.Tensor) -> torch.Tensor:
        x = anchors[:, 0] - anchors[:, 2] / 2.0
        y = anchors[:, 1] - anchors[:, 3] / 2.0
        return torch.stack([x, y, anchors[:, 2], anchors[:, 3]], dim=1)

    @staticmethod
    def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        return torch.stack([cx, cy, w, h], dim=1)
