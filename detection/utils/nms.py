"""
Custom NMS (Non-Maximum Suppression) implementations for occlusion handling.

This module provides:
- Standard NMS: Traditional hard suppression
- Soft-NMS: Gaussian decay instead of hard removal
- DIoU-NMS: Distance-IoU based suppression for better occlusion handling
"""

import numpy as np
import torch
from typing import Tuple, List, Optional


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_diou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Distance-IoU (DIoU) between two boxes.
    DIoU considers both overlap and center point distance.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        DIoU value (can be negative if boxes are far apart)
    """
    iou = calculate_iou(box1, box2)
    
    # Calculate center points
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    # Euclidean distance between centers
    center_distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    # Enclosing box (smallest box containing both boxes)
    enclose_x1 = min(box1[0], box2[0])
    enclose_y1 = min(box1[1], box2[1])
    enclose_x2 = max(box1[2], box2[2])
    enclose_y2 = max(box1[3], box2[3])
    
    # Diagonal length of enclosing box
    diagonal = np.sqrt((enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2)
    
    if diagonal == 0:
        return iou
    
    # DIoU = IoU - (center_distance^2 / diagonal^2)
    diou = iou - (center_distance**2 / diagonal**2)
    
    return diou


def standard_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Non-Maximum Suppression.
    
    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Tuple of (kept_boxes, kept_scores, kept_indices)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Keep the highest scoring box
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        
        ious = np.array([calculate_iou(current_box, boxes[idx]) for idx in remaining_indices])
        
        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    keep_indices = np.array(keep_indices)
    return boxes[keep_indices], scores[keep_indices], keep_indices


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    method: str = 'gaussian'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression.
    Instead of hard removal, reduces scores based on IoU overlap.
    
    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold (used for linear method)
        sigma: Gaussian decay parameter (lower = more aggressive)
        score_threshold: Minimum score threshold after decay
        method: 'gaussian' or 'linear'
    
    Returns:
        Tuple of (kept_boxes, kept_scores, kept_indices)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Make copies to avoid modifying originals
    boxes = boxes.copy()
    scores = scores.copy()
    
    N = len(boxes)
    indices = np.arange(N)
    
    # Sort by scores initially
    sorted_order = np.argsort(scores)[::-1]
    boxes = boxes[sorted_order]
    scores = scores[sorted_order]
    indices = indices[sorted_order]
    
    keep_indices = []
    
    for i in range(N):
        if scores[i] < score_threshold:
            continue
        
        keep_indices.append(i)
        
        # Decay scores of remaining boxes based on IoU
        for j in range(i + 1, N):
            if scores[j] < score_threshold:
                continue
            
            iou = calculate_iou(boxes[i], boxes[j])
            
            if method == 'gaussian':
                # Gaussian decay: score = score * exp(-(iou^2) / sigma)
                weight = np.exp(-(iou ** 2) / sigma)
            else:
                # Linear decay
                if iou > iou_threshold:
                    weight = 1 - iou
                else:
                    weight = 1.0
            
            scores[j] *= weight
    
    keep_indices = np.array(keep_indices)
    final_mask = scores[keep_indices] >= score_threshold
    keep_indices = keep_indices[final_mask]
    
    return boxes[keep_indices], scores[keep_indices], indices[keep_indices]


def diou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance-IoU Non-Maximum Suppression.
    Uses DIoU instead of IoU for better handling of overlapping objects.
    
    DIoU considers both overlap and center distance, making it more robust
    for occlusion scenarios where objects may overlap but have distant centers.
    
    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format
        scores: Array of shape (N,) with confidence scores
        iou_threshold: DIoU threshold for suppression
    
    Returns:
        Tuple of (kept_boxes, kept_scores, kept_indices)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Keep the highest scoring box
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate DIoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        
        dious = np.array([calculate_diou(current_box, boxes[idx]) for idx in remaining_indices])
        
        # Keep boxes with DIoU below threshold
        # DIoU can be negative, but typically positive for overlapping boxes
        mask = dious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    keep_indices = np.array(keep_indices)
    return boxes[keep_indices], scores[keep_indices], keep_indices


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    method: str = 'standard',
    iou_threshold: float = 0.45,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the specified NMS method.
    
    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format
        scores: Array of shape (N,) with confidence scores
        method: 'standard', 'soft', or 'diou'
        iou_threshold: IoU/DIoU threshold
        sigma: Sigma parameter for soft-NMS
    
    Returns:
        Tuple of (kept_boxes, kept_scores, kept_indices)
    """
    if method == 'soft':
        return soft_nms(boxes, scores, iou_threshold, sigma)
    elif method == 'diou':
        return diou_nms(boxes, scores, iou_threshold)
    else:
        return standard_nms(boxes, scores, iou_threshold)


def calculate_occlusion_scores(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate occlusion scores for each detection.
    
    Occlusion score is based on:
    - Number of overlapping boxes
    - Maximum IoU with other boxes
    - Confidence relative to overlapping boxes
    
    Args:
        boxes: Array of shape (N, 4)
        scores: Array of shape (N,)
        iou_threshold: Minimum IoU to consider as overlapping
    
    Returns:
        Tuple of (occlusion_scores, overlapping_counts)
    """
    N = len(boxes)
    occlusion_scores = np.zeros(N)
    overlapping_counts = np.zeros(N, dtype=int)
    
    for i in range(N):
        max_iou = 0
        overlap_count = 0
        
        for j in range(N):
            if i == j:
                continue
            
            iou = calculate_iou(boxes[i], boxes[j])
            
            if iou > iou_threshold:
                overlap_count += 1
                max_iou = max(max_iou, iou)
        
        overlapping_counts[i] = overlap_count
        
        # Occlusion score combines overlap ratio and number of overlaps
        if overlap_count > 0:
            # Normalize by max possible overlaps (assume max 5)
            overlap_factor = min(overlap_count / 5.0, 1.0)
            occlusion_scores[i] = 0.5 * max_iou + 0.5 * overlap_factor
        else:
            occlusion_scores[i] = 0.0
    
    return occlusion_scores, overlapping_counts


# PyTorch versions for GPU acceleration (if available)
def torch_calculate_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes using PyTorch (GPU accelerated).
    
    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)
    
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    union = area1[:, None] + area2 - inter_area
    
    return inter_area / torch.clamp(union, min=1e-6)
