"""
Utility modules for detection app.
"""

from .nms import (
    calculate_iou,
    calculate_diou,
    standard_nms,
    soft_nms,
    diou_nms,
    apply_nms,
    calculate_occlusion_scores
)

from .inference import (
    YOLODetector,
    get_detector,
    DRIVING_CLASSES,
    COCO_CLASSES
)

from .metrics import (
    calculate_precision_recall,
    calculate_map,
    calculate_detection_stats,
    compare_nms_methods
)

__all__ = [
    # NMS functions
    'calculate_iou',
    'calculate_diou',
    'standard_nms',
    'soft_nms',
    'diou_nms',
    'apply_nms',
    'calculate_occlusion_scores',
    
    # Inference
    'YOLODetector',
    'get_detector',
    'DRIVING_CLASSES',
    'COCO_CLASSES',
    
    # Metrics
    'calculate_precision_recall',
    'calculate_map',
    'calculate_detection_stats',
    'compare_nms_methods',
]
