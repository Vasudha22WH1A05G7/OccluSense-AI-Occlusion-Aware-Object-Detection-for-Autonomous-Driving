"""
YOLO Detector wrapper for inference with custom NMS methods.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
from ultralytics import YOLO

from .nms import apply_nms, calculate_occlusion_scores
from .driving_logic import check_environment, get_driving_suggestions, detect_traffic_light_color


# COCO class names relevant to autonomous driving
DRIVING_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign',
}

# All COCO class names for reference
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}


class YOLODetector:
    """
    YOLO Detector wrapper with custom NMS support for occlusion handling.
    """
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        device: Optional[str] = None,
        target_classes: Optional[List[int]] = None
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights (auto-downloads if not found)
            device: 'cuda', 'cpu', or None for auto-detect
            target_classes: List of class IDs to detect (None for all classes)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_classes = target_classes or list(DRIVING_CLASSES.keys())
        
        # Load model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def preprocess_image(
        self,
        image: np.ndarray,
        enhance_low_light: bool = False,
        reduce_motion_blur: bool = False
    ) -> np.ndarray:
        """
        Preprocess image for better detection in challenging conditions.
        
        Args:
            image: Input BGR image
            enhance_low_light: Apply low-light enhancement
            reduce_motion_blur: Apply motion blur reduction
        
        Returns:
            Preprocessed image
        """
        if enhance_low_light:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        if reduce_motion_blur:
            # Apply sharpening filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)
        
        return image
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_method: str = 'standard',
        sigma: float = 0.5,
        enhance_low_light: bool = False,
        agnostic_nms: bool = True
    ) -> Dict:
        """
        Run detection on an image.
        
        Args:
            image: BGR image (numpy array)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            nms_method: 'standard', 'soft', or 'diou'
            sigma: Sigma for soft-NMS
            enhance_low_light: Apply low-light preprocessing
            agnostic_nms: Apply class-agnostic NMS (all classes together)
        
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Preprocess
        processed_image = self.preprocess_image(image, enhance_low_light=enhance_low_light)
        
        # Run YOLO inference (with standard NMS disabled for custom NMS)
        inference_start = time.time()
        
        # Use high IoU to get all candidates, then apply custom NMS
        results = self.model(
            processed_image,
            conf=confidence_threshold * 0.5,  # Lower to get more candidates
            iou=0.9 if nms_method != 'standard' else iou_threshold,
            classes=self.target_classes,
            verbose=False
        )
        
        inference_time = (time.time() - inference_start) * 1000  # ms
        
        # Extract raw detections
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter by confidence
                mask = scores >= confidence_threshold
                all_boxes.extend(boxes[mask])
                all_scores.extend(scores[mask])
                all_classes.extend(classes[mask])
        
        all_boxes = np.array(all_boxes) if all_boxes else np.array([]).reshape(0, 4)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)
        
        # Apply custom NMS
        nms_start = time.time()
        
        if len(all_boxes) > 0:
            if agnostic_nms:
                # Apply NMS across all classes
                kept_boxes, kept_scores, kept_indices = apply_nms(
                    all_boxes, all_scores, nms_method, iou_threshold, sigma
                )
                kept_classes = all_classes[kept_indices]
            else:
                # Apply NMS per class
                kept_boxes = []
                kept_scores = []
                kept_classes = []
                
                for class_id in np.unique(all_classes):
                    class_mask = all_classes == class_id
                    class_boxes = all_boxes[class_mask]
                    class_scores = all_scores[class_mask]
                    
                    boxes_out, scores_out, _ = apply_nms(
                        class_boxes, class_scores, nms_method, iou_threshold, sigma
                    )
                    
                    kept_boxes.extend(boxes_out)
                    kept_scores.extend(scores_out)
                    kept_classes.extend([class_id] * len(boxes_out))
                
                kept_boxes = np.array(kept_boxes) if kept_boxes else np.array([]).reshape(0, 4)
                kept_scores = np.array(kept_scores)
                kept_classes = np.array(kept_classes)
        else:
            kept_boxes = np.array([]).reshape(0, 4)
            kept_scores = np.array([])
            kept_classes = np.array([])
        
        nms_time = (time.time() - nms_start) * 1000  # ms
        
        # Calculate occlusion scores
        if len(kept_boxes) > 0:
            occlusion_scores, overlapping_counts = calculate_occlusion_scores(
                kept_boxes, kept_scores
            )
        else:
            occlusion_scores = np.array([])
            overlapping_counts = np.array([])
        
        total_time = (time.time() - start_time) * 1000  # ms
        
        # Build detections list
        detections = []
        for i in range(len(kept_boxes)):
            class_id = int(kept_classes[i])
            detection = {
                'class_id': class_id,
                'class_name': COCO_CLASSES.get(class_id, f'class_{class_id}'),
                'confidence': float(kept_scores[i]),
                'bbox': kept_boxes[i].tolist(),
                'is_occluded': bool(occlusion_scores[i] > 0.3),
                'occlusion_score': float(occlusion_scores[i]),
                'overlapping_objects': int(overlapping_counts[i])
            }
            
            # Detect traffic light color
            if detection['class_name'] == 'traffic light':
                detection['color'] = detect_traffic_light_color(image, detection['bbox'])
                
            detections.append(detection)
        
        # Environment and Suggestions (for single image)
        env_warning = check_environment(image)
        suggestions = get_driving_suggestions(detections, image.shape)

        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': image.shape[:2],
            'nms_method': nms_method,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'inference_time_ms': inference_time,
            'nms_time_ms': nms_time,
            'total_time_ms': total_time,
            'fps': 1000 / total_time if total_time > 0 else 0,
            'environment_warning': env_warning,
            'driving_suggestions': suggestions
        }
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms_method: str = 'standard',
        sigma: float = 0.5,
        max_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Run detection on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            confidence_threshold: Minimum confidence
            iou_threshold: IoU threshold
            nms_method: NMS method
            sigma: Soft-NMS sigma
            max_frames: Maximum frames to process (None for all)
            progress_callback: Callback function(current_frame, total_frames)
        
        Returns:
            Dictionary with aggregated results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Setup video writer with browser-compatible codec
        writer = None
        temp_output_path = None
        if output_path:
            # Try H.264 codec first (browser compatible)
            # Use avc1/H264 for better browser compatibility
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise Exception("avc1 codec not available")
            except:
                # Fallback to mp4v and convert later
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                temp_output_path = output_path.replace('.mp4', '_temp.mp4')
                writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        all_detections = []
        unique_suggestions = set()
        frame_count = 0
        total_time = 0
        
        env_warning = None
        
        while cap.isOpened() and (not max_frames or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            result = self.detect(
                frame,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                nms_method=nms_method,
                sigma=sigma
            )
            
            # Check environment on first frame
            if frame_count == 0:
                env_warning = check_environment(frame)
            
            # Get driving suggestions
            suggestions = get_driving_suggestions(result['detections'], frame.shape)
            result['suggestions'] = suggestions
            result['env_warning'] = env_warning
            
            # Add frame number to detections
            for det in result['detections']:
                det['frame_number'] = frame_count
            
            all_detections.extend(result['detections'])
            total_time += result['total_time_ms']
            
            # Draw annotations if saving video
            if writer:
                annotated_frame = self.draw_detections(
                    frame, 
                    result['detections'],
                    env_warning=env_warning,
                    suggestions=suggestions
                )
                writer.write(annotated_frame)
            
            if suggestions:
                unique_suggestions.update(suggestions)
            
            frame_count += 1
            
            if progress_callback:
                progress_callback(frame_count, total_frames)
        
        cap.release()
        if writer:
            writer.release()
        
        # If we used temp file, try to convert to browser-compatible format
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                import subprocess
                # Try ffmpeg conversion
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', temp_output_path,
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-movflags', '+faststart',
                    output_path
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
                if os.path.exists(output_path):
                    os.remove(temp_output_path)
                else:
                    # ffmpeg failed, use the temp file
                    os.rename(temp_output_path, output_path)
            except Exception:
                # ffmpeg not available, just rename temp file
                os.rename(temp_output_path, output_path)
        
        # Aggregate results
        avg_time = total_time / frame_count if frame_count > 0 else 0
        
        return {
            'detections': all_detections,
            'total_detections': len(all_detections),
            'total_frames': frame_count,
            'video_fps': fps,
            'detection_fps': 1000 / avg_time if avg_time > 0 else 0,
            'avg_frame_time_ms': avg_time,
            'nms_method': nms_method,
            'output_path': output_path,
            'environment_warning': env_warning,
            'driving_suggestions': list(unique_suggestions)
        }
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_occlusion: bool = True,
        env_warning: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: BGR image
            detections: List of detection dictionaries
            show_confidence: Show confidence scores
            show_occlusion: Show occlusion indicators
        
        Returns:
            Annotated image
        """
        image = image.copy()
        
        # Color scheme based on confidence and occlusion
        def get_color(confidence, is_occluded):
            if is_occluded:
                return (0, 165, 255)  # Orange for occluded
            elif confidence >= 0.7:
                return (0, 255, 0)  # Green for high confidence
            elif confidence >= 0.5:
                return (0, 255, 255)  # Yellow for medium
            else:
                return (0, 0, 255)  # Red for low
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det['confidence']
            class_name = det['class_name']
            is_occluded = det.get('is_occluded', False)
            
            color = get_color(confidence, is_occluded)
            
            # Draw bounding box
            thickness = 3 if is_occluded else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label = class_name
            if show_confidence:
                label += f' {confidence:.2f}'
            if show_occlusion and is_occluded:
                label += ' [OCC]'
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                image,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Draw Environment Warning
        if env_warning:
            cv2.putText(image, env_warning, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 0, 255), 3)
        
        # Draw Suggestions
        if suggestions:
            y_pos = 100
            for suggestion in suggestions:
                color = (0, 0, 255) if "STOP" in suggestion or "CAUTION" in suggestion else (0, 255, 0)
                cv2.putText(image, suggestion, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, color, 3)
                y_pos += 40
        
        return image


# Singleton detector instance
_detector_instance = None


def get_detector(
    model_path: str = 'yolov8n.pt',
    target_classes: Optional[List[int]] = None
) -> YOLODetector:
    """
    Get or create a singleton detector instance.
    
    Args:
        model_path: Path to YOLO weights
        target_classes: Target class IDs
    
    Returns:
        YOLODetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = YOLODetector(model_path, target_classes=target_classes)
    
    return _detector_instance
