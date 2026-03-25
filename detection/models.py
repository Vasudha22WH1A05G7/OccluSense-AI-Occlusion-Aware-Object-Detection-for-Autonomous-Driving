"""
Database models for detection results and performance metrics.
"""

from django.db import models
from django.utils import timezone
import json


class UploadedMedia(models.Model):
    """Model for storing uploaded images and videos."""
    
    MEDIA_TYPE_CHOICES = [
        ('image', 'Image'),
        ('video', 'Video'),
    ]
    
    file = models.FileField(upload_to='uploads/')
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.PositiveIntegerField(default=0)  # in bytes
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Uploaded Media'
        verbose_name_plural = 'Uploaded Media'
    
    def __str__(self):
        return f"{self.original_filename} ({self.media_type})"


class DetectionSession(models.Model):
    """Model for storing a detection session with settings."""
    
    NMS_METHOD_CHOICES = [
        ('standard', 'Standard NMS'),
        ('soft', 'Soft-NMS'),
        ('diou', 'DIoU-NMS'),
    ]
    
    media = models.ForeignKey(UploadedMedia, on_delete=models.CASCADE, related_name='sessions')
    nms_method = models.CharField(max_length=10, choices=NMS_METHOD_CHOICES, default='standard')
    confidence_threshold = models.FloatField(default=0.25)
    iou_threshold = models.FloatField(default=0.45)
    soft_nms_sigma = models.FloatField(default=0.5)
    
    # Results
    result_image = models.ImageField(upload_to='results/', null=True, blank=True)
    result_video = models.FileField(upload_to='results/', null=True, blank=True)
    processing_time = models.FloatField(default=0.0)  # in seconds
    fps = models.FloatField(default=0.0)
    
    # Status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True, null=True)
    
    # Driving Analysis
    environment_warning = models.CharField(max_length=255, null=True, blank=True)
    driving_suggestions = models.TextField(null=True, blank=True)  # JSON-encoded list or text summary

    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Detection Session'
        verbose_name_plural = 'Detection Sessions'
    
    def __str__(self):
        return f"Session {self.id} - {self.nms_method} ({self.status})"
    
    def mark_completed(self):
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()
    
    def mark_failed(self, error_message):
        self.status = 'failed'
        self.error_message = error_message
        self.completed_at = timezone.now()
        self.save()


class DetectionResult(models.Model):
    """Model for storing individual detection results."""
    
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, related_name='detections')
    frame_number = models.PositiveIntegerField(default=0)  # For videos
    
    # Object detection data
    class_id = models.PositiveIntegerField()
    class_name = models.CharField(max_length=50)
    confidence = models.FloatField()
    
    # Bounding box (normalized coordinates 0-1)
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    
    # Occlusion analysis
    is_occluded = models.BooleanField(default=False)
    occlusion_score = models.FloatField(default=0.0)  # 0-1, higher = more occluded
    overlapping_objects = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-confidence']
        verbose_name = 'Detection Result'
        verbose_name_plural = 'Detection Results'
    
    def __str__(self):
        return f"{self.class_name} ({self.confidence:.2f}) - Frame {self.frame_number}"
    
    @property
    def bbox(self):
        return [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]
    
    def to_dict(self):
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'is_occluded': self.is_occluded,
            'occlusion_score': self.occlusion_score,
            'frame_number': self.frame_number,
        }


class PerformanceMetrics(models.Model):
    """Model for storing performance metrics per session."""
    
    session = models.OneToOneField(DetectionSession, on_delete=models.CASCADE, related_name='metrics')
    
    # Detection metrics
    total_objects = models.PositiveIntegerField(default=0)
    avg_confidence = models.FloatField(default=0.0)
    
    # Class distribution (stored as JSON)
    class_distribution = models.JSONField(default=dict)
    
    # Occlusion metrics
    occluded_objects = models.PositiveIntegerField(default=0)
    avg_occlusion_score = models.FloatField(default=0.0)
    
    # Performance
    inference_time = models.FloatField(default=0.0)  # ms per frame
    nms_time = models.FloatField(default=0.0)  # ms per frame
    total_frames = models.PositiveIntegerField(default=1)
    
    # Computed metrics (for comparison)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    mAP = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Performance Metrics'
        verbose_name_plural = 'Performance Metrics'
    
    def __str__(self):
        return f"Metrics for Session {self.session_id}"
    
    def to_dict(self):
        return {
            'total_objects': self.total_objects,
            'avg_confidence': self.avg_confidence,
            'class_distribution': self.class_distribution,
            'occluded_objects': self.occluded_objects,
            'avg_occlusion_score': self.avg_occlusion_score,
            'inference_time': self.inference_time,
            'nms_time': self.nms_time,
            'total_frames': self.total_frames,
            'fps': self.session.fps if self.session else 0,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mAP': self.mAP,
        }


class NMSComparison(models.Model):
    """Model for storing NMS method comparison results."""
    
    media = models.ForeignKey(UploadedMedia, on_delete=models.CASCADE, related_name='comparisons')
    
    # Sessions for each NMS method
    standard_session = models.ForeignKey(
        DetectionSession, on_delete=models.SET_NULL, null=True, 
        related_name='standard_comparisons'
    )
    soft_session = models.ForeignKey(
        DetectionSession, on_delete=models.SET_NULL, null=True,
        related_name='soft_comparisons'
    )
    diou_session = models.ForeignKey(
        DetectionSession, on_delete=models.SET_NULL, null=True,
        related_name='diou_comparisons'
    )
    
    # Configuration used
    confidence_threshold = models.FloatField(default=0.25)
    iou_threshold = models.FloatField(default=0.45)
    soft_nms_sigma = models.FloatField(default=0.5)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'NMS Comparison'
        verbose_name_plural = 'NMS Comparisons'
    
    def __str__(self):
        return f"Comparison {self.id} - {self.media.original_filename}"
