from django.contrib import admin
from .models import UploadedMedia, DetectionSession, DetectionResult, PerformanceMetrics, NMSComparison


@admin.register(UploadedMedia)
class UploadedMediaAdmin(admin.ModelAdmin):
    list_display = ['id', 'original_filename', 'media_type', 'uploaded_at', 'file_size']
    list_filter = ['media_type', 'uploaded_at']
    search_fields = ['original_filename']


@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'media', 'nms_method', 'status', 'processing_time', 'fps', 'created_at']
    list_filter = ['nms_method', 'status', 'created_at']
    search_fields = ['media__original_filename']


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'class_name', 'confidence', 'is_occluded', 'frame_number']
    list_filter = ['class_name', 'is_occluded']
    search_fields = ['class_name']


@admin.register(PerformanceMetrics)
class PerformanceMetricsAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'total_objects', 'avg_confidence', 'precision', 'recall', 'f1_score']
    list_filter = ['session__nms_method']


@admin.register(NMSComparison)
class NMSComparisonAdmin(admin.ModelAdmin):
    list_display = ['id', 'media', 'created_at']
    list_filter = ['created_at']
