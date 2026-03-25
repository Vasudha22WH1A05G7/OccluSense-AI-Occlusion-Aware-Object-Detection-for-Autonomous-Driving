"""
Views for browsing and filtering detection results.
"""

from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.db.models import Count, Avg

from detection.models import DetectionSession, DetectionResult, PerformanceMetrics


class ResultsBrowserView(ListView):
    """Browse all detection results with filtering."""
    model = DetectionSession
    template_name = 'results/browser.html'
    context_object_name = 'sessions'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = DetectionSession.objects.filter(status='completed')
        
        # Filter by NMS method
        nms_method = self.request.GET.get('nms_method')
        if nms_method and nms_method in ['standard', 'soft', 'diou']:
            queryset = queryset.filter(nms_method=nms_method)
        
        # Filter by media type
        media_type = self.request.GET.get('media_type')
        if media_type and media_type in ['image', 'video']:
            queryset = queryset.filter(media__media_type=media_type)
        
        # Sort
        sort = self.request.GET.get('sort', '-created_at')
        if sort in ['created_at', '-created_at', 'fps', '-fps', 'processing_time', '-processing_time']:
            queryset = queryset.order_by(sort)
        else:
            queryset = queryset.order_by('-created_at')
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['current_nms'] = self.request.GET.get('nms_method', '')
        context['current_media'] = self.request.GET.get('media_type', '')
        context['current_sort'] = self.request.GET.get('sort', '-created_at')
        return context


class ResultDetailView(DetailView):
    """Detailed view of a single detection session."""
    model = DetectionSession
    template_name = 'results/detail.html'
    context_object_name = 'session'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        session = self.object
        
        # Get all detections
        detections = session.detections.all()
        context['detections'] = detections
        
        # Get metrics
        try:
            context['metrics'] = session.metrics
        except PerformanceMetrics.DoesNotExist:
            context['metrics'] = None
        
        # Get class distribution
        class_dist = detections.values('class_name').annotate(
            count=Count('id')
        ).order_by('-count')
        context['class_distribution'] = list(class_dist)
        
        # Occlusion stats
        occluded = detections.filter(is_occluded=True)
        context['occluded_count'] = occluded.count()
        context['total_count'] = detections.count()
        
        return context


class StatisticsView(ListView):
    """Overall statistics and analytics."""
    model = DetectionSession
    template_name = 'results/statistics.html'
    context_object_name = 'sessions'
    
    def get_queryset(self):
        return DetectionSession.objects.filter(status='completed').order_by('-created_at')[:50]
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        all_sessions = DetectionSession.objects.filter(status='completed')
        
        # Overall stats
        context['total_sessions'] = all_sessions.count()
        context['total_detections'] = DetectionResult.objects.filter(
            session__status='completed'
        ).count()
        
        # Average metrics
        metrics = PerformanceMetrics.objects.filter(session__status='completed')
        avg_data = metrics.aggregate(
            avg_precision=Avg('precision'),
            avg_recall=Avg('recall'),
            avg_f1=Avg('f1_score'),
            avg_confidence=Avg('avg_confidence')
        )
        context['avg_metrics'] = avg_data
        
        # NMS distribution
        nms_dist = all_sessions.values('nms_method').annotate(
            count=Count('id')
        )
        context['nms_distribution'] = {item['nms_method']: item['count'] for item in nms_dist}
        
        return context
