"""
Detection views for web interface.
"""

import os
import json
import uuid
import cv2
import numpy as np
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.views.generic import TemplateView, DetailView, ListView
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
import threading
import logging

logger = logging.getLogger(__name__)

from .models import (
    UploadedMedia, DetectionSession, DetectionResult, 
    PerformanceMetrics, NMSComparison
)
from .utils import (
    get_detector, calculate_detection_stats, 
    calculate_precision_recall, calculate_map
)


class HomeView(TemplateView):
    """Home page with upload form."""
    template_name = 'detection/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_sessions'] = DetectionSession.objects.filter(
            status='completed'
        ).order_by('-created_at')[:5]
        return context


class UploadView(TemplateView):
    """Upload page for images and videos."""
    template_name = 'detection/upload.html'


class DetectionView(View):
    """Handle detection requests."""
    
    def post(self, request):
        """Process uploaded file for detection."""
        try:
            # Get uploaded file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            # Get detection parameters
            nms_method = request.POST.get('nms_method', 'standard')
            confidence_threshold = float(request.POST.get('confidence', 0.25))
            iou_threshold = float(request.POST.get('iou_threshold', 0.45))
            sigma = float(request.POST.get('sigma', 0.5))
            
            # Determine media type
            filename = uploaded_file.name.lower()
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                media_type = 'image'
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                media_type = 'video'
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
            
            # Save uploaded file
            unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            save_path = f"uploads/{media_type}s/{unique_name}"
            saved_path = default_storage.save(save_path, uploaded_file)
            full_path = os.path.join(settings.MEDIA_ROOT, saved_path)
            
            # Create media record
            media = UploadedMedia.objects.create(
                file=saved_path,
                media_type=media_type,
                original_filename=uploaded_file.name,
                file_size=uploaded_file.size
            )
            
            # Create detection session
            session = DetectionSession.objects.create(
                media=media,
                nms_method=nms_method,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                soft_nms_sigma=sigma,
                status='processing'
            )
            
            # Start processing in background thread
            def process_task():
                try:
                    # Get detector
                    detector = get_detector()
                    
                    if media_type == 'image':
                        self._process_image(
                            full_path, detector, session,
                            confidence_threshold, iou_threshold, nms_method, sigma
                        )
                    else:
                        self._process_video(
                            full_path, detector, session,
                            confidence_threshold, iou_threshold, nms_method, sigma
                        )
                    
                    # Mark completed
                    session.mark_completed()
                    
                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    session.mark_failed(str(e))

            thread = threading.Thread(target=process_task)
            thread.daemon = True
            thread.start()
            
            return JsonResponse({
                'success': True,
                'session_id': session.id,
                'redirect_url': f'/results/{session.id}/'
            })
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    def _process_image(self, path, detector, session, conf, iou, nms_method, sigma):
        """Process a single image."""
        import time
        start_time = time.time()
        
        # Read image
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Could not read image")
        
        # Run detection
        result = detector.detect(
            image,
            confidence_threshold=conf,
            iou_threshold=iou,
            nms_method=nms_method,
            sigma=sigma
        )
        
        # Draw annotations
        annotated = detector.draw_detections(image, result['detections'])
        
        # Save annotated image
        result_filename = f"result_{session.id}_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, annotated)
        
        session.result_image = f"results/{result_filename}"
        session.environment_warning = result.get('environment_warning')
        session.driving_suggestions = json.dumps(result.get('driving_suggestions', []))
        session.processing_time = time.time() - start_time
        session.fps = result['fps']
        session.save()
        
        # Save detections
        for det in result['detections']:
            DetectionResult.objects.create(
                session=session,
                class_id=det['class_id'],
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox_x1=det['bbox'][0],
                bbox_y1=det['bbox'][1],
                bbox_x2=det['bbox'][2],
                bbox_y2=det['bbox'][3],
                is_occluded=det['is_occluded'],
                occlusion_score=det['occlusion_score'],
                overlapping_objects=det['overlapping_objects']
            )
        
        # Calculate and save metrics
        stats = calculate_detection_stats(result['detections'])
        metrics_data = calculate_precision_recall(result['detections'])
        map_data = calculate_map(result['detections'])
        
        PerformanceMetrics.objects.create(
            session=session,
            total_objects=stats['total'],
            avg_confidence=stats['avg_confidence'],
            class_distribution=stats['class_distribution'],
            occluded_objects=stats['occluded_count'],
            avg_occlusion_score=stats['avg_occlusion_score'],
            inference_time=result['inference_time_ms'],
            nms_time=result['nms_time_ms'],
            total_frames=1,
            precision=metrics_data['precision'],
            recall=metrics_data['recall'],
            f1_score=metrics_data['f1_score'],
            mAP=map_data['mAP']
        )
        
        return result
    
    def _process_video(self, path, detector, session, conf, iou, nms_method, sigma):
        """Process a video file."""
        import time
        start_time = time.time()
        
        # Output path for annotated video
        result_filename = f"result_{session.id}_{uuid.uuid4().hex}.mp4"
        result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        # Process video
        result = detector.detect_video(
            path,
            output_path=result_path,
            confidence_threshold=conf,
            iou_threshold=iou,
            nms_method=nms_method,
            sigma=sigma,
            max_frames=300  # Limit for demo
        )
        
        session.result_video = f"results/{result_filename}"
        session.environment_warning = result.get('environment_warning')
        session.driving_suggestions = json.dumps(result.get('driving_suggestions', []))
        session.processing_time = time.time() - start_time
        session.fps = result['detection_fps']
        session.save()
        
        # Save sample detections (every 10th frame to avoid too many records)
        for det in result['detections'][::10]:
            DetectionResult.objects.create(
                session=session,
                frame_number=det.get('frame_number', 0),
                class_id=det['class_id'],
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox_x1=det['bbox'][0],
                bbox_y1=det['bbox'][1],
                bbox_x2=det['bbox'][2],
                bbox_y2=det['bbox'][3],
                is_occluded=det.get('is_occluded', False),
                occlusion_score=det.get('occlusion_score', 0),
                overlapping_objects=det.get('overlapping_objects', 0)
            )
        
        # Calculate metrics
        stats = calculate_detection_stats(result['detections'])
        metrics_data = calculate_precision_recall(result['detections'])
        
        PerformanceMetrics.objects.create(
            session=session,
            total_objects=stats['total'],
            avg_confidence=stats['avg_confidence'],
            class_distribution=stats['class_distribution'],
            occluded_objects=stats['occluded_count'],
            avg_occlusion_score=stats.get('avg_occlusion_score', 0),
            inference_time=result['avg_frame_time_ms'],
            nms_time=0,
            total_frames=result['total_frames'],
            precision=metrics_data['precision'],
            recall=metrics_data['recall'],
            f1_score=metrics_data['f1_score']
        )
        
        return result


class ResultsView(DetailView):
    """View detection results."""
    model = DetectionSession
    template_name = 'detection/results.html'
    context_object_name = 'session'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        session = self.object
        
        # Get detections
        detections = session.detections.all()
        context['detections'] = detections
        context['detections_json'] = json.dumps([d.to_dict() for d in detections])
        
        # Get metrics if available
        try:
            context['metrics'] = session.metrics
        except PerformanceMetrics.DoesNotExist:
            context['metrics'] = None
            
        # Parse driving suggestions
        if session.driving_suggestions:
            try:
                context['suggestions'] = json.loads(session.driving_suggestions)
            except:
                context['suggestions'] = []
        
        return context


class ResultsListView(ListView):
    """List all detection sessions."""
    model = DetectionSession
    template_name = 'detection/results_list.html'
    context_object_name = 'sessions'
    paginate_by = 10
    
    def get_queryset(self):
        return DetectionSession.objects.filter(
            status='completed'
        ).order_by('-created_at')


class DashboardView(TemplateView):
    """Analytics dashboard."""
    template_name = 'detection/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get all completed sessions
        sessions = DetectionSession.objects.filter(status='completed')
        
        # Aggregate statistics
        total_sessions = sessions.count()
        total_detections = DetectionResult.objects.filter(
            session__status='completed'
        ).count()
        
        # NMS method distribution
        nms_counts = {}
        for method in ['standard', 'soft', 'diou']:
            nms_counts[method] = sessions.filter(nms_method=method).count()
        
        # Class distribution
        class_dist = {}
        for det in DetectionResult.objects.filter(session__status='completed'):
            class_dist[det.class_name] = class_dist.get(det.class_name, 0) + 1
        
        # Performance over time (last 10 sessions)
        recent_sessions = sessions.order_by('-created_at')[:10]
        performance_data = []
        for s in recent_sessions:
            try:
                metrics = s.metrics
                performance_data.append({
                    'id': s.id,
                    'date': s.created_at.isoformat(),
                    'fps': s.fps,
                    'precision': metrics.precision or 0,
                    'recall': metrics.recall or 0,
                    'nms_method': s.nms_method
                })
            except:
                pass
        
        context.update({
            'total_sessions': total_sessions,
            'total_detections': total_detections,
            'nms_counts': nms_counts,
            'class_distribution': class_dist,
            'performance_data': json.dumps(performance_data[::-1]),
            'nms_counts_json': json.dumps(nms_counts),
            'class_dist_json': json.dumps(class_dist)
        })
        
        return context


class CompareNMSView(TemplateView):
    """Compare NMS methods side-by-side."""
    template_name = 'detection/compare.html'


class CompareNMSProcessView(View):
    """Process NMS comparison."""
    
    def post(self, request):
        """Run detection with all NMS methods and compare."""
        try:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            confidence = float(request.POST.get('confidence', 0.25))
            iou_threshold = float(request.POST.get('iou_threshold', 0.45))
            sigma = float(request.POST.get('sigma', 0.5))
            
            # Save file
            unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            save_path = f"uploads/images/{unique_name}"
            saved_path = default_storage.save(save_path, uploaded_file)
            full_path = os.path.join(settings.MEDIA_ROOT, saved_path)
            
            # Create media record
            media = UploadedMedia.objects.create(
                file=saved_path,
                media_type='image',
                original_filename=uploaded_file.name,
                file_size=uploaded_file.size
            )
            
            # Get detector
            detector = get_detector()
            
            # Read image
            image = cv2.imread(full_path)
            if image is None:
                return JsonResponse({'error': 'Could not read image'}, status=400)
            
            results = {}
            sessions = {}
            
            # Run detection with each NMS method
            for nms_method in ['standard', 'soft', 'diou']:
                result = detector.detect(
                    image,
                    confidence_threshold=confidence,
                    iou_threshold=iou_threshold,
                    nms_method=nms_method,
                    sigma=sigma
                )
                
                # Draw and save annotated image
                annotated = detector.draw_detections(image, result['detections'])
                result_filename = f"compare_{nms_method}_{uuid.uuid4().hex}.jpg"
                result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, annotated)
                
                # Create session
                session = DetectionSession.objects.create(
                    media=media,
                    nms_method=nms_method,
                    confidence_threshold=confidence,
                    iou_threshold=iou_threshold,
                    soft_nms_sigma=sigma,
                    result_image=f"results/{result_filename}",
                    processing_time=result['total_time_ms'] / 1000,
                    fps=result['fps'],
                    status='completed',
                    completed_at=timezone.now()
                )
                sessions[nms_method] = session
                
                # Save detections
                for det in result['detections']:
                    DetectionResult.objects.create(
                        session=session,
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        confidence=det['confidence'],
                        bbox_x1=det['bbox'][0],
                        bbox_y1=det['bbox'][1],
                        bbox_x2=det['bbox'][2],
                        bbox_y2=det['bbox'][3],
                        is_occluded=det['is_occluded'],
                        occlusion_score=det['occlusion_score'],
                        overlapping_objects=det['overlapping_objects']
                    )
                
                results[nms_method] = {
                    'session_id': session.id,
                    'image_url': f"/media/results/{result_filename}",
                    'num_detections': result['num_detections'],
                    'processing_time_ms': result['total_time_ms'],
                    'fps': result['fps'],
                    'detections': result['detections']
                }
            
            # Create comparison record
            comparison = NMSComparison.objects.create(
                media=media,
                standard_session=sessions['standard'],
                soft_session=sessions['soft'],
                diou_session=sessions['diou'],
                confidence_threshold=confidence,
                iou_threshold=iou_threshold,
                soft_nms_sigma=sigma
            )
            
            return JsonResponse({
                'success': True,
                'comparison_id': comparison.id,
                'results': results
            })
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class WebcamView(TemplateView):
    """Real-time webcam detection page."""
    template_name = 'detection/webcam.html'
