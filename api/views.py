"""
REST API views for object detection.
"""

import os
import uuid
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image

from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from detection.models import (
    UploadedMedia, DetectionSession, DetectionResult, 
    PerformanceMetrics, NMSComparison
)
from detection.utils import (
    get_detector, calculate_detection_stats,
    calculate_precision_recall, calculate_map, compare_nms_methods
)


class DetectAPIView(APIView):
    """
    POST /api/detect/
    
    Upload an image or video for object detection.
    
    Parameters:
        - file: Image or video file
        - nms_method: 'standard', 'soft', or 'diou' (default: 'standard')
        - confidence: Confidence threshold 0-1 (default: 0.25)
        - iou_threshold: IoU threshold 0-1 (default: 0.45)
        - sigma: Soft-NMS sigma (default: 0.5)
        - target_classes: Comma-separated class IDs (optional)
    
    Returns:
        JSON with detection results, annotated image URL, and metrics.
    """
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        try:
            # Get file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return Response(
                    {'error': 'No file provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Parse parameters
            nms_method = request.data.get('nms_method', 'standard')
            if nms_method not in ['standard', 'soft', 'diou']:
                nms_method = 'standard'
            
            confidence = float(request.data.get('confidence', 0.25))
            iou_threshold = float(request.data.get('iou_threshold', 0.45))
            sigma = float(request.data.get('sigma', 0.5))
            
            # Validate file type
            filename = uploaded_file.name.lower()
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                media_type = 'image'
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                media_type = 'video'
            else:
                return Response(
                    {'error': 'Unsupported file format. Use JPG, PNG, MP4, etc.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Save file
            unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            save_path = f"uploads/{media_type}s/{unique_name}"
            saved_path = default_storage.save(save_path, uploaded_file)
            full_path = os.path.join(settings.MEDIA_ROOT, saved_path)
            
            # Create records
            media = UploadedMedia.objects.create(
                file=saved_path,
                media_type=media_type,
                original_filename=uploaded_file.name,
                file_size=uploaded_file.size
            )
            
            session = DetectionSession.objects.create(
                media=media,
                nms_method=nms_method,
                confidence_threshold=confidence,
                iou_threshold=iou_threshold,
                soft_nms_sigma=sigma,
                status='processing'
            )
            
            # Get detector
            detector = get_detector()
            
            if media_type == 'image':
                # Process image
                image = cv2.imread(full_path)
                if image is None:
                    session.mark_failed('Could not read image')
                    return Response(
                        {'error': 'Could not read image'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                result = detector.detect(
                    image,
                    confidence_threshold=confidence,
                    iou_threshold=iou_threshold,
                    nms_method=nms_method,
                    sigma=sigma
                )
                
                # Draw and save annotated image
                annotated = detector.draw_detections(image, result['detections'])
                result_filename = f"result_{session.id}_{uuid.uuid4().hex}.jpg"
                result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, annotated)
                
                session.result_image = f"results/{result_filename}"
                session.processing_time = result['total_time_ms'] / 1000
                session.fps = result['fps']
                session.status = 'completed'
                session.completed_at = timezone.now()
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
                
                # Calculate metrics
                stats = calculate_detection_stats(result['detections'])
                metrics = calculate_precision_recall(result['detections'])
                
                PerformanceMetrics.objects.create(
                    session=session,
                    total_objects=stats['total'],
                    avg_confidence=stats['avg_confidence'],
                    class_distribution=stats['class_distribution'],
                    occluded_objects=stats['occluded_count'],
                    avg_occlusion_score=stats.get('avg_occlusion_score', 0),
                    inference_time=result['inference_time_ms'],
                    nms_time=result['nms_time_ms'],
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    f1_score=metrics['f1_score']
                )
                
                return Response({
                    'success': True,
                    'session_id': session.id,
                    'media_type': media_type,
                    'nms_method': nms_method,
                    'detections': result['detections'],
                    'num_detections': result['num_detections'],
                    'annotated_image_url': f"/media/results/{result_filename}",
                    'processing_time_ms': result['total_time_ms'],
                    'fps': result['fps'],
                    'metrics': {
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'avg_confidence': stats['avg_confidence'],
                        'occluded_count': stats['occluded_count'],
                        'class_distribution': stats['class_distribution']
                    }
                })
            
            else:
                # Video processing
                result_filename = f"result_{session.id}_{uuid.uuid4().hex}.mp4"
                result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                
                result = detector.detect_video(
                    full_path,
                    output_path=result_path,
                    confidence_threshold=confidence,
                    iou_threshold=iou_threshold,
                    nms_method=nms_method,
                    sigma=sigma,
                    max_frames=300
                )
                
                session.result_video = f"results/{result_filename}"
                session.processing_time = result['total_frames'] / result['detection_fps'] if result['detection_fps'] > 0 else 0
                session.fps = result['detection_fps']
                session.status = 'completed'
                session.completed_at = timezone.now()
                session.save()
                
                return Response({
                    'success': True,
                    'session_id': session.id,
                    'media_type': media_type,
                    'nms_method': nms_method,
                    'total_detections': result['total_detections'],
                    'total_frames': result['total_frames'],
                    'annotated_video_url': f"/media/results/{result_filename}",
                    'detection_fps': result['detection_fps'],
                    'video_fps': result['video_fps']
                })
        
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DetectRealtimeAPIView(APIView):
    """
    POST /api/detect/realtime/
    
    Process a base64-encoded frame for real-time detection.
    
    Parameters:
        - frame: Base64-encoded image
        - nms_method: NMS method (default: 'standard')
        - confidence: Confidence threshold (default: 0.25)
        - iou_threshold: IoU threshold (default: 0.45)
    
    Returns:
        JSON with detections and base64 annotated frame.
    """
    parser_classes = [JSONParser]
    
    def post(self, request):
        try:
            frame_data = request.data.get('frame')
            if not frame_data:
                return Response(
                    {'error': 'No frame provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            nms_method = request.data.get('nms_method', 'standard')
            confidence = float(request.data.get('confidence', 0.25))
            iou_threshold = float(request.data.get('iou_threshold', 0.45))
            sigma = float(request.data.get('sigma', 0.5))
            
            # Decode base64 image
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return Response(
                    {'error': 'Could not decode frame'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get detector and run inference
            detector = get_detector()
            result = detector.detect(
                frame,
                confidence_threshold=confidence,
                iou_threshold=iou_threshold,
                nms_method=nms_method,
                sigma=sigma
            )
            
            # Draw annotations
            annotated = detector.draw_detections(frame, result['detections'])
            
            # Encode result as base64
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return Response({
                'success': True,
                'detections': result['detections'],
                'num_detections': result['num_detections'],
                'annotated_frame': f"data:image/jpeg;base64,{annotated_base64}",
                'processing_time_ms': result['total_time_ms'],
                'fps': result['fps']
            })
        
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ResultsAPIView(APIView):
    """
    GET /api/results/<id>/
    
    Retrieve detection results for a session.
    """
    
    def get(self, request, session_id):
        try:
            session = DetectionSession.objects.get(id=session_id)
            detections = session.detections.all()
            
            try:
                metrics = session.metrics
                metrics_data = metrics.to_dict()
            except PerformanceMetrics.DoesNotExist:
                metrics_data = None
            
            result_url = None
            if session.result_image:
                result_url = f"/media/{session.result_image}"
            elif session.result_video:
                result_url = f"/media/{session.result_video}"
            
            return Response({
                'session_id': session.id,
                'status': session.status,
                'nms_method': session.nms_method,
                'confidence_threshold': session.confidence_threshold,
                'iou_threshold': session.iou_threshold,
                'processing_time': session.processing_time,
                'fps': session.fps,
                'result_url': result_url,
                'detections': [d.to_dict() for d in detections],
                'metrics': metrics_data,
                'created_at': session.created_at.isoformat(),
                'completed_at': session.completed_at.isoformat() if session.completed_at else None
            })
        
        except DetectionSession.DoesNotExist:
            return Response(
                {'error': 'Session not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class MetricsAPIView(APIView):
    """
    GET /api/metrics/
    
    Get aggregated performance statistics.
    
    Query parameters:
        - nms_method: Filter by NMS method
        - limit: Number of recent sessions to include (default: 100)
    """
    
    def get(self, request):
        nms_method = request.query_params.get('nms_method')
        limit = int(request.query_params.get('limit', 100))
        
        sessions = DetectionSession.objects.filter(status='completed')
        if nms_method:
            sessions = sessions.filter(nms_method=nms_method)
        
        sessions = sessions.order_by('-created_at')[:limit]
        
        # Aggregate metrics
        total_sessions = sessions.count()
        total_detections = sum(s.detections.count() for s in sessions)
        
        avg_fps = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        
        metrics_count = 0
        for session in sessions:
            try:
                m = session.metrics
                if m.precision:
                    avg_precision += m.precision
                    avg_recall += m.recall or 0
                    avg_f1 += m.f1_score or 0
                    metrics_count += 1
            except:
                pass
            avg_fps += session.fps or 0
        
        if total_sessions > 0:
            avg_fps /= total_sessions
        if metrics_count > 0:
            avg_precision /= metrics_count
            avg_recall /= metrics_count
            avg_f1 /= metrics_count
        
        # NMS method distribution
        nms_distribution = {
            'standard': sessions.filter(nms_method='standard').count(),
            'soft': sessions.filter(nms_method='soft').count(),
            'diou': sessions.filter(nms_method='diou').count()
        }
        
        # Class distribution
        class_dist = {}
        for session in sessions:
            for det in session.detections.all():
                class_dist[det.class_name] = class_dist.get(det.class_name, 0) + 1
        
        return Response({
            'total_sessions': total_sessions,
            'total_detections': total_detections,
            'avg_fps': avg_fps,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'nms_distribution': nms_distribution,
            'class_distribution': class_dist
        })


class CompareNMSAPIView(APIView):
    """
    POST /api/compare-nms/
    
    Compare NMS methods side-by-side on the same image.
    
    Parameters:
        - file: Image file
        - confidence: Confidence threshold
        - iou_threshold: IoU threshold
        - sigma: Soft-NMS sigma
    
    Returns:
        Results from all three NMS methods.
    """
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        try:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return Response(
                    {'error': 'No file provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            confidence = float(request.data.get('confidence', 0.25))
            iou_threshold = float(request.data.get('iou_threshold', 0.45))
            sigma = float(request.data.get('sigma', 0.5))
            
            # Check file type
            filename = uploaded_file.name.lower()
            if not filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                return Response(
                    {'error': 'Only image files supported for comparison'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
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
            
            # Read image
            image = cv2.imread(full_path)
            if image is None:
                return Response(
                    {'error': 'Could not read image'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            detector = get_detector()
            results = {}
            
            for nms_method in ['standard', 'soft', 'diou']:
                result = detector.detect(
                    image,
                    confidence_threshold=confidence,
                    iou_threshold=iou_threshold,
                    nms_method=nms_method,
                    sigma=sigma
                )
                
                # Draw and save
                annotated = detector.draw_detections(image, result['detections'])
                result_filename = f"compare_{nms_method}_{uuid.uuid4().hex}.jpg"
                result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, annotated)
                
                stats = calculate_detection_stats(result['detections'])
                
                results[nms_method] = {
                    'num_detections': result['num_detections'],
                    'processing_time_ms': result['total_time_ms'],
                    'fps': result['fps'],
                    'annotated_image_url': f"/media/results/{result_filename}",
                    'avg_confidence': stats['avg_confidence'],
                    'occluded_count': stats['occluded_count'],
                    'class_distribution': stats['class_distribution'],
                    'detections': result['detections']
                }
            
            # Comparison summary
            comparison = compare_nms_methods(
                results['standard']['detections'],
                results['soft']['detections'],
                results['diou']['detections']
            )
            
            return Response({
                'success': True,
                'results': results,
                'comparison': comparison['summary'],
                'configuration': {
                    'confidence_threshold': confidence,
                    'iou_threshold': iou_threshold,
                    'sigma': sigma
                }
            })
        
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
