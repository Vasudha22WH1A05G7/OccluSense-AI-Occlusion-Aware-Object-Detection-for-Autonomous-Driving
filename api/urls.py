"""
URL configuration for API app.
"""

from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('detect/', views.DetectAPIView.as_view(), name='detect'),
    path('detect/realtime/', views.DetectRealtimeAPIView.as_view(), name='detect_realtime'),
    path('results/<int:session_id>/', views.ResultsAPIView.as_view(), name='results'),
    path('metrics/', views.MetricsAPIView.as_view(), name='metrics'),
    path('compare-nms/', views.CompareNMSAPIView.as_view(), name='compare_nms'),
]
