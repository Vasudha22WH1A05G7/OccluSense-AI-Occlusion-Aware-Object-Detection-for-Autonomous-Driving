"""
URL configuration for detection app.
"""

from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('detect/', views.DetectionView.as_view(), name='detect'),
    path('results/<int:pk>/', views.ResultsView.as_view(), name='results'),
    path('results/', views.ResultsListView.as_view(), name='results_list'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('compare/', views.CompareNMSView.as_view(), name='compare'),
    path('compare/process/', views.CompareNMSProcessView.as_view(), name='compare_process'),
    path('webcam/', views.WebcamView.as_view(), name='webcam'),
]
