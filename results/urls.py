"""
URL configuration for results app.
"""

from django.urls import path
from . import views

app_name = 'results'

urlpatterns = [
    path('browser/', views.ResultsBrowserView.as_view(), name='browser'),
    path('detail/<int:pk>/', views.ResultDetailView.as_view(), name='detail'),
    path('statistics/', views.StatisticsView.as_view(), name='statistics'),
]
