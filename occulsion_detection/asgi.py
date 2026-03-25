"""
ASGI config for occlusion_detection project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'occlusion_detection.settings')

application = get_asgi_application()
