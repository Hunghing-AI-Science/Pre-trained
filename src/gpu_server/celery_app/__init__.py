"""
Celery application and tasks for GPU server
"""
from .celery_app import celery_app

__all__ = ["celery_app"]

