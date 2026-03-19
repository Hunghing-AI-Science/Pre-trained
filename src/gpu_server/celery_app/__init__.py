"""
Celery application and tasks for GPU server
"""
from src.gpu_server.celery_app import celery_app

__all__ = ["celery_app"]

