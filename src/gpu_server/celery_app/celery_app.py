import os
# CRITICAL: Set multiprocessing start method FIRST, before any other imports
# This must be done before importing torch, celery, or any CUDA-related code
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, that's fine
    pass

from celery import Celery
from kombu import Queue


# Load Celery configuration from environment variables
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")
CELERY_TASK_SERIALIZER = os.getenv("CELERY_TASK_SERIALIZER", "json")
CELERY_ACCEPT_CONTENT = os.getenv("CELERY_ACCEPT_CONTENT", "json").split(",")
CELERY_RESULT_SERIALIZER = os.getenv("CELERY_RESULT_SERIALIZER", "json")
CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "UTC")
CELERY_ENABLE_UTC = os.getenv("CELERY_ENABLE_UTC", "true").lower() == "true"
CELERY_TASK_TRACK_STARTED = os.getenv("CELERY_TASK_TRACK_STARTED", "true").lower() == "true"
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600"))
CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "3000"))
CELERY_WORKER_PREFETCH_MULTIPLIER = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1"))
CELERY_WORKER_MAX_TASKS_PER_CHILD = int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "1000"))

task_queues = (
    Queue("ocr"),
    Queue("gpt"),
)

task_routes = {
    "src.gpu_server.celery_app.gpt_tasks.process_chat_completion": {"queue": "gpt"},
    "src.gpu_server.celery_app.ocr_tasks.*": {"queue": "ocr"},
}

celery_app = Celery(
    "gpu_server",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["src.gpu_server.celery_app.gpt_tasks",
             "src.gpu_server.celery_app.ocr_tasks"],
)

# Discover tasks in the celery_app package
celery_app.autodiscover_tasks([
    "src.gpu_server.celery_app"
])

celery_app.conf.update(
    task_serializer=CELERY_TASK_SERIALIZER,
    accept_content=CELERY_ACCEPT_CONTENT,
    result_serializer=CELERY_RESULT_SERIALIZER,
    timezone=CELERY_TIMEZONE,
    enable_utc=CELERY_ENABLE_UTC,
    task_track_started=CELERY_TASK_TRACK_STARTED,
    task_time_limit=CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=CELERY_TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=CELERY_WORKER_PREFETCH_MULTIPLIER,
    worker_max_tasks_per_child=CELERY_WORKER_MAX_TASKS_PER_CHILD,
    task_queues=task_queues,
    task_routes=task_routes,
    # CRITICAL: Use solo pool to avoid CUDA fork issues
    # Solo pool runs tasks in the main worker process (no forking)
    worker_pool='solo',
)

