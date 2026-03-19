import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import os
from celery import Celery
from dotenv import load_dotenv


load_dotenv()

BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

celery_app = Celery(
    "pretrained_vllm_tasks",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

# Reasonable defaults; override via env if you want.
celery_app.conf.update(
    task_serializer=os.getenv("CELERY_TASK_SERIALIZER", "json"),
    accept_content=[os.getenv("CELERY_ACCEPT_CONTENT", "json")],
    result_serializer=os.getenv("CELERY_RESULT_SERIALIZER", "json"),
    timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
    enable_utc=os.getenv("CELERY_ENABLE_UTC", "true").lower() == "true",
    task_track_started=os.getenv("CELERY_TASK_TRACK_STARTED", "true").lower() == "true",
)