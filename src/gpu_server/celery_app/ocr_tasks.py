import celery
from celery import Task
from celery.worker.control import time_limit

from src.gpu_server.celery_app.celery_app import celery_app
from src.gpu_server.database import SessionLocal, OCRTask, GPTTask
from src.ocr.deepseek_ocr_service import get_ocr_service
from src.gpt_oss.gpt_oss_service import get_gpt_service
import traceback
from datetime import datetime, timezone
import logging
from dotenv import load_dotenv
import os
load_dotenv()

from typing import List, Dict, Optional

import sys
IS_FLOWER =  "flower" in sys.argv
IS_CELERY_WORKER = "ocr_worker" in sys.argv
logger = logging.getLogger(__name__)
TIMEOUT = int(os.environ.get("CELERY_OCR_TIMEOUT", 300))


# Load OCR service once at module import time - it will persist for worker lifetime
logger.info("Initializing OCR service for worker process...")
logger.info(f"IS_CELERY_WORKER: {IS_CELERY_WORKER}, IS_FLOWER: {IS_FLOWER}")

ocr_service = None


def get_shared_ocr_service():
    """Initialize OCR service only on the worker, and only once per process."""
    global ocr_service
    if ocr_service is None:
        from celery import current_app
        # Only load in worker process, not when starting Flower
        if IS_CELERY_WORKER:
            logger.info("Worker detected — initializing OCR model...")
            ocr_service = get_ocr_service()
        else:
            logger.info("Not a worker — skipping OCR model load (e.g., Flower process).")
    return ocr_service


ocr_service = get_shared_ocr_service()
logger.info("OCR service initialized and ready")


class DatabaseTask(Task):
    """Base task that provides database session and AI services"""
    _db = None

    @property
    def db(self):
        """Get database session (per-task)"""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    @property
    def gpt_service(self):
        """Get GPT service (shared across all tasks in this worker)"""
        return get_shared_ocr_service()


    def after_return(self, *args, **kwargs):
        """Clean up database connection after task completes"""
        if self._db is not None:
            self._db.close()
            self._db = None


import concurrent.futures

def safe_perform_ocr(ocr_service, image_path, prompt, timeout=TIMEOUT):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(ocr_service.perform_ocr, image_path, prompt)
        try:
            result = future.result(timeout=timeout)
            logger.debug(f"OCR result: {result}")
            return result
        except concurrent.futures.TimeoutError:
            logger.debug(f"OCR took too long (>{timeout}s). Timeout!")
            return None

@celery_app.task(base=DatabaseTask, bind=True, name="src.gpu_server.celery_app.ocr_tasks.process_chat_completion", max_retries = 0)
def process_ocr_task(self, task_id: str, image_path: str, prompt: str):
    """
    Process OCR task using DeepSeek OCR model
    """
    db = self.db
    self.ocr_service = get_shared_ocr_service()
    try:
        # Update task status to processing
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = "processing"
        db.commit()

        # Perform OCR
        ocr_result = safe_perform_ocr(self.ocr_service, image_path, prompt, timeout=TIMEOUT)        # Update task with results
        if ocr_result is None:
            logger.debug(f"OCR processing for task {task_id} timed out")
            raise TimeoutError("OCR processing timed out")
        task.status = "completed"
        task.result = {
            "text": ocr_result.get("text", ""),
            "metadata": ocr_result.get("metadata", {}),
            "usage": {
                "prompt_tokens": ocr_result.get("prompt_tokens", 0),
                "completion_tokens": ocr_result.get("completion_tokens", 0),
                "total_tokens": ocr_result.get("total_tokens", 0)
            }
        }
        task.updated_at = datetime.now(timezone.utc)
        db.commit()

        return {
            "task_id": task_id,
            "status": "completed",
            "result": task.result
        }

    except Exception as e:
        # Update task with error
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if task:
            task.status = "failed"
            task.error = str(e)
            task.updated_at = datetime.now(timezone.utc)
            db.commit()

        # Log the error
        logger.error(f"Error processing OCR task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())

        raise
