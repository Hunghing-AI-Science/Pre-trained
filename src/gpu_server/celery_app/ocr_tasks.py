import multiprocessing

import celery
from celery import Task
from celery.worker.control import time_limit

from src.gpu_server.celery_app.celery_app import app
from src.gpu_server.database import SessionLocal, OCRTask
from src.ocr.deepseek_ocr_service import get_ocr_service
import traceback
from datetime import datetime, timezone
import logging
from dotenv import load_dotenv
import os
load_dotenv()

from typing import List, Dict, Optional

import sys
IS_FLOWER = "flower" in sys.argv
IS_CELERY_WORKER = "ocr_worker" in sys.argv  # exact element match: won't match "vllm_ocr_worker"
logger = logging.getLogger(__name__)
TIMEOUT = int(os.environ.get("CELERY_OCR_TIMEOUT", 300))

ocr_service = None


def get_shared_ocr_service():
    """Initialize OCR service only on the ocr_worker process, and only once."""
    global ocr_service
    if ocr_service is None:
        if IS_FLOWER:
            logger.info("[ocr_worker] Flower process detected — skipping OCR model load.")
        elif IS_CELERY_WORKER:
            logger.info("[ocr_worker] Worker detected — initializing OCR model...")
            ocr_service = get_ocr_service()
            logger.info("[ocr_worker] OCR service initialized and ready.")
        else:
            logger.debug("[ocr_worker] Not an ocr_worker process — skipping OCR model load.")
    return ocr_service


# Eagerly pre-load inside a real ocr_worker so the model is warm before the first task
if IS_CELERY_WORKER:
    get_shared_ocr_service()


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
            try:
                # Rollback any active transaction before closing
                self._db.rollback()
                self._db.close()
            except Exception as e:
                logger.warning(f"Error closing database session: {e}")
            finally:
                self._db = None





def _ocr_worker(queue, service, image_path, prompt):
    """Separate top-level OCR subprocess target"""
    try:
        result = service.perform_ocr(image_path, prompt)
        queue.put(result)
    except Exception as e:
        logger.error(f"Error in OCR subprocess: {e}", exc_info=True)
        queue.put({"error": str(e)})

def perform_ocr_in_process(ocr_service, image_path, prompt, timeout):
    """Run OCR in an isolated process with enforced timeout."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_ocr_worker,
        args=(queue, ocr_service, image_path, prompt)
    )

    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        logger.warning(f"OCR subprocess timed out after {timeout}s")
        return None

    if not queue.empty():
        result = queue.get()
        if isinstance(result, dict) and "error" in result:
            logger.error(f"OCR subprocess reported error: {result['error']}")
            return None
        return result

    logger.warning("OCR subprocess finished with no result")
    return None

@app.task(
    base=DatabaseTask,
    bind=True,
    name="src.gpu_server.celery_app.ocr_tasks.process_chat_completion",
    soft_time_limit=TIMEOUT - 5,  # warning signal before kill
    time_limit=TIMEOUT,  # hard kill after TIMEOUT
    max_retries=0
)
def process_ocr_task(self, task_id: str, image_path: str, prompt: str):
    """
    Process OCR task using DeepSeek OCR model
    """
    logger.info(f"Processing OCR task {task_id}")
    db = self.db
    self.ocr_service = get_shared_ocr_service()
    logger.info(f"OCR service: {self.ocr_service}. TIMEOUT: {TIMEOUT}s")
    start_time = datetime.now(timezone.utc)
    try:
        # Update task status to processing
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = "processing"
        db.commit()
        logger.info(f"OCR task {task_id} status updated to processing")

        # Perform OCR
        ocr_result = perform_ocr_in_process(self.ocr_service, image_path, prompt, TIMEOUT)

        logger.info(f"Received. OCR result: {ocr_result}")
        end_time = datetime.now(timezone.utc)
        logger.info(f"OCR task {task_id} completed in {(end_time - start_time).total_seconds()} seconds")
        if ocr_result is None:
            logger.debug(f"OCR processing for task {task_id} timed out")
            return {
                "task_id": task_id,
                "status": "failed",
                "result": None
            }
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
