from celery import Task
from src.gpu_server.celery_app.celery_app import celery_app
from src.gpu_server.database import SessionLocal, OCRTask, GPTTask
from src.ocr.deepseek_ocr_service import get_ocr_service
from src.gpt_oss.gpt_oss_service import get_gpt_service
import traceback
from datetime import datetime, timezone
import logging

from typing import List, Dict, Optional

logger = logging.getLogger(__name__)



# Load OCR service once at module import time - it will persist for worker lifetime
logger.info("Initializing OCR service for worker process...")
ocr_service = get_ocr_service()
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
    def ocr_service(self):
        """Get OCR service (shared across all tasks in this worker)"""
        return ocr_service


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


@celery_app.task(base=DatabaseTask, bind=True, name="tasks.process_ocr", time_limit=180, soft_time_limit=170)
def process_ocr_task(self, task_id: str, image_path: str, prompt: str):
    """
    Process OCR task using DeepSeek OCR model
    """
    db = self.db

    try:
        # Update task status to processing
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = "processing"
        db.commit()

        # Perform OCR
        ocr_result = self.ocr_service.perform_ocr(image_path, prompt)

        # Update task with results
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
