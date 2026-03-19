"""
vllm_ocr_tasks.py
-----------------
Celery tasks that use the vLLM-backed DeepSeek OCR service.

Queue:  vllm_ocr
Tasks:
  - process_vllm_ocr_task   — single-image OCR
  - process_vllm_ocr_batch  — multi-image batch OCR
"""

import sys
import traceback
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

from celery import Task
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TIMEOUT = int(os.environ.get("CELERY_VLLM_OCR_TIMEOUT", 600))

IS_FLOWER = "flower" in sys.argv
IS_CELERY_WORKER = "vllm_ocr_worker" in sys.argv

# ---------------------------------------------------------------------------
# Service initialisation (lazy, singleton per worker process)
# ---------------------------------------------------------------------------

_vllm_ocr_service = None


def get_shared_vllm_ocr_service():
    """
    Initialise the vLLM OCR service only inside a worker process and only once.
    """
    global _vllm_ocr_service
    if _vllm_ocr_service is None:
        if IS_FLOWER:
            logger.info("Flower process detected — skipping vLLM OCR model load.")
        else:
            logger.info("Initialising vLLM OCR service for worker process…")
            from src.vllm.deepseek_ocr_vllm_service import get_vllm_ocr_service
            _vllm_ocr_service = get_vllm_ocr_service()
            logger.info("vLLM OCR service ready.")
    return _vllm_ocr_service


# Eagerly load inside a real worker so the GPU model is warm before the first task
if IS_CELERY_WORKER:
    get_shared_vllm_ocr_service()

# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------

from src.gpu_server.celery_app.celery_app import app
from src.gpu_server.database import SessionLocal, OCRTask


class VllmOCRDatabaseTask(Task):
    """Base task providing a per-task DB session and the shared vLLM OCR service."""

    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    @property
    def vllm_ocr_service(self):
        return get_shared_vllm_ocr_service()

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            try:
                self._db.rollback()
                self._db.close()
            except Exception as exc:
                logger.warning(f"Error closing DB session: {exc}")
            finally:
                self._db = None


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.task(
    base=VllmOCRDatabaseTask,
    bind=True,
    name="src.gpu_server.celery_app.vllm_ocr_tasks.process_vllm_ocr_task",
    queue="vllm_ocr",
    soft_time_limit=TIMEOUT - 10,
    time_limit=TIMEOUT,
    max_retries=0,
)
def process_vllm_ocr_task(
    self,
    task_id: str,
    image_path: str,
    prompt: str = "Free OCR.",
) -> Dict:
    """
    Process a single-image OCR request via the vLLM DeepSeek OCR model.

    Args:
        task_id:    UUID of the OCRTask row in the database.
        image_path: Absolute path to the image on the GPU server.
        prompt:     OCR instruction sent to the model.

    Returns:
        dict with keys: task_id, status, result
    """
    logger.info(f"[vLLM OCR] Starting task {task_id} | image={image_path}")
    db = self.db
    service = self.vllm_ocr_service
    start_time = datetime.now(timezone.utc)

    try:
        # ---- Mark task as processing ----
        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if task is None:
            raise ValueError(f"OCRTask {task_id} not found in database.")

        task.status = "processing"
        task.updated_at = datetime.now(timezone.utc)
        db.commit()

        # ---- Run inference ----
        ocr_result = service.perform_ocr(image_path=image_path, prompt=prompt)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"[vLLM OCR] Task {task_id} completed in {elapsed:.2f}s")

        # ---- Persist result ----
        task.status = "completed"
        task.result = {
            "text": ocr_result.get("text", ""),
            "metadata": ocr_result.get("metadata", {}),
            "usage": ocr_result.get("usage", {}),
        }
        task.updated_at = datetime.now(timezone.utc)
        db.commit()

        return {
            "task_id": task_id,
            "status": "completed",
            "result": task.result,
        }

    except Exception as exc:
        logger.error(f"[vLLM OCR] Task {task_id} failed: {exc}")
        logger.error(traceback.format_exc())

        task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
        if task:
            task.status = "failed"
            task.error = str(exc)
            task.updated_at = datetime.now(timezone.utc)
            db.commit()

        raise


@app.task(
    base=VllmOCRDatabaseTask,
    bind=True,
    name="src.gpu_server.celery_app.vllm_ocr_tasks.process_vllm_ocr_batch",
    queue="vllm_ocr",
    soft_time_limit=TIMEOUT - 10,
    time_limit=TIMEOUT,
    max_retries=0,
)
def process_vllm_ocr_batch(
    self,
    requests: List[Dict],
) -> List[Dict]:
    """
    Process a batch of OCR requests in a single vLLM forward pass.

    Each element of *requests* must contain:
        - task_id    (str)  : UUID of the OCRTask row
        - image_path (str)  : absolute path to the image
        - prompt     (str, optional): OCR instruction (default "Free OCR.")

    Returns:
        List of dicts with keys: task_id, status, result
    """
    logger.info(f"[vLLM OCR Batch] Processing {len(requests)} requests.")
    db = self.db
    service = self.vllm_ocr_service
    start_time = datetime.now(timezone.utc)

    task_ids = [r["task_id"] for r in requests]

    try:
        # ---- Mark all tasks as processing ----
        for task_id in task_ids:
            task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
            if task:
                task.status = "processing"
                task.updated_at = datetime.now(timezone.utc)
        db.commit()

        # ---- Run batch inference ----
        ocr_results = service.perform_ocr_batch(requests)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"[vLLM OCR Batch] Completed in {elapsed:.2f}s")

        # ---- Persist results ----
        response = []
        for req, ocr_result in zip(requests, ocr_results):
            task_id = req["task_id"]
            task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
            result_payload = {
                "text": ocr_result.get("text", ""),
                "metadata": ocr_result.get("metadata", {}),
                "usage": ocr_result.get("usage", {}),
            }
            if task:
                task.status = "completed"
                task.result = result_payload
                task.updated_at = datetime.now(timezone.utc)
            response.append(
                {"task_id": task_id, "status": "completed", "result": result_payload}
            )
        db.commit()

        return response

    except Exception as exc:
        logger.error(f"[vLLM OCR Batch] Batch failed: {exc}")
        logger.error(traceback.format_exc())

        for task_id in task_ids:
            task = db.query(OCRTask).filter(OCRTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error = str(exc)
                task.updated_at = datetime.now(timezone.utc)
        db.commit()

        raise

