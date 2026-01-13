import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


from celery import Task
from src.gpu_server.celery_app import celery_app
from src.gpu_server.database import SessionLocal, OCRTask, GPTTask
from src.ocr.deepseek_ocr_service import get_ocr_service
from src.gpt_oss.gpt_oss_service import get_gpt_service
import traceback
from datetime import datetime, timezone
import logging

from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)



# Load OCR service once at module import time - it will persist for worker lifetime
logger.info("Initializing OCR service for worker process...")
ocr_service = get_ocr_service()
logger.info("OCR service initialized and ready")

# Load GPT service once at module import time - it will persist for worker lifetime
logger.info("Initializing GPT service for worker process...")
gpt_service = get_gpt_service()  # Default model
logger.info("GPT service initialized and ready")


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

    @property
    def gpt_service(self):
        """Get GPT service (shared across all tasks in this worker)"""
        return gpt_service

    def after_return(self, *args, **kwargs):
        """Clean up database connection after task completes"""
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(base=DatabaseTask, bind=True, name="tasks.process_ocr")
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


@celery_app.task(base=DatabaseTask, bind=True, name="tasks.process_gpt")
def process_gpt_task(
    self,
    task_id: str,
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    n: int = 1
):
    """
    Process GPT chat completion task

    Args:
        task_id: Unique task identifier
        messages: List of chat messages with 'role' and 'content'
        model_name: Model to use (e.g., 'openai/gpt-oss-20b')
        temperature: Sampling temperature (0.0 to 2.0)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        stop: List of stop sequences
        presence_penalty: Presence penalty (-2.0 to 2.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        n: Number of completions to generate
    """
    db = self.db

    try:
        # Update task status to processing
        task = db.query(GPTTask).filter(GPTTask.id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = "processing"
        db.commit()

        # Get the appropriate GPT service (uses model from task or default)
        gpt_service_instance = get_gpt_service(model_name or task.model_name)

        logger.info(f"Processing GPT task {task_id} with model {gpt_service_instance.model_name}")
        logger.info(f"Message count: {len(messages)}")

        # Generate chat completion
        result = gpt_service_instance.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n
        )

        # Update task with results
        task.status = "completed"
        task.result = {
            "text": result.get("text", ""),
            "messages": result.get("messages", []),
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            },
            "model": gpt_service_instance.model_name,
            "temperature": result.get("temperature"),
            "top_p": result.get("top_p"),
            "max_tokens": result.get("max_tokens")
        }
        task.updated_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(f"GPT task {task_id} completed successfully")
        logger.info(f"Tokens used: {result.get('total_tokens', 0)}")

        return {
            "task_id": task_id,
            "status": "completed",
            "result": task.result
        }

    except Exception as e:
        # Update task with error
        task = db.query(GPTTask).filter(GPTTask.id == task_id).first()
        if task:
            task.status = "failed"
            task.error = str(e)
            task.updated_at = datetime.now(timezone.utc)
            db.commit()

        # Log the error
        logger.error(f"Error processing GPT task {task_id}: {str(e)}")
        logger.error(traceback.format_exc())

        raise


