
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

# Initialize global GPT service variable
gpt_service = None

def get_shared_gpt_service():
    """Get or initialize the shared GPT service for this worker process"""
    global gpt_service
    if gpt_service is None:
        logger.info("Initializing GPT service for worker process...")
        gpt_service = get_gpt_service()
        logger.info("GPT service initialized and ready")
    return gpt_service



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
        return get_shared_gpt_service()

    def after_return(self, *args, **kwargs):
        """Clean up database connection after task completes"""
        if self._db is not None:
            self._db.close()
            self._db = None



@celery_app.task(base=DatabaseTask, bind=True, name="src.gpu_server.celery_app.gpt_tasks.process_chat_completion")
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


