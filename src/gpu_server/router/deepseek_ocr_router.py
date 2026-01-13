from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from src.gpu_server.database import get_db, OCRTask
from src.gpu_server.schemas import (
    OCRResponse, OCRChoice, Usage,
    OCRTaskStatus, OCRAsyncResponse
)
from src.gpu_server.celery_app import celery_app  # ✅ Only import celery_app, not tasks
import uuid
import os
import logging
from typing import Optional
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/ocr",
    tags=["DeepSeek OCR"]
)

# Create upload directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/ocr_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/completions", response_model=OCRAsyncResponse)
async def create_ocr_completion(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: str = Form(default="Free OCR.", description="OCR prompt"),
    model: str = Form(default="deepseek-ocr", description="Model identifier"),
    base_size: Optional[int] = Form(default=1024),
    image_size: Optional[int] = Form(default=640),
    crop_mode: Optional[bool] = Form(default=True),
    db: Session = Depends(get_db)
):
    """
    Create an OCR completion task (async)

    This endpoint accepts an image and returns a task ID immediately.
    The actual OCR processing is done asynchronously via Celery.
    """
    # Generate task ID
    task_id = f"ocr_{uuid.uuid4().hex}"

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    image_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_extension}")

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create task in database
    db_task = OCRTask(
        id=task_id,
        status="pending",
        prompt=prompt,
        image_path=image_path
    )
    db.add(db_task)
    db.commit()

    # Enqueue Celery task using send_task (no import of heavy task implementation needed)
    celery_app.send_task(
        "tasks.process_ocr",  # Task name registered in Celery
        args=[task_id, image_path, prompt],
        task_id=task_id
    )

    return OCRAsyncResponse(
        id=task_id,
        status="pending",
        created_at=db_task.created_at
    )


@router.get("/tasks/{task_id}", response_model=OCRTaskStatus)
async def get_ocr_task_status(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the status and result of an OCR task
    """
    task = db.query(OCRTask).filter(OCRTask.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response_data = {
        "id": task.id,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "error": task.error
    }

    # If task is completed, format result in OpenAI style
    if task.status == "completed" and task.result:
        created_timestamp = int(task.created_at.timestamp())

        response_data["result"] = OCRResponse(
            id=task.id,
            created=created_timestamp,
            model="deepseek-ocr",
            choices=[
                OCRChoice(
                    index=0,
                    text=task.result.get("text", ""),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=task.result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=task.result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=task.result.get("usage", {}).get("total_tokens", 0)
            )
        )

    return OCRTaskStatus(**response_data)


@router.delete("/tasks/{task_id}")
async def delete_ocr_task(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete an OCR task and its associated files
    """
    task = db.query(OCRTask).filter(OCRTask.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Delete associated image file if exists
    if task.image_path and os.path.exists(task.image_path):
        try:
            os.remove(task.image_path)
        except Exception as e:
            logger.warning(f"Failed to delete file {task.image_path}: {str(e)}")

    # Delete task from database
    db.delete(task)
    db.commit()

    return {"status": "deleted", "id": task_id}

