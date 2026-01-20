from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from src.gpu_server.database import get_db, OCRTask
from src.gpu_server.schemas import (
    OCRResponse, OCRChoice, Usage,
    OCRTaskStatus,
    OCRCompletionRequest
)
from src.gpu_server.celery_app.celery_app import celery_app
import uuid
import os
import logging
import base64
import re
import httpx
import asyncio
import time

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/ocr",
    tags=["DeepSeek OCR"]
)

# Create upload directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/ocr_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_image_from_data_uri(data_uri: str) -> tuple[bytes, str]:
    """
    Extract image bytes from base64 data URI

    Args:
        data_uri: Data URI like "data:image/jpeg;base64,/9j/4AAQ..."

    Returns:
        Tuple of (image_bytes, file_extension)
    """
    # Parse data URI: data:image/jpeg;base64,<data>
    match = re.match(r'data:image/(\w+);base64,(.+)', data_uri)
    if not match:
        raise ValueError("Invalid data URI format. Expected: data:image/{type};base64,{data}")

    image_format = match.group(1)
    base64_data = match.group(2)

    # Decode base64
    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {str(e)}")

    # Map format to extension
    format_map = {
        'jpeg': '.jpg',
        'jpg': '.jpg',
        'png': '.png',
        'gif': '.gif',
        'bmp': '.bmp',
        'webp': '.webp'
    }

    file_extension = format_map.get(image_format.lower(), '.jpg')

    return image_bytes, file_extension

@router.post("/chat/completions", response_model=OCRResponse)
async def create_ocr_chat_completion(
    request: OCRCompletionRequest = Body(...),
    db: Session = Depends(get_db)
):
    """
    Create an OCR completion task (Simplified format)

    This endpoint accepts JSON requests with base64 encoded images:

    Example:
    ```json
    {
      "model": "deepseek-ocr",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Extract all text from this image"
            },
            {
              "type": "image",
              "image": "data:image/jpeg;base64,/9j/4AAQ..."
            }
          ]
        }
      ]
    }
    ```

    Returns the final OCR result after internally polling for completion. The task ID
    is preserved in the response for traceability.
    """
    # Extract prompt and image from messages
    prompt = "Free OCR."
    image_data = None

    for message in request.messages:
        if isinstance(message.content, str):
            # Simple text message
            prompt = message.content
        elif isinstance(message.content, list):
            # Multimodal content
            for part in message.content:
                if part.type == "text" and part.text:
                    prompt = part.text
                elif part.type == "image" and part.image:
                    image_data = part.image

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Include an 'image' content part with base64 data."
        )

    # Only accept base64 data URIs
    if not image_data.startswith("data:image/"):
        raise HTTPException(
            status_code=400,
            detail="Only base64 encoded images are supported. Use format: data:image/jpeg;base64,..."
        )

    # Generate task ID
    task_id = f"ocr_{uuid.uuid4().hex}"

    try:
        # Extract base64 image
        logger.info(f"Processing base64 encoded image")
        image_bytes, file_extension = extract_image_from_data_uri(image_data)

        image_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_extension}")

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Saved base64 image to {image_path} ({len(image_bytes)} bytes)")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    # Create task in database
    db_task = OCRTask(
        id=task_id,
        status="pending",
        prompt=prompt,
        image_path=image_path
    )
    db.add(db_task)
    db.commit()

    # Enqueue Celery task
    celery_app.send_task(
        "tasks.process_ocr",
        args=[task_id, image_path, prompt],
        task_id=task_id
    )

    logger.info(f"OCR task {task_id} enqueued, polling for result...")

    # Poll the task status internally until completion, similar to GPT router behavior
    poll_interval = 1.0  # seconds
    max_wait_time = 300  # 5 minutes timeout
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"Task {task_id} timed out after {max_wait_time}s")
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timed out after {max_wait_time} seconds"
                )

            db.refresh(db_task)
            task_status = db_task.status

            if task_status == "completed":
                logger.info(f"Task {task_id} completed in {elapsed:.1f}s")

                result_text = db_task.result.get("text", "") if db_task.result else ""
                usage_data = db_task.result.get("usage", {}) if db_task.result else {}

                return OCRResponse(
                    id=task_id,
                    object="ocr.completion",
                    created=int(db_task.created_at.timestamp()),
                    model="deepseek-ocr",
                    choices=[
                        OCRChoice(
                            index=0,
                            text=result_text,
                            finish_reason="stop"
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0)
                    )
                )

            if task_status == "failed":
                error_msg = db_task.error or "Unknown error"
                logger.error(f"Task {task_id} failed: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Task failed: {error_msg}"
                )

            await asyncio.sleep(poll_interval)


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
