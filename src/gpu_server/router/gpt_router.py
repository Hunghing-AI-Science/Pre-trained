from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session
from typing import Optional
import logging
import os
import httpx
import asyncio
import time
from src.gpu_server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage
)
from src.gpu_server.database import get_db, GPTTask
from src.gpu_server.celery_app.celery_app import celery_app  # ✅ Only import celery_app, not tasks
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/chat",
    tags=["GPT Chat Completions"]
)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Create a chat completion using GPT model.

    This endpoint accepts requests compatible with OhMyGPT chatbot interface:
    - model: The model to use (e.g., "gpt-3.5-turbo", "gpt-4")
    - messages: List of chat messages with role and content
    - temperature: Controls randomness (0.0 to 2.0)
    - top_p: Nucleus sampling parameter
    - max_tokens: Maximum tokens to generate
    - stream: Whether to stream the response

    The request is processed asynchronously via Celery, then polled internally
    using httpx.AsyncClient until complete. Returns the final result directly.

    Authorization: Bearer token required in headers
    """
    raw_body = await request.body()
    print("\n📩 Raw incoming body:", raw_body.decode())

    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)

    # ✅ Proceed to validate with your model
    try:
        request = ChatCompletionRequest(**data)
    except ValidationError as e:
        print("\n❌ Validation error details:", e.errors())
        return JSONResponse({"error": e.errors()}, status_code=422)
    logger.info("Creating chat completion")
    # Validate API key
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(
    #         status_code=401,
    #         detail="Missing or invalid API key. Expected format: 'Bearer YOUR_API_KEY'"
    #     )

    ### NO API KEY VALIDATION TEMPORARILY
    # Extract API key
    # api_key = authorization.replace("Bearer ", "")
    #
    # if not api_key:
    #     raise HTTPException(
    #         status_code=401,
    #         detail="API key is missing"
    #     )

    # Validate request
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Messages list cannot be empty"
        )

    # Generate unique task ID
    task_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info(f"Creating chat completion task {task_id} with model {request.model}")

    # Map OpenAI model names to actual model identifiers
    model_map = {
        'openai/gpt-oss-20b': 'openai/gpt-oss-20b',
        'openai/gpt-oss-120b': 'openai/gpt-oss-120b',
    }

    actual_model = model_map.get(request.model, request.model)

    # Convert Pydantic messages to dict format for storage
    messages_dict = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]

    # Create task in database
    db_task = GPTTask(
        id=task_id,
        status="pending",
        messages=messages_dict,
        model_name=actual_model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    db.add(db_task)
    db.commit()
    TASK_NAME = "src.gpu_server.celery_app.gpt_tasks.process_chat_completion"

    logger.info("Task created in database with ID: %s", task_id)
    # Enqueue Celery task using send_task (no import of heavy task implementation needed)
    celery_app.send_task(
        TASK_NAME,  # Task name registered in Celery
        kwargs={
            'task_id': task_id,
            'messages': messages_dict,
            'model_name': actual_model,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'max_tokens': request.max_tokens,
            'stop': request.stop,
            'presence_penalty': request.presence_penalty,
            'frequency_penalty': request.frequency_penalty,
            'n': request.n or 1
        },
        task_id=task_id
    )

    logger.info(f"Chat completion task {task_id} enqueued, polling for result...")

    # Poll the task status internally using httpx until completion
    poll_interval = 1.0  # seconds
    max_wait_time = 600  # 10 minutes timeout
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

            # Query database for task status
            db.refresh(db_task)
            task_status = db_task.status

            if task_status == "completed":
                logger.info(f"Task {task_id} completed in {elapsed:.1f}s")

                # Extract result
                result_text = db_task.result.get("text", "")

                # Handle multiple completions if it's a list
                if isinstance(result_text, list):
                    choices = [
                        ChatCompletionChoice(
                            index=i,
                            message=ChatMessage(
                                role="assistant",
                                content=text
                            ),
                            finish_reason="stop"
                        )
                        for i, text in enumerate(result_text)
                    ]
                else:
                    choices = [
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=result_text
                            ),
                            finish_reason="stop"
                        )
                    ]

                # Return completed response
                return ChatCompletionResponse(
                    id=task_id,
                    object="chat.completion",
                    created=int(db_task.created_at.timestamp()),
                    model=request.model,
                    choices=choices,
                    usage=Usage(
                        prompt_tokens=db_task.result.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=db_task.result.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=db_task.result.get("usage", {}).get("total_tokens", 0)
                    )
                )

            elif task_status == "failed":
                error_msg = db_task.error or "Unknown error"
                logger.error(f"Task {task_id} failed: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Task failed: {error_msg}"
                )

            # Wait before next poll (non-blocking async sleep)
            await asyncio.sleep(poll_interval)


