from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import logging
import os
from src.gpu_server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage
)
from src.gpt_oss.gpt_oss_service import get_gpt_service
import uuid
import time

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/chat",
    tags=["GPT Chat Completions"]
)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
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

    Authorization: Bearer token required in headers
    """
    # Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Expected format: 'Bearer YOUR_API_KEY'"
        )

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

    # Generate unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    current_timestamp = int(time.time())

    logger.info(f"Processing chat completion request {completion_id} with model {request.model}")

    try:
        # Get GPT service instance for the requested model
        # Map OpenAI model names to actual model identifiers
        model_map = {
            'gpt-3.5-turbo': os.getenv('GPT_35_MODEL', 'openai/gpt-oss-20b'),
            'gpt-4': os.getenv('GPT_4_MODEL', 'openai/gpt-oss-120b'),
            'openai/gpt-oss-20b': 'openai/gpt-oss-20b',
            'openai/gpt-oss-120b': 'openai/gpt-oss-120b',
        }

        actual_model = model_map.get(request.model, request.model)
        gpt_service = get_gpt_service(actual_model)

        # Convert Pydantic messages to dict format
        messages_dict = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # Generate completion using GPT service
        result = gpt_service.generate_chat_completion(
            messages=messages_dict,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            n=request.n or 1
        )

        # Handle multiple completions if n > 1
        if request.n and request.n > 1:
            choices = [
                ChatCompletionChoice(
                    index=i,
                    message=ChatMessage(
                        role="assistant",
                        content=text
                    ),
                    finish_reason="stop"
                )
                for i, text in enumerate(result["text"])
            ]
        else:
            choices = [
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result["text"]
                    ),
                    finish_reason="stop"
                )
            ]

        # Create response
        response = ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=current_timestamp,
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"]
            )
        )

        logger.info(f"Chat completion {completion_id} completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error processing chat completion {completion_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate completion: {str(e)}"
        )



