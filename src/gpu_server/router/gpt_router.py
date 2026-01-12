from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from src.gpu_server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage
)
import uuid
import time

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

    ### NO API KEY TEMPORARILY
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

    # TODO: Process the actual GPT request
    # This is just the interface - actual processing should be implemented
    # You would typically:
    # 1. Call the GPT model service with request parameters
    # 2. Handle streaming if request.stream is True
    # 3. Calculate token usage
    # 4. Format and return the response

    # Placeholder response structure
    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=current_timestamp,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content="[Response placeholder - implement actual GPT processing]"
                ),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=0,  # Calculate based on input
            completion_tokens=0,  # Calculate based on output
            total_tokens=0
        )
    )

    return response


