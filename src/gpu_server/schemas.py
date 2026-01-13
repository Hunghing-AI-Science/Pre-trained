from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class OCRRequest(BaseModel):
    """Request schema following OpenAI-style API"""
    prompt: str = Field(
        default="Free OCR.",
        description="The prompt to guide the OCR process"
    )
    model: str = Field(
        default="deepseek-ocr",
        description="Model identifier"
    )
    # Additional parameters
    base_size: Optional[int] = Field(default=1024, description="Base size for processing")
    image_size: Optional[int] = Field(default=640, description="Image size for processing")
    crop_mode: Optional[bool] = Field(default=True, description="Enable crop mode")


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OCRChoice(BaseModel):
    """OCR result choice"""
    index: int = 0
    text: str
    finish_reason: str = "stop"


class OCRResponse(BaseModel):
    """Response schema following OpenAI-style API"""
    id: str
    object: str = "ocr.completion"
    created: int
    model: str
    choices: list[OCRChoice]
    usage: Usage


class OCRTaskStatus(BaseModel):
    """Task status response"""
    id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    updated_at: datetime
    result: Optional[OCRResponse] = None
    error: Optional[str] = None


class OCRAsyncResponse(BaseModel):
    """Async task creation response"""
    id: str
    object: str = "ocr.task"
    status: str
    created_at: datetime


# GPT Chat Completion Schemas
class ChatMessage(BaseModel):
    """Chat message schema"""
    role: str  # system, user, assistant
    content: str


class ChatCompletionRequest(BaseModel):
    """Request schema for GPT chat completion"""
    model: str = Field(default="gpt-3.5-turbo", description="Model to use")
    messages: list[ChatMessage] = Field(default=[], description="List of messages")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Stream the response")
    n: Optional[int] = Field(default=1, description="Number of completions to generate")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)


class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"  # stop, length, content_filter, null


class ChatCompletionResponse(BaseModel):
    """Response schema for GPT chat completion"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionAsyncResponse(BaseModel):
    """Async task creation response for chat completion"""
    id: str
    object: str = "chat.completion.task"
    status: str
    model: str
    created_at: datetime


class ChatCompletionTaskStatus(BaseModel):
    """Task status response for chat completion"""
    id: str
    status: str  # pending, processing, completed, failed
    model: str
    created_at: datetime
    updated_at: datetime
    result: Optional[ChatCompletionResponse] = None
    error: Optional[str] = None


