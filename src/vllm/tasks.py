import os
import json
import base64
from typing import Any, Dict, Optional, Union

import requests
from celery.utils.log import get_task_logger

from src.vllm.celery_app import celery_app

logger = get_task_logger(__name__)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://172.17.10.38:6379")
VLLM_CHAT_COMPLETIONS_URL = VLLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
VLLM_API_KEY = os.getenv("VLLM_API_KEY")  # optional; only used if your vLLM is behind auth

DEFAULT_MODEL = os.getenv("OCR_MODEL_NAME", "deepseek-ai/DeepSeek-OCR")


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        # OpenAI-compatible convention
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    return headers


def _image_to_data_url(image: Union[str, bytes]) -> str:
    """
    Accepts:
      - base64 string (no prefix), OR
      - raw bytes
    Returns:
      - data URL: data:image/png;base64,...
    """
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode("utf-8")
    else:
        # assume already base64
        b64 = image
    # if you know format, change image/png accordingly
    return f"data:image/png;base64,{b64}"


@celery_app.task(name="vllm.deepseek_ocr_chat", bind=True, autoretry_for=(requests.RequestException,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def deepseek_ocr_vllm_chat(
    self,
    *,
    image_base64: Optional[str] = None,
    prompt: str = "Extract all text from this image. Return plain text.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calls vLLM OpenAI-compatible /v1/chat/completions for OCR (multimodal).

    NOTE: The exact multimodal schema depends on the model + vLLM build.
    This payload follows the common OpenAI-style content blocks format.
    """
    content = [{"type": "text", "text": prompt}]

    if image_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image_base64)},
            }
        )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if extra:
        payload.update(extra)

    logger.info("Calling vLLM chat completions at %s with model=%s", VLLM_CHAT_COMPLETIONS_URL, model)

    resp = requests.post(
        VLLM_CHAT_COMPLETIONS_URL,
        headers=_headers(),
        data=json.dumps(payload),
        timeout=float(os.getenv("VLLM_HTTP_TIMEOUT", "300")),
    )
    resp.raise_for_status()
    return resp.json()