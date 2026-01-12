"""
Example: How to integrate the GPT router into your FastAPI application
"""

from fastapi import FastAPI
from src.gpu_server.router.gpt_router import router as gpt_router

app = FastAPI(
    title="GPU Server API",
    description="API with GPT Chat Completions",
    version="1.0.0"
)

# Include the GPT router
app.include_router(gpt_router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "GPU Server API",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/chat/models"
        }
    }


# To use this with OhMyGPT client:
#
# class OhMyGPT(Chatbot_Interface):
#     def __init__(self, arg):
#         self.url = "http://your-server:8000/v1/chat/completions"
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": "Bearer " + arg["key"],
#         }
#         self.model = arg.get('model', 'gpt-3.5-turbo')
#         self.arg = arg
#         # ... rest of initialization
#
# The router will accept:
# - model: str (e.g., "gpt-3.5-turbo", "gpt-4")
# - messages: list of {role: str, content: str}
# - temperature: float (0.0 to 2.0, default 0.7)
# - top_p: float (0.0 to 1.0, default 0.95)
# - max_tokens: int (optional)
# - stream: bool (default False)
#
# Response format:
# {
#   "id": "chatcmpl-...",
#   "object": "chat.completion",
#   "created": timestamp,
#   "model": "gpt-3.5-turbo",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "..."
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 10,
#     "completion_tokens": 20,
#     "total_tokens": 30
#   }
# }

