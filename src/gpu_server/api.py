from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.gpu_server.database import init_db
from src.gpu_server.router import deepseek_ocr_router, gpt_router
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    init_db()
    logger.info("Database initialized")
    yield
    # Cleanup code can go here if needed
    logger.info("Shutting down...")


app = FastAPI(
    title="GPU Server API",
    description="Multi-model API supporting DeepSeek OCR and GPT models following OpenAI architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(deepseek_ocr_router.router)
app.include_router(gpt_router.router)




@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "GPU Server API",
        "version": "1.0.0",
        "endpoints": {
            "ocr": "/v1/ocr",
            "chat": "/v1/chat",
            "models": "/v1/models",
            "docs": "/docs"
        }
    }


@app.get("/v1/models")
async def list_models():
    """
    List available models (OpenAI-style endpoint)
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-ocr",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek-ai",
                "type": "ocr"
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
                "type": "chat"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882410,
                "owned_by": "openai",
                "type": "chat"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import os
    import dotenv

    dotenv.load_dotenv()
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("API_LOG_LEVEL", "info").lower()

    # SSL/HTTPS configuration (optional)
    ssl_keyfile = os.getenv("SSL_KEYFILE", None)
    ssl_certfile = os.getenv("SSL_CERTFILE", None)

    logger.info(f"Starting FastAPI server on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}, Log Level: {log_level}")

    if ssl_keyfile and ssl_certfile:
        logger.info(f"HTTPS enabled with cert: {ssl_certfile}")
    else:
        logger.info("Running in HTTP mode (no SSL certificates provided)")

    uvicorn_config = {
        "app": "src.gpu_server.api:app",
        "host": host,
        "port": port,
        "workers": workers if not reload else 1,
        "reload": reload,
        "log_level": log_level,
    }

    # Add SSL configuration if certificates are provided
    if ssl_keyfile and ssl_certfile:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        uvicorn_config["ssl_certfile"] = ssl_certfile

    uvicorn.run(**uvicorn_config)




