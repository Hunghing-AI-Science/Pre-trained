# src/gpu_server/logging_config.py

import logging
from logging.config import dictConfig

def setup_logging():
    """Configure logging for the GPU server and Celery tasks."""
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": "logs/ocr_worker.log",
                "level": "DEBUG",
            },
        },
        "root": {"handlers": ["console", "file"], "level": "DEBUG"},
        "loggers": {
            "celery": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "src.gpu_server": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    })

    # Create directory for logs if not exists
    import os
    os.makedirs("logs", exist_ok=True)