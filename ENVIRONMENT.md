# Environment Configuration Guide

This file explains all environment variables used in the DeepSeek OCR API.

## Quick Setup

```bash
# Copy example to create your .env file
cp .env.example .env

# Edit with your preferred settings
nano .env  # or use your favorite editor
```

## Configuration Sections

### Database Configuration

Controls PostgreSQL database connection and pooling:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/ocr_db
DATABASE_POOL_SIZE=5                # Number of connections to maintain
DATABASE_MAX_OVERFLOW=10            # Max extra connections when pool is full
DATABASE_POOL_TIMEOUT=30            # Seconds to wait for connection
DATABASE_POOL_RECYCLE=3600          # Recycle connections after this many seconds
```

### RabbitMQ Configuration

Message broker for Celery task queue:

```env
RABBITMQ_URL=amqp://guest:guest@localhost:5672//
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VHOST=/
```

**Management UI**: http://localhost:15672

### Celery Configuration

Task queue worker settings:

```env
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=rpc://        # Use RPC for results
CELERY_TASK_SERIALIZER=json
CELERY_ACCEPT_CONTENT=json
CELERY_RESULT_SERIALIZER=json
CELERY_TIMEZONE=UTC
CELERY_ENABLE_UTC=true
CELERY_TASK_TRACK_STARTED=true
CELERY_TASK_TIME_LIMIT=3600         # Max task runtime (1 hour)
CELERY_TASK_SOFT_TIME_LIMIT=3000    # Soft limit before hard kill
CELERY_WORKER_CONCURRENCY=1         # Number of worker processes
CELERY_WORKER_PREFETCH_MULTIPLIER=1 # Tasks to prefetch per worker
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000  # Restart worker after N tasks
```

**Important**: 
- `CELERY_WORKER_CONCURRENCY=1` recommended for GPU workloads
- Model persists for worker lifetime, reload happens only on worker restart

### Celery Flower Configuration

Web-based monitoring tool:

```env
FLOWER_PORT=5555
FLOWER_BROKER_API=http://guest:guest@localhost:15672/api/
FLOWER_PERSISTENT=true
FLOWER_DB=/tmp/flower.db
```

**Access Flower**: http://localhost:5555

### OCR Model Configuration

DeepSeek OCR model settings:

```env
OCR_MODEL_NAME=deepseek-ai/DeepSeek-OCR
OCR_MODEL_CACHE_DIR=~/.cache/huggingface  # Model cache location
OCR_ATTENTION_IMPLEMENTATION=flash_attention_2
OCR_USE_SAFETENSORS=true
OCR_TORCH_DTYPE=bfloat16            # or float16
OCR_DEVICE=cuda                      # or cpu
```

**Model Persistence**:
- Model loads ONCE when Celery worker starts
- Model stays in memory for entire worker lifetime
- No reload between tasks
- Only reloads if worker process restarts

**Supported dtypes**: `bfloat16` (recommended), `float16`, `float32`

### Upload & Storage Configuration

File upload settings:

```env
UPLOAD_DIR=/tmp/ocr_uploads
MAX_UPLOAD_SIZE=52428800            # 50MB in bytes
ALLOWED_EXTENSIONS=png,jpg,jpeg,gif,bmp,tiff,webp,pdf
```

### CUDA Configuration

GPU settings:

```env
CUDA_VISIBLE_DEVICES=0              # GPU device ID (0,1,2... or -1 for CPU)
TORCH_CUDA_ARCH_LIST=               # Optional: specific CUDA architectures
```

**Multiple GPUs**: Set to `0,1,2` to use multiple GPUs (not recommended for single worker)

### API Configuration

FastAPI server settings:

```env
API_HOST=0.0.0.0                    # Listen on all interfaces
API_PORT=8000
API_WORKERS=1                       # Number of API workers
API_RELOAD=true                     # Auto-reload on code changes (dev only)
API_LOG_LEVEL=info                  # debug, info, warning, error
```

### Security Configuration (Optional)

Future authentication features:

```env
# API_KEY=your-secret-api-key-here
# ENABLE_CORS=true
# CORS_ORIGINS=*
```

### Logging Configuration

Logging settings:

```env
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Production Recommendations

For production deployment:

```env
# Database
DATABASE_URL=postgresql://prod_user:strong_password@db.example.com:5432/ocr_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# RabbitMQ
RABBITMQ_URL=amqp://prod_user:strong_password@mq.example.com:5672/prod

# Celery
CELERY_WORKER_CONCURRENCY=1         # Keep at 1 for GPU workloads
CELERY_WORKER_MAX_TASKS_PER_CHILD=100  # Restart more frequently in prod

# API
API_RELOAD=false                    # Disable reload in production
API_WORKERS=4                       # Multiple API workers (not Celery workers!)
API_LOG_LEVEL=warning

# OCR Model
OCR_MODEL_CACHE_DIR=/opt/ml/models  # Persistent storage
OCR_TORCH_DTYPE=bfloat16           # Best quality

# Security
API_KEY=your-production-secret-key
ENABLE_CORS=true
CORS_ORIGINS=https://yourdomain.com

# Logging
LOG_LEVEL=WARNING
```

## Model Persistence Behavior

### How It Works

1. **Worker Startup**: Model loads when Celery worker starts
2. **Task Processing**: Model stays in GPU memory, reused for all tasks
3. **Between Tasks**: Model remains loaded (no reload!)
4. **Worker Shutdown**: Model released when worker stops

### Benefits

- ✅ Fast inference (no model reload overhead)
- ✅ Efficient GPU memory usage
- ✅ Consistent performance across tasks
- ✅ Lower latency after first task

### When Model Reloads

Model reloads ONLY when:
- Worker process restarts
- Worker crashes or is killed
- `CELERY_WORKER_MAX_TASKS_PER_CHILD` limit reached (optional restart)
- Manual worker restart

## Troubleshooting

### Model Not Loading
- Check CUDA availability: `nvidia-smi`
- Verify `OCR_MODEL_CACHE_DIR` has space
- Check internet connection for model download

### Out of Memory
- Reduce `CELERY_WORKER_CONCURRENCY` to 1
- Use `OCR_TORCH_DTYPE=float16` instead of bfloat16
- Ensure only one model instance per GPU

### RabbitMQ Connection Issues
- Verify RabbitMQ is running: `docker-compose ps`
- Check `RABBITMQ_URL` format
- Access management UI: http://localhost:15672

## Environment Variables Priority

Settings are loaded in this order (later overrides earlier):

1. Default values in code
2. `.env` file
3. System environment variables
4. Command-line arguments (where applicable)
