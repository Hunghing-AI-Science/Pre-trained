# Database Setup Guide

## Overview

This project uses PostgreSQL with SQLAlchemy ORM. The database initialization is automated.

## Quick Start

### Initialize Database

```bash
make init-db
# Or: python init_database.py
```

### Configuration

Configure database settings in the `.env` file:

```env
DATABASE_URL=postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE_NAME
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
```

Replace `USERNAME`, `PASSWORD`, `HOST`, `PORT`, and `DATABASE_NAME` with your actual credentials.

## Database Schema

### OCRTask Table

| Column      | Type     | Description                              |
|-------------|----------|------------------------------------------|
| id          | String   | Primary key, unique task identifier      |
| status      | String   | pending/processing/completed/failed      |
| prompt      | Text     | OCR prompt text                          |
| image_path  | String   | Path to uploaded image                   |
| result      | JSON     | OCR result with text and usage info      |
| error       | Text     | Error message if task failed             |
| created_at  | DateTime | Task creation timestamp (UTC)            |
| updated_at  | DateTime | Last update timestamp (UTC)              |

## Manual Database Operations

### Connect to PostgreSQL

```bash
# Using psql
psql -h HOST -U USERNAME -d DATABASE_NAME

# List all tables
\dt

# Describe a table
\d ocr_tasks

# Query tasks
SELECT id, status, created_at FROM ocr_tasks;
```

### Create Database Manually

```sql
-- Connect to postgres database first
psql -h HOST -U USERNAME -d postgres

-- Create database
CREATE DATABASE ocr_db;

-- Grant permissions (if needed)
GRANT ALL PRIVILEGES ON DATABASE ocr_db TO USERNAME;
```

## Troubleshooting

### Error: database "ocr_db" does not exist

**Solution:** Run the initialization script:
```bash
python init_database.py
```

### Error: FATAL: role "postgres" does not exist

**Solution:** Update the `DATABASE_URL` in `.env` with the correct username.

### Error: connection refused

**Solution:** 
1. Check if PostgreSQL is running
2. Verify the host and port in `DATABASE_URL`
3. Check firewall settings

### Error: permission denied to create database

**Solution:** The database user needs `CREATEDB` privilege:
```sql
ALTER USER USERNAME CREATEDB;
```

## Using Alembic for Migrations

For more advanced schema management, you can use Alembic:

### Initialize Alembic

```bash
python setup_alembic.py
```

### Create a Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Edit Alembic Configuration

Edit `alembic/env.py` to use your models:

```python
from src.gpu_server.database import Base
target_metadata = Base.metadata
```

## Schema Updates

When you modify the SQLAlchemy models (e.g., `src/gpu_server/database.py`):

### Option 1: Simple Update (Development)

Drop and recreate tables:
```bash
python init_database.py --force-recreate
```

⚠️ **Warning:** This will delete all existing data!

### Option 2: Using Alembic (Production)

Create and apply migrations:
```bash
alembic revision --autogenerate -m "Add new column"
alembic upgrade head
```

