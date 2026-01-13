from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/ocr_db")
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))

engine = create_engine(
    DATABASE_URL,
    pool_size=DATABASE_POOL_SIZE,
    max_overflow=DATABASE_MAX_OVERFLOW,
    pool_timeout=DATABASE_POOL_TIMEOUT,
    pool_recycle=DATABASE_POOL_RECYCLE,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class OCRTask(Base):
    __tablename__ = "ocr_tasks"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    prompt = Column(Text)
    image_path = Column(String)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    model_name = Column(String, default="deepseek-ocr")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class GPTTask(Base):
    __tablename__ = "gpt_tasks"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    messages = Column(JSON)  # List of chat messages
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    model_name = Column(String, default="openai/gpt-oss-20b")
    temperature = Column(JSON, nullable=True)  # Store as float in JSON
    max_tokens = Column(JSON, nullable=True)  # Store as int in JSON
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)

