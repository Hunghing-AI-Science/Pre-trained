#!/usr/bin/env python3
"""
Flexible Database Initialization & Auto-Schema Synchronization Script
- Creates the database if missing
- Dynamically loads all SQLAlchemy models
- Creates missing tables and columns automatically
- Verifies setup

Usage:
    python init_db.py
"""

import os
import sys
import logging
import importlib
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import inspect, text, String, Integer, Float, Boolean, DateTime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def parse_database_url(db_url):
    """Parse PostgreSQL connection string components."""
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    return {
        "user": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/")
    }


def check_database_exists(conn_params):
    """Check if the database exists."""
    conn = psycopg2.connect(
        host=conn_params["host"],
        port=conn_params["port"],
        user=conn_params["user"],
        password=conn_params["password"],
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (conn_params["database"],))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists


def create_database(conn_params):
    """Create the database if not exists."""
    logger.info(f"Creating database '{conn_params['database']}'...")
    conn = psycopg2.connect(
        host=conn_params["host"],
        port=conn_params["port"],
        user=conn_params["user"],
        password=conn_params["password"],
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE {conn_params['database']}")
    cur.close()
    conn.close()
    logger.info(f"Database '{conn_params['database']}' created successfully.")


def import_database_module(module_path="src.gpu_server.database"):
    """Import SQLAlchemy Base and engine dynamically."""
    module = importlib.import_module(module_path)
    Base = getattr(module, "Base")
    engine = getattr(module, "engine")
    return Base, engine


def map_sqlalchemy_type_to_postgres(column_type):
    """Translate SQLAlchemy column types to PostgreSQL types."""
    if isinstance(column_type, String):
        return f"VARCHAR({column_type.length or 255})"
    if isinstance(column_type, Integer):
        return "INTEGER"
    if isinstance(column_type, Float):
        return "DOUBLE PRECISION"
    if isinstance(column_type, Boolean):
        return "BOOLEAN"
    if isinstance(column_type, DateTime):
        return "TIMESTAMP"
    return "TEXT"  # fallback


def add_missing_columns(engine, Base):
    """Detect and auto-add missing columns to PostgreSQL tables."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    with engine.connect() as conn:
        for table_name, table_obj in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue

            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            for column in table_obj.columns:
                if column.name not in existing_columns:
                    col_type = map_sqlalchemy_type_to_postgres(column.type)
                    alter_sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {col_type}'
                    logger.warning(f"⚠️ Adding missing column: {table_name}.{column.name} ({col_type})")
                    conn.execute(text(alter_sql))
                    logger.info(f"✅ Column '{column.name}' added to '{table_name}'.")


def sync_schema(Base, engine):
    """Create tables and sync missing columns automatically."""
    logger.info("🔍 Checking tables and columns...")
    Base.metadata.create_all(bind=engine, checkfirst=True)
    add_missing_columns(engine, Base)
    logger.info("✅ Schema synchronization complete.")


def verify_setup(engine):
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).fetchone()[0]
        logger.info(f"PostgreSQL version: {version}")
        tables = inspect(engine).get_table_names()
        logger.info(f"Tables detected: {tables}")


def main(module_path="src.gpu_server.database"):
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL missing in environment variables.")

        logger.info(f"Database URL: {database_url}")
        conn_params = parse_database_url(database_url)

        if not check_database_exists(conn_params):
            create_database(conn_params)
        else:
            logger.info(f"Database '{conn_params['database']}' already exists.")

        Base, engine = import_database_module(module_path)
        sync_schema(Base, engine)
        verify_setup(engine)

        logger.info("🎉 Database initialization & auto-schema sync completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())