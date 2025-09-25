"""
settings.py
===================================================================
Centralized configuration management for the RAG system
-------------------------------------------------------------------

This module handles environment configuration, application settings,
and logging setup. It uses **Pydantic** models to define and validate
configuration blocks, making the system flexible and maintainable.

Main responsibilities:
- Load environment variables from a `.env` file located at the project root.
- Configure application-wide logging (INFO-level with timestamps).
- Define structured settings for:
  * `LLMSettings`: generic parameters for language models.
  * `OpenAISettings`: API key, default model, embedding model.
  * `DatabaseSettings`: connection URL for Timescale/pgvector.
  * `VectorStoreSettings`: embedding table name, dimension, and partitioning.
- Provide a single entrypoint `get_settings()` that returns a cached
  `Settings` object (ensuring consistent configuration across modules).

Typical usage:
--------------
```python
from app.config.settings import get_settings

settings = get_settings()
print(settings.openai.api_key)
print(settings.database.service_url)
"""

import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Find the .env file relative to this settings.py file
# This will work regardless of where you run the script from
BASE_DIR = Path(__file__).resolve().parent.parent  # Goes up to 'app' directory
ENV_PATH = BASE_DIR / ".env"

# Load the .env file with the correct path
load_dotenv(dotenv_path=ENV_PATH)


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-small")


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()

    # Debug: Print to verify the API key is loaded
    if not settings.openai.api_key:
        logging.error(f"OpenAI API key not found! Looked for .env at: {ENV_PATH}")
        logging.error(f"Does the file exist? {ENV_PATH.exists()}")
        if ENV_PATH.exists():
            logging.error("File exists but OPENAI_API_KEY might be missing or empty")
    else:
        logging.info(f"OpenAI API key loaded successfully (length: {len(settings.openai.api_key)})")

    return settings