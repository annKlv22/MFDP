"""Единые настройки сервиса (читаются из переменных окружения / .env).

Один и тот же класс настроек используют и API, и воркер — поэтому он лежит в `common`.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Разрешённые форматы аудио (по ТЗ: WAV, MP3, M4A; ogg/flac — бонусом)
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


class Settings(BaseSettings):
    # --- Приложение ---
    APP_NAME: str = "Определитель птиц по голосу"
    APP_DESCRIPTION: str = "Региональный определитель видов птиц (СПб / Ленобласть) по голосу"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- База данных ---
    DB_HOST: str = "db"
    DB_PORT: int = 5432
    DB_USER: str = "birduser"
    DB_PASS: str = "birdpass"
    DB_NAME: str = "birds"
    # Если задан явно (напр. sqlite:// в тестах) — используется как есть.
    DATABASE_URL: Optional[str] = None

    # --- RabbitMQ ---
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "rmuser"
    RABBITMQ_PASS: str = "rmpassword"
    RABBITMQ_QUEUE: str = "bird_predictions"

    # --- Хранилище и модель ---
    UPLOAD_DIR: str = "/data/uploads"
    MODEL_DIR: str = "/code/worker/models_store"
    AST_MODEL_NAME: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    DEVICE: str = "cpu"

    # --- Параметры инференса (совпадают с ноутбуком) ---
    CONFIDENCE_THRESHOLD: float = 0.4
    SEGMENT_SECONDS: float = 5.0
    HOP_SECONDS: float = 2.5
    MIN_SEGMENT_RMS: float = 0.005
    TARGET_SR: int = 16000
    TOP_K_STORED: int = 5

    # --- Загрузка файлов ---
    MAX_UPLOAD_MB: int = 15

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    @property
    def sqlalchemy_url(self) -> str:
        """DSN для SQLAlchemy. По умолчанию PostgreSQL через psycopg (v3)."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> "Settings":
    return Settings()


settings = get_settings()
