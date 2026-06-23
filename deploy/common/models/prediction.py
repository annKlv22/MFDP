"""Сущности «Запрос на определение» и «Оценка по виду».

Prediction (1) ──< PredictionScore (N): одна запись пользователя → топ-N вероятностей по видам.
Статус хранится строкой (портируемо между SQLite и PostgreSQL, без native ENUM).
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class PredictionStatus(str, Enum):
    PENDING = "PENDING"        # принято, ждёт воркера
    PROCESSING = "PROCESSING"  # воркер обрабатывает
    DONE = "DONE"              # готово
    FAILED = "FAILED"          # ошибка обработки


class Prediction(SQLModel, table=True):
    __tablename__ = "prediction"

    id: str = Field(primary_key=True)                       # uuid4 hex
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    # Загруженный файл
    filename: str = ""
    audio_path: str = ""
    content_type: str = ""
    size_bytes: int = 0
    duration_sec: Optional[float] = None

    # Состояние и результат
    status: str = Field(default=PredictionStatus.PENDING.value, index=True)
    error: Optional[str] = None
    is_confident: bool = False
    confidence: Optional[float] = None           # max вероятность по видам
    top_species: Optional[str] = None            # имя наиболее вероятного вида
    processing_ms: Optional[int] = None

    scores: List["PredictionScore"] = Relationship(
        back_populates="prediction",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "PredictionScore.rank",
        },
    )


class PredictionScore(SQLModel, table=True):
    __tablename__ = "prediction_score"

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: str = Field(foreign_key="prediction.id", index=True)
    species_id: Optional[int] = Field(default=None, foreign_key="species.id")
    species_name: str = ""
    probability: float = 0.0
    rank: int = 0                                # 1 = самый вероятный

    prediction: Optional[Prediction] = Relationship(back_populates="scores")
