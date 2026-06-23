"""Pydantic-схемы REST-ответов."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SpeciesOut(BaseModel):
    id: int
    name: str
    latin: str
    description: str
    photo_url: str


class ScoreOut(BaseModel):
    rank: int
    name: str
    latin: str = ""
    photo_url: str = ""
    description: str = ""
    probability: float
    percent: float


class PredictionCreatedOut(BaseModel):
    id: str
    status: str


class PredictionOut(BaseModel):
    id: str
    status: str
    created_at: datetime
    finished_at: Optional[datetime] = None
    filename: str
    is_confident: bool = False
    confidence: Optional[float] = None
    confidence_percent: Optional[float] = None
    top_species: Optional[str] = None
    duration_sec: Optional[float] = None
    processing_ms: Optional[int] = None
    error: Optional[str] = None
    scores: List[ScoreOut] = []
