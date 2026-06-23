"""Доменные сущности сервиса (SQLModel)."""
from common.models.prediction import Prediction, PredictionScore, PredictionStatus
from common.models.species import Species

__all__ = ["Species", "Prediction", "PredictionScore", "PredictionStatus"]
