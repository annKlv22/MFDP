"""REST: создание запроса на определение и получение результата.

Поток: загрузка файла -> сохранение -> запись PENDING в БД -> публикация задачи в RabbitMQ.
Воркер асинхронно обрабатывает и обновляет запись. UI опрашивает GET /{id}.
"""
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.publisher import Publisher, get_publisher
from app.schemas import PredictionCreatedOut, PredictionOut, ScoreOut
from common.config import ALLOWED_AUDIO_EXTS, settings
from common.database import get_session
from common.logging_conf import get_logger
from common.models import Prediction, PredictionStatus, Species

logger = get_logger("api.prediction")
router = APIRouter()


def serialize(prediction: Prediction, session: Session) -> PredictionOut:
    species_by_name = {s.name: s for s in session.exec(select(Species)).all()}
    scores = []
    for sc in sorted(prediction.scores, key=lambda x: x.rank):
        sp = species_by_name.get(sc.species_name)
        scores.append(ScoreOut(
            rank=sc.rank,
            name=sc.species_name,
            latin=sp.latin if sp else "",
            photo_url=sp.photo_url if sp else "",
            description=sp.description if sp else "",
            probability=sc.probability,
            percent=round(sc.probability * 100, 1),
        ))
    return PredictionOut(
        id=prediction.id,
        status=prediction.status,
        created_at=prediction.created_at,
        finished_at=prediction.finished_at,
        filename=prediction.filename,
        is_confident=prediction.is_confident,
        confidence=prediction.confidence,
        confidence_percent=round(prediction.confidence * 100, 1) if prediction.confidence is not None else None,
        top_species=prediction.top_species,
        duration_sec=prediction.duration_sec,
        processing_ms=prediction.processing_ms,
        error=prediction.error,
        scores=scores,
    )


@router.post("/", response_model=PredictionCreatedOut, status_code=status.HTTP_201_CREATED,
             summary="Загрузить аудио и поставить в очередь на определение")
async def create_prediction(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    publisher: Publisher = Depends(get_publisher),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый формат '{ext}'. Разрешены: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Пустой файл")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл больше {settings.MAX_UPLOAD_MB} МБ",
        )

    pred_id = uuid.uuid4().hex
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    audio_path = os.path.join(settings.UPLOAD_DIR, f"{pred_id}{ext}")
    with open(audio_path, "wb") as f:
        f.write(data)

    prediction = Prediction(
        id=pred_id,
        filename=file.filename or f"{pred_id}{ext}",
        audio_path=audio_path,
        content_type=file.content_type or "",
        size_bytes=len(data),
        status=PredictionStatus.PENDING.value,
    )
    session.add(prediction)
    session.commit()

    try:
        publisher.publish(pred_id)
    except Exception as exc:  # брокер недоступен — фиксируем ошибку, но не теряем запись
        logger.error("Не удалось опубликовать задачу %s: %s", pred_id, exc)
        prediction.status = PredictionStatus.FAILED.value
        prediction.error = "Сервис обработки временно недоступен, попробуйте позже."
        session.add(prediction)
        session.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=prediction.error)

    return PredictionCreatedOut(id=pred_id, status=prediction.status)


@router.get("/{prediction_id}", response_model=PredictionOut, summary="Статус и результат определения")
def get_prediction(prediction_id: str, session: Session = Depends(get_session)):
    prediction = session.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Запрос не найден")
    return serialize(prediction, session)


@router.get("/", response_model=List[PredictionOut], summary="История последних определений")
def list_predictions(limit: int = 20, session: Session = Depends(get_session)):
    rows = session.exec(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    ).all()
    return [serialize(p, session) for p in rows]
