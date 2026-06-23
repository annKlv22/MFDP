"""Воркер с моделью: потребляет задачи из RabbitMQ, прогоняет инференс, пишет результат в БД.

Горизонтальное масштабирование: `docker compose up -d --scale worker=N`.
Несколько воркеров — конкурирующие потребители одной durable-очереди; prefetch=1 + ручной ack
обеспечивают честное распределение и отсутствие потерь задач при падении воркера.
"""
import json
import time
from datetime import datetime
from typing import Optional

import pika
from sqlmodel import select

from common.database import init_db, session_scope
from common.logging_conf import get_logger
from common.models import Prediction, PredictionScore, PredictionStatus, Species
from common.mq import QUEUE_NAME, connection_params
from worker.inference import BirdClassifier

logger = get_logger("worker")
classifier: Optional[BirdClassifier] = None


def process_prediction(prediction_id: str) -> None:
    started = time.time()

    # 1) помечаем PROCESSING и забираем путь к аудио
    with session_scope() as session:
        prediction = session.get(Prediction, prediction_id)
        if prediction is None:
            logger.warning("Запрос %s не найден в БД — пропускаю", prediction_id)
            return
        prediction.status = PredictionStatus.PROCESSING.value
        session.add(prediction)
        audio_path = prediction.audio_path

    # 2) инференс (долгая операция — вне открытой сессии)
    try:
        result = classifier.predict(audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка инференса для %s", prediction_id)
        with session_scope() as session:
            p = session.get(Prediction, prediction_id)
            if p is not None:
                p.status = PredictionStatus.FAILED.value
                p.error = f"Не удалось обработать аудио: {exc}"
                p.finished_at = datetime.utcnow()
                session.add(p)
        return

    # 3) сохраняем результат и оценки по видам
    with session_scope() as session:
        p = session.get(Prediction, prediction_id)
        if p is None:
            return
        for old in list(p.scores):       # на случай повторной обработки
            session.delete(old)
        species_by_name = {s.name: s for s in session.exec(select(Species)).all()}
        for s in result["scores"]:
            sp = species_by_name.get(s["name"])
            session.add(PredictionScore(
                prediction_id=prediction_id,
                species_id=sp.id if sp else None,
                species_name=s["name"],
                probability=s["probability"],
                rank=s["rank"],
            ))
        p.status = PredictionStatus.DONE.value
        p.is_confident = result["is_confident"]
        p.confidence = result["confidence"]
        p.top_species = result["top_species"]
        p.duration_sec = result.get("duration_sec")
        p.processing_ms = int((time.time() - started) * 1000)
        p.finished_at = datetime.utcnow()
        session.add(p)

    logger.info(
        "Готово %s -> %s (%.1f%%, %d сегм.)",
        prediction_id, result["top_species"], result["confidence"] * 100, result["n_segments"],
    )


def on_message(ch, method, _properties, body) -> None:
    try:
        payload = json.loads(body)
        prediction_id = payload["prediction_id"]
        logger.info("Получена задача: %s", prediction_id)
        process_prediction(prediction_id)
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось обработать сообщение")
    finally:
        # ack всегда: ошибки уже зафиксированы статусом FAILED, чтобы не зациклить «отравленное» сообщение
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main() -> None:
    global classifier
    logger.info("Старт воркера. Инициализация схемы БД...")
    _retry(init_db, what="init_db")
    logger.info("Загрузка модели (при первом запуске AST ~346 МБ качается с HuggingFace)...")
    classifier = BirdClassifier()

    while True:
        try:
            connection = pika.BlockingConnection(connection_params())
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)
            logger.info("Ожидаю задачи из очереди '%s'...", QUEUE_NAME)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            logger.warning("Нет связи с RabbitMQ — повтор через 5 с")
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Остановка воркера")
            break
        except Exception:  # noqa: BLE001
            logger.exception("Сбой потребителя — перезапуск через 5 с")
            time.sleep(5)


def _retry(fn, *, what: str, attempts: int = 30, delay: float = 2.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s не удалось (%d/%d): %s", what, i + 1, attempts, exc)
            time.sleep(delay)
    raise RuntimeError(f"{what}: исчерпаны попытки")


if __name__ == "__main__":
    main()
