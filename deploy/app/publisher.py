"""Издатель задач в RabbitMQ (producer).

Подключается на каждую публикацию — просто и устойчиво к разрывам для нагрузки MVP.
Сообщение и очередь — durable, чтобы задачи переживали перезапуск брокера.
"""
import json

import pika

from common.logging_conf import get_logger
from common.mq import QUEUE_NAME, connection_params

logger = get_logger("publisher")


class Publisher:
    def publish(self, prediction_id: str) -> None:
        connection = pika.BlockingConnection(connection_params())
        try:
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=QUEUE_NAME,
                body=json.dumps({"prediction_id": prediction_id}).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
            )
            logger.info("Задача опубликована в очередь '%s': %s", QUEUE_NAME, prediction_id)
        finally:
            connection.close()


_publisher = Publisher()


def get_publisher() -> Publisher:
    """FastAPI-зависимость (в тестах подменяется фейком)."""
    return _publisher
