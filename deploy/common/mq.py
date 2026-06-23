"""Параметры подключения к RabbitMQ. Общие для издателя (API) и потребителя (воркер)."""
import pika

from common.config import settings

QUEUE_NAME = settings.RABBITMQ_QUEUE


def connection_params() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        virtual_host="/",
        credentials=pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASS),
        heartbeat=600,
        blocked_connection_timeout=300,
    )
