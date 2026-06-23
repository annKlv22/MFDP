"""Общие фикстуры тестов: изолированная БД SQLite в памяти + фейковый издатель очереди.

Тяжёлые ML-зависимости не нужны: логика инференса тестируется на чистых функциях,
а API — с подменённым publisher (RabbitMQ не поднимается).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # гарантируем deploy/ в sys.path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from common import database


@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database.set_engine(engine)
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)
    database.set_engine(None)


@pytest.fixture(name="client")
def client_fixture(engine):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.publisher import get_publisher

    class FakePublisher:
        def __init__(self):
            self.published = []

        def publish(self, prediction_id: str):
            self.published.append(prediction_id)

    fake = FakePublisher()
    app.dependency_overrides[get_publisher] = lambda: fake

    with TestClient(app) as test_client:   # запускает lifespan: init_db + seed
        test_client.fake_publisher = fake
        yield test_client

    app.dependency_overrides.clear()
