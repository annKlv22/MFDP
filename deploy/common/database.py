"""Подключение к БД, создание таблиц и выдача сессий (sync SQLModel)."""
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import common.models  # noqa: F401  — регистрирует таблицы в SQLModel.metadata
from common.config import settings

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Ленивая инициализация движка (один на процесс)."""
    global _engine
    if _engine is None:
        url = settings.sqlalchemy_url
        if url.startswith("sqlite"):
            _engine = create_engine(
                url,
                echo=False,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            _engine = create_engine(url, echo=False, pool_pre_ping=True)
    return _engine


def set_engine(engine: Engine) -> None:
    """Подмена движка (используется в тестах)."""
    global _engine
    _engine = engine


def init_db() -> None:
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI-зависимость: выдаёт сессию на время запроса."""
    with Session(get_engine()) as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """Контекстная сессия с commit/rollback (для воркера и сидинга)."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
