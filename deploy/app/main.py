"""Точка входа FastAPI: REST + UI. Авторизация для конечного пользователя не требуется"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import home, prediction, species
from app.seed import seed_species
from common.config import settings
from common.database import init_db
from common.logging_conf import get_logger

logger = get_logger("api")
APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Инициализация базы данных и справочника видов...")
    init_db()
    seed_species()
    logger.info("API запущен и готов принимать запросы")
    yield
    logger.info("Остановка API")


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
    application.include_router(home.router, tags=["UI"])
    application.include_router(species.router, prefix="/api/species", tags=["Species"])
    application.include_router(prediction.router, prefix="/api/predictions", tags=["Predictions"])
    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=settings.DEBUG)
