"""UI-страница и healthcheck."""
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from common.config import settings

APP_DIR = Path(__file__).resolve().parents[1]   # .../app
templates = Jinja2Templates(directory=str(APP_DIR / "view"))

router = APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": settings.APP_NAME, "max_mb": settings.MAX_UPLOAD_MB},
    )


@router.get("/health", tags=["Health"], summary="Проверка работоспособности")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
