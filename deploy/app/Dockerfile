# API-шлюз: FastAPI + UI. Контекст сборки = каталог deploy/ (чтобы скопировать общий пакет common/).
FROM python:3.12-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/code

COPY app/requirements.txt /code/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /code/app/requirements.txt

COPY common /code/common
COPY app /code/app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
