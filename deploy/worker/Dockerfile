# Воркер с ML-моделью. Контекст сборки = каталог deploy/ (для общего пакета common/).
FROM python:3.12-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/code \
    HF_HOME=/cache/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# ffmpeg + libsndfile нужны librosa/soundfile для декодирования mp3/m4a/wav
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-сборка torch без CUDA-зависимостей (компактнее и достаточно для инференса)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY worker/requirements.txt /code/worker/requirements.txt
RUN pip install --no-cache-dir -r /code/worker/requirements.txt

COPY common /code/common
COPY worker /code/worker

CMD ["python", "-m", "worker.worker"]
