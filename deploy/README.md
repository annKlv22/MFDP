# 🐦 Определитель птиц по голосу — MVP

Веб-сервис, который по короткой аудиозаписи определяет вид птицы и показывает вероятность.
Региональный фокус — **топ-20 видов Санкт-Петербурга и Ленинградской области**. Без регистрации.

Модель: эмбеддинги **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) + лёгкая голова
(`StandardScaler` + `LogisticRegression`, обучена на региональных данных). Предсказание по записи =
усреднение вероятностей по 5-секундным сегментам + порог уверенности.

---

## Возможности

- 📤 Загрузка готового аудиофайла (WAV / MP3 / M4A, до 15 МБ) — ключевая фича, которой нет у Merlin/BirdNET.
- 🇷🇺 Полностью русскоязычный интерфейс и названия видов.
- 🎯 Результат: вид, вероятность в %, топ-3 кандидата, фото и краткое описание.
- ⚡ Асинхронная обработка через очередь — UI не блокируется, воркеры масштабируются.
- 🔌 REST API с автодокументацией (Swagger / ReDoc).

---

## Архитектура

```mermaid
flowchart LR
    U[Браузер<br/>UI на русском] -->|HTTP :80| N[nginx]
    N -->|proxy| A[FastAPI app :8080<br/>REST + UI]
    A -->|PENDING + аудио| DB[(PostgreSQL)]
    A -->|задача в очередь| MQ{{RabbitMQ}}
    MQ -->|competing consumers| W1[worker #1<br/>AST + голова]
    MQ --> W2[worker #2]
    MQ --> W3[worker #N]
    W1 -->|результат| DB
    W2 --> DB
    W3 --> DB
    U -. опрос GET /id .-> A
```

Поток: **загрузка → запись `PENDING` в БД + публикация задачи в RabbitMQ → воркер забирает задачу,
прогоняет модель, пишет результат в БД → UI опрашивает статус и показывает карточку вида.**

Несколько воркеров — *конкурирующие потребители* одной durable-очереди (`prefetch=1` + ручной `ack`):
задачи честно распределяются, при падении воркера задача не теряется.

---

## Доменная модель (основные сущности)

| Сущность | Назначение | Ключевые поля | Связи |
|---|---|---|---|
| **Species** (вид) | справочник 20 региональных видов | `name` (рус., = метка класса), `latin`, `description`, `photo_url`, `order_index` | 1 — N → PredictionScore |
| **Prediction** (запрос на определение) | загруженная запись и её обработка | `id` (uuid), `status` (PENDING/PROCESSING/DONE/FAILED), `audio_path`, `confidence`, `is_confident`, `top_species`, `duration_sec`, `processing_ms` | 1 — N → PredictionScore |
| **PredictionScore** (оценка по виду) | вероятности топ-N видов для записи | `probability`, `rank`, `species_name` | N — 1 → Prediction, → Species |

```
Species (1) ──< (N) PredictionScore (N) >── (1) Prediction
```

Код сущностей: [`common/models/`](common/models). B2B-расширения (организация, подписка, лимит «5 бесплатных
определений») в MVP сознательно не реализованы — авторизация конечного пользователя по ТЗ не нужна;
эти сущности добавляются поверх текущей схемы без её изменения.

---

## Технологии

FastAPI · SQLModel · PostgreSQL · RabbitMQ (pika) · PyTorch + Transformers (AST) · scikit-learn ·
librosa · Jinja2 + ванильный JS · nginx · Docker Compose · pytest.

---

## Быстрый старт (Docker, тома на диске D)

> Требуется Docker Desktop. Все тяжёлые данные (БД, очередь, **кэш модели ~346 МБ**, загруженные аудио)
> хранятся на **диске D** через переменную `HOST_DATA_DIR`.

```powershell
cd deploy
copy .env.example .env          # при желании поправьте пароли/путь HOST_DATA_DIR
mkdir D:\mfdp-data              # каталог для томов на D (если ещё нет)

docker compose up -d --build    # поднять весь стек
```

Откройте **http://localhost** — это UI. Swagger: **http://localhost/api/docs**.
Консоль RabbitMQ: **http://localhost:15672** (логин/пароль из `.env`).

При первом старте воркер скачивает модель AST с HuggingFace в `D:/mfdp-data/hf-cache` (один раз).
Прогресс видно в логах: `docker compose logs -f worker`.

> 💡 Чтобы и сам образ Docker лежал на D, в Docker Desktop:
> **Settings → Resources → Advanced → Disk image location → `D:\DockerData`** (разово).

Остановить: `docker compose down` (данные на D сохраняются).

---

## Масштабирование воркеров с моделью

Главное требование задания — горизонтальное масштабирование модели. Делается одной командой:

```powershell
docker compose up -d --scale worker=3
```

Поднимутся 3 одинаковых воркера, подключённых к одной очереди. RabbitMQ распределяет задачи между ними
(round-robin с `prefetch=1`). Проверить число потребителей очереди можно в консоли RabbitMQ
(http://localhost:15672 → Queues → `bird_predictions` → Consumers).

---

## REST API

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/api/predictions/` | Загрузить аудио (`multipart/form-data`, поле `file`) → создаёт запрос, отдаёт `{id, status}` |
| `GET` | `/api/predictions/{id}` | Статус и результат определения (вид, %, топ-3) |
| `GET` | `/api/predictions/` | История последних определений |
| `GET` | `/api/species/` | Справочник 20 видов |
| `GET` | `/health` | Healthcheck сервиса |
| `GET` | `/` | Веб-интерфейс |

Пример:

```bash
# поставить запись в очередь
curl -F "file=@bird.mp3" http://localhost/api/predictions/
# -> {"id":"a1b2...","status":"PENDING"}

# забрать результат
curl http://localhost/api/predictions/a1b2...
# -> {"status":"DONE","top_species":"Зяблик","confidence":0.81,"is_confident":true,"scores":[...]}
```

---

## Тесты

Покрыты критичные части: **логика инференса** (сегментация, агрегация, ранжирование, порог),
**REST-контракт** (загрузка → очередь → статус, валидация формата/размера) и **сидинг справочника**.
Тесты не требуют torch и RabbitMQ: ML-логика проверяется на чистых функциях с фейковой моделью,
API — с in-memory SQLite и подменённым издателем очереди.

```powershell
cd deploy
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

---

## Структура проекта

```
deploy/
├── docker-compose.yaml      # db, rabbitmq, app, worker (масштабируемый), web(nginx)
├── .env.example             # настройки + HOST_DATA_DIR=D:/mfdp-data
├── nginx/nginx.conf
├── common/                  # общий код API и воркера
│   ├── config.py            # настройки (pydantic-settings)
│   ├── database.py          # движок, сессии, init_db
│   ├── mq.py                # параметры RabbitMQ
│   └── models/              # СУЩНОСТИ: Species, Prediction, PredictionScore
├── app/                     # API-шлюз + UI
│   ├── main.py              # FastAPI (startup: init_db + сидинг)
│   ├── routes/              # /api/predictions, /api/species, /, /health
│   ├── publisher.py         # издатель задач в RabbitMQ
│   ├── seed.py              # 20 видов
│   ├── view/ static/        # Jinja2-шаблон + CSS/JS/заглушка
│   └── Dockerfile
├── worker/                  # масштабируемый воркер с моделью
│   ├── inference.py         # AST + голова: сегментация, агрегация, порог
│   ├── worker.py            # потребитель RabbitMQ -> инференс -> БД
│   ├── models_store/        # ast_head.joblib + classes.json
│   └── Dockerfile           # torch (CPU) + ffmpeg
└── tests/                   # pytest: inference, api, seed
```

---

## Соответствие критериям задания

| # | Критерий | Балл | Реализация |
|---|---|---|---|
| 1 | Доменная модель | 2 | `Species`, `Prediction`, `PredictionScore` со связями (`common/models/`) |
| 2 | Хранение в СУБД | 2 | PostgreSQL + SQLModel (история запросов и оценок) |
| 3 | REST-интерфейс | 2 | FastAPI: загрузка, статус, история, справочник + Swagger |
| 4 | Пользовательский интерфейс | 3 | Веб-UI на русском: drag&drop, превью, карточка с фото и % |
| 5 | Тесты критичных частей | 2 | pytest: инференс + REST + сидинг |
| 6 | Docker | 2 | `docker-compose` из 5 сервисов, тома на D |
| 7 | Масштабирование воркеров | 2 | RabbitMQ + `--scale worker=N` (конкурирующие потребители) |

---

## Ограничения и заметки

- **Версия scikit-learn.** Голова `ast_head.joblib` обучалась в Colab. Если при загрузке появится ошибка
  несовместимости версий — подгоните версию в `worker/requirements.txt` под ту, что использовалась при обучении.
- **Фото видов.** В `app/static/species/` лежат реальные фото всех 20 видов (~3 МБ), скачанные с
  Wikipedia / Wikimedia Commons скриптом [`scripts/fetch_species_photos.py`](scripts/fetch_species_photos.py)
  (источники — в `app/static/species/CREDITS.md`, лицензии в основном CC BY-SA / Public Domain).
  SVG-заглушка остаётся фолбэком (`onerror`), если файл недоступен. Обновить фото — перезапустить скрипт.
- **Первый запуск воркера** дольше обычного: качается backbone AST (~346 МБ) в кэш на D.
- **Точность.** По данным §4 ноутбука улучшения: при пороге τ=0.4 — точность ~92% при покрытии ~81%; порог
  настраивается через `CONFIDENCE_THRESHOLD` в `.env`.
