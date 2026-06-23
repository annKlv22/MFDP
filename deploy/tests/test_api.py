"""Критичная часть: REST-контракт API (загрузка -> очередь -> статус) и валидация."""
import io
import math
import struct
import wave


def _wav_bytes(seconds: int = 1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(1000 * math.sin(2 * math.pi * 440 * t / sr)))
            for t in range(sr * seconds)
        )
        w.writeframes(frames)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_species_seeded_20(client):
    r = client.get("/api/species/")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 20
    assert data[0]["name"] == "Белобровик"   # первый по order_index


def test_upload_creates_pending_and_publishes(client, tmp_path, monkeypatch):
    monkeypatch.setattr("common.config.settings.UPLOAD_DIR", str(tmp_path))
    r = client.post(
        "/api/predictions/",
        files={"file": ("bird.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "PENDING"
    pid = body["id"]

    # задача ушла в (фейковую) очередь
    assert pid in client.fake_publisher.published

    # статус доступен через GET
    g = client.get(f"/api/predictions/{pid}")
    assert g.status_code == 200
    assert g.json()["status"] == "PENDING"
    assert g.json()["filename"] == "bird.wav"


def test_reject_unsupported_extension(client):
    r = client.post(
        "/api/predictions/",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_reject_too_large_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr("common.config.settings.UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr("common.config.settings.MAX_UPLOAD_MB", 0)  # любой непустой файл превысит лимит
    r = client.post(
        "/api/predictions/",
        files={"file": ("bird.wav", b"\x00" * 2048, "audio/wav")},
    )
    assert r.status_code == 413


def test_get_unknown_prediction_404(client):
    r = client.get("/api/predictions/nonexistent")
    assert r.status_code == 404
