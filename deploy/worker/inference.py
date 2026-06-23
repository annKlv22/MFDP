# -*- coding: utf-8 -*-
"""Инференс регионального определителя: AST-эмбеддинги + лёгкая голова (StandardScaler+LogReg).

Контракт повторяет ноутбук обучения:
  аудио -> librosa 16 кГц моно -> 5-сек сегменты (шаг 2.5 с, отсев тихих по RMS)
  -> AST pooler_output (768) -> head.predict_proba по сегментам -> усреднение -> argmax.
Порог уверенности τ отсекает ответы «не уверены».

Тяжёлые зависимости (torch/transformers/librosa/joblib) импортируются лениво,
поэтому чистые функции ниже можно тестировать без них.
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from common.config import settings
from common.logging_conf import get_logger

logger = get_logger("inference")


# --------------------------- чистые функции (без torch) ---------------------------

def rms(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0


def segment_signal(
    y: np.ndarray, sr: int, seg_sec: float, hop_sec: float, min_rms: float
) -> List[np.ndarray]:
    """Нарезка сигнала на окна seg_sec с шагом hop_sec. Тихие окна (RMS < min_rms) отбрасываются.
    Короткий сигнал дополняется нулями до одного окна; хвост — если он не короче половины окна."""
    y = np.asarray(y, dtype=np.float32)
    seg_len = int(seg_sec * sr)
    hop_len = max(1, int(hop_sec * sr))
    if seg_len <= 0:
        return []

    if len(y) < seg_len:
        padded = np.zeros(seg_len, dtype=np.float32)
        padded[: len(y)] = y
        return [padded] if rms(padded) >= min_rms else []

    out: List[np.ndarray] = []
    start = 0
    while start + seg_len <= len(y):
        seg = y[start : start + seg_len]
        if rms(seg) >= min_rms:
            out.append(seg.astype(np.float32))
        start += hop_len

    remainder = y[start:]
    if len(remainder) >= seg_len // 2:
        padded = np.zeros(seg_len, dtype=np.float32)
        padded[: len(remainder)] = remainder
        if rms(padded) >= min_rms:
            out.append(padded)
    return out


def aggregate(proba_segments: np.ndarray) -> np.ndarray:
    """Усреднение вероятностей по сегментам -> вероятности уровня записи."""
    return np.asarray(proba_segments, dtype=np.float64).mean(axis=0)


def rank_scores(rec_proba: np.ndarray, classes: List[str], top_k: int) -> List[dict]:
    order = np.argsort(rec_proba)[::-1][: max(1, top_k)]
    return [
        {"rank": i + 1, "name": str(classes[j]), "probability": float(rec_proba[j])}
        for i, j in enumerate(order)
    ]


def run_pipeline(y, sr, embed_fn, head, *, threshold, seg_sec, hop_sec, min_rms, top_k) -> dict:
    """Полный конвейер от сигнала к результату. embed_fn и head внедряются (удобно для тестов).

    embed_fn(list[np.ndarray]) -> np.ndarray (n_segments, dim)
    head: объект с .classes_ и .predict_proba(X) -> (n_segments, n_classes)
    """
    segments = segment_signal(y, sr, seg_sec, hop_sec, min_rms)
    if not segments:  # запись слишком тихая/короткая — берём одно дополненное окно
        seg_len = max(1, int(seg_sec * sr))
        padded = np.zeros(seg_len, dtype=np.float32)
        n = min(len(y), seg_len)
        padded[:n] = np.asarray(y, dtype=np.float32)[:n]
        segments = [padded]

    emb = np.asarray(embed_fn(segments))
    proba = np.asarray(head.predict_proba(emb), dtype=np.float64)
    rec_proba = aggregate(proba)
    classes = list(head.classes_)
    scores = rank_scores(rec_proba, classes, top_k)
    confidence = float(np.max(rec_proba))
    return {
        "scores": scores,
        "confidence": confidence,
        "is_confident": bool(confidence >= threshold),
        "top_species": scores[0]["name"] if scores else None,
        "n_segments": len(segments),
    }


# --------------------------- модель (ленивая загрузка) ---------------------------

class BirdClassifier:
    """Backbone AST (HuggingFace) + голова (joblib). Загружается один раз на процесс воркера."""

    def __init__(self) -> None:
        import joblib
        import torch
        from transformers import ASTModel, AutoFeatureExtractor

        self._torch = torch
        self.device = settings.DEVICE
        head_path = os.path.join(settings.MODEL_DIR, "ast_head.joblib")
        classes_path = os.path.join(settings.MODEL_DIR, "classes.json")

        logger.info("Загрузка головы классификатора: %s", head_path)
        self.head = joblib.load(head_path)
        with open(classes_path, "r", encoding="utf-8") as fh:
            self.classes_json = json.load(fh)

        logger.info("Загрузка backbone AST '%s' (device=%s)...", settings.AST_MODEL_NAME, self.device)
        self.fe = AutoFeatureExtractor.from_pretrained(settings.AST_MODEL_NAME)
        self.ast = ASTModel.from_pretrained(settings.AST_MODEL_NAME).to(self.device).eval()
        logger.info("Модель готова. Классов в голове: %d", len(list(self.head.classes_)))

    def _embed(self, wavs: List[np.ndarray], batch: int = 8) -> np.ndarray:
        torch = self._torch
        chunks = []
        with torch.no_grad():
            for i in range(0, len(wavs), batch):
                part = [np.asarray(w, dtype=np.float32) for w in wavs[i : i + batch]]
                inp = self.fe(part, sampling_rate=settings.TARGET_SR, return_tensors="pt")
                feats = self.ast(inp["input_values"].to(self.device)).pooler_output
                chunks.append(feats.cpu().numpy())
        return np.concatenate(chunks)

    def predict(self, audio_path: str) -> dict:
        import librosa

        y, _ = librosa.load(audio_path, sr=settings.TARGET_SR, mono=True)
        duration = float(len(y) / settings.TARGET_SR) if len(y) else 0.0
        result = run_pipeline(
            y,
            settings.TARGET_SR,
            self._embed,
            self.head,
            threshold=settings.CONFIDENCE_THRESHOLD,
            seg_sec=settings.SEGMENT_SECONDS,
            hop_sec=settings.HOP_SECONDS,
            min_rms=settings.MIN_SEGMENT_RMS,
            top_k=settings.TOP_K_STORED,
        )
        result["duration_sec"] = round(duration, 2)
        return result
