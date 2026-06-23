"""Критичная часть: логика инференса (сегментация, агрегация, ранжирование, порог).

Тестируется без torch/transformers — на чистых функциях с внедрёнными fake embed_fn и head.
"""
import numpy as np

from worker.inference import aggregate, rank_scores, rms, run_pipeline, segment_signal

SR = 16000


def test_rms_silence_is_zero():
    assert rms(np.zeros(100)) == 0.0
    assert rms(np.ones(100)) == 1.0


def test_segment_lengths_are_full_window():
    y = np.ones(SR * 5, dtype=np.float32)              # 5 секунд, не тихие
    segs = segment_signal(y, SR, seg_sec=5.0, hop_sec=2.5, min_rms=0.005)
    assert len(segs) >= 1
    assert all(len(s) == SR * 5 for s in segs)         # каждое окно ровно 5 сек


def test_segment_drops_full_silence():
    y = np.zeros(SR * 10, dtype=np.float32)
    assert segment_signal(y, SR, 5.0, 2.5, 0.005) == []


def test_segment_short_signal_is_padded_to_one_window():
    y = np.ones(SR * 2, dtype=np.float32)              # 2 сек < окна
    segs = segment_signal(y, SR, 5.0, 2.5, 0.005)
    assert len(segs) == 1
    assert len(segs[0]) == SR * 5


def test_aggregate_is_mean_over_segments():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_allclose(aggregate(proba), [0.4, 0.6])


def test_rank_scores_sorted_desc_with_ranks():
    scores = rank_scores(np.array([0.1, 0.7, 0.2]), ["A", "B", "C"], top_k=3)
    assert [s["name"] for s in scores] == ["B", "C", "A"]
    assert [s["rank"] for s in scores] == [1, 2, 3]
    assert abs(scores[0]["probability"] - 0.7) < 1e-9


def _fake_embed(segments):
    return np.zeros((len(segments), 4), dtype=np.float32)


class _FakeHead:
    def __init__(self, classes, row):
        self.classes_ = np.array(classes)
        self._row = np.array(row, dtype=np.float64)

    def predict_proba(self, X):
        return np.tile(self._row, (len(X), 1))


def test_run_pipeline_picks_top_and_marks_confident():
    y = np.ones(SR * 10, dtype=np.float32)
    head = _FakeHead(["Воробей", "Зяблик", "Кряква"], [0.1, 0.8, 0.1])
    res = run_pipeline(y, SR, _fake_embed, head, threshold=0.4,
                       seg_sec=5.0, hop_sec=2.5, min_rms=0.005, top_k=3)
    assert res["top_species"] == "Зяблик"
    assert res["is_confident"] is True
    assert abs(res["confidence"] - 0.8) < 1e-9
    assert res["scores"][0]["name"] == "Зяблик"
    assert len(res["scores"]) == 3


def test_run_pipeline_low_confidence_below_threshold():
    y = np.ones(SR * 5, dtype=np.float32)
    head = _FakeHead(["A", "B", "C", "D"], [0.25, 0.25, 0.25, 0.25])
    res = run_pipeline(y, SR, _fake_embed, head, threshold=0.4,
                       seg_sec=5.0, hop_sec=2.5, min_rms=0.005, top_k=4)
    assert res["is_confident"] is False
    assert abs(res["confidence"] - 0.25) < 1e-9
