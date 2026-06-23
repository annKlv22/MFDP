"""Сидинг справочника видов идемпотентен и совпадает с метками классов модели."""
import json
from pathlib import Path

from sqlmodel import select

from app.seed import SPECIES_SEED, seed_species
from common.database import session_scope
from common.models import Species


def test_seed_has_20_unique_species():
    names = [s["name"] for s in SPECIES_SEED]
    assert len(names) == 20
    assert len(set(names)) == 20


def test_seed_matches_model_classes():
    classes_path = Path(__file__).resolve().parents[1] / "worker" / "models_store" / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    assert set(s["name"] for s in SPECIES_SEED) == set(classes)


def test_seed_is_idempotent(engine):
    first = seed_species()
    second = seed_species()
    assert first == 20      # первый прогон добавляет все 20
    assert second == 0      # повторный — ничего
    with session_scope() as session:
        assert len(session.exec(select(Species)).all()) == 20
