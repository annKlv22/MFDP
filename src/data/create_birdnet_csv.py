"""Формирует CSV train/val/test БЕЗ утечки данных.

Ключевое отличие от наивного разбиения: делим выборку ПО ЗАПИСЯМ (Xeno-Canto id),
а не по файлам. Все сегменты одной физической записи (и их аугментации) попадают
строго в один сплит. Стратификация по виду. Аугментация (`_aug`) допускается только в train.

Подвиды по умолчанию схлопываются до бинома `Genus species` (как задумано в EDA).
"""
import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def normalize_species(dirname: str) -> str:
    """Вид = имя папки. Объединения видов (Воробей, Чайка и т.п.) уже заданы структурой data/processed."""
    return dirname.replace("_", " ")


def recording_id(path: Path) -> str:
    """Все сегменты одной физической записи имеют один XC-идентификатор."""
    m = re.search(r"XC\d+", path.name)
    return m.group(0) if m else path.stem


def collect_samples(processed_dir: Path) -> list[dict]:
    samples = []
    for species_dir in sorted(processed_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species = normalize_species(species_dir.name)
        for audio_file in sorted(species_dir.iterdir()):
            if audio_file.suffix.lower() not in AUDIO_EXTS:
                continue
            samples.append({
                "filepath": str(audio_file),
                "label": species,
                "rec_id": recording_id(audio_file),
                "is_aug": "_aug" in audio_file.stem,
                "start": 0.0,
                "end": 5.0,
                "confidence": 1.0,
            })
    return samples


def assign_recording_splits(samples: list[dict], val_ratio: float,
                            test_ratio: float, seed: int = 42) -> dict:
    """Сплит по записям со стратификацией по виду, без утечки.

    Малое число записей у вида:
      1 запись  -> только train (валидировать нельзя);
      2 записи  -> train + test;
      3 записи  -> train + val + test;
      4+        -> доли val/test с гарантией, что train не опустеет.
    При test_ratio == 0 делаем только train/val (старое поведение, но по записям).
    """
    rng = random.Random(seed)

    rec_labels = defaultdict(list)
    for s in samples:
        rec_labels[s["rec_id"]].append(s["label"])
    rec_species = {r: Counter(ls).most_common(1)[0][0] for r, ls in rec_labels.items()}

    by_species = defaultdict(list)
    for rec, sp in rec_species.items():
        by_species[sp].append(rec)

    three_way = test_ratio > 0
    split_of = {}
    for sp in sorted(by_species):
        recs = sorted(by_species[sp])
        rng.shuffle(recs)
        n = len(recs)

        if not three_way:
            if n == 1:
                parts = {"train": recs}
            else:
                n_val = min(max(1, round(val_ratio * n)), n - 1)
                parts = {"val": recs[:n_val], "train": recs[n_val:]}
        else:
            if n == 1:
                parts = {"train": recs}
            elif n == 2:
                parts = {"train": recs[:1], "test": recs[1:]}
            elif n == 3:
                parts = {"train": recs[:1], "val": recs[1:2], "test": recs[2:]}
            else:
                n_test = min(max(1, round(test_ratio * n)), n - 2)
                n_val = min(max(1, round(val_ratio * n)), n - 1 - n_test)
                parts = {
                    "test": recs[:n_test],
                    "val": recs[n_test:n_test + n_val],
                    "train": recs[n_test + n_val:],
                }

        for split, rs in parts.items():
            for r in rs:
                split_of[r] = split
    return split_of


def build_splits(samples: list[dict], split_of: dict) -> dict:
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        split = split_of.get(s["rec_id"])
        if split is None:
            continue
        if s["is_aug"] and split != "train":   # аугментацию — только в train
            continue
        out[split].append(s)
    return out


def write_csv(rows: list[dict], path: Path):
    fieldnames = ["filepath", "label", "start", "end", "confidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_stats(name: str, rows: list[dict]):
    counts = Counter(r["label"] for r in rows)
    recs = len({r["rec_id"] for r in rows})
    aug = sum(r["is_aug"] for r in rows)
    print(f"\n{name}: {len(rows)} сегментов | {recs} записей | {len(counts)} видов | аугментаций {aug}")
    for species, n in sorted(counts.items()):
        print(f"  {species:<40} {n:>4}")


def assert_no_leakage(splits: dict):
    ids = {k: {r["rec_id"] for r in v} for k, v in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = ids[a] & ids[b]
        assert not overlap, f"УТЕЧКА: {len(overlap)} записей одновременно в {a} и {b}"
    print("\nПроверка утечки: пересечений записей между сплитами нет (0). OK")


def main():
    parser = argparse.ArgumentParser(
        description="CSV train/val/test без утечки: сплит по записям (Xeno-Canto id).")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/metadata")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="доля на test; 0 -> только train/val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="не печатать поштучную статистику по видам")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(processed_dir)
    if not samples:
        print(f"Не найдено аудио в {processed_dir}")
        return

    split_of = assign_recording_splits(samples, args.val_split, args.test_split, args.seed)
    splits = build_splits(samples, split_of)

    write_csv(splits["train"], output_dir / "train.csv")
    write_csv(splits["val"], output_dir / "val.csv")
    if args.test_split > 0:
        write_csv(splits["test"], output_dir / "test.csv")

    all_species = {s["label"] for s in samples}
    n_recs = len({s["rec_id"] for s in samples})
    n_aug = sum(s["is_aug"] for s in samples)
    print(f"Видов: {len(all_species)} | записей: {n_recs} | "
          f"сегментов: {len(samples)} (в т.ч. аугментаций: {n_aug})")

    for name in ("train", "val", "test"):
        if splits[name]:
            if args.quiet:
                recs = len({r['rec_id'] for r in splits[name]})
                print(f"{name.upper()}: {len(splits[name])} сегментов | {recs} записей | "
                      f"{len({r['label'] for r in splits[name]})} видов")
            else:
                print_stats(name.upper(), splits[name])

    assert_no_leakage(splits)

    val_species = {r["label"] for r in splits["val"]}
    test_species = {r["label"] for r in splits["test"]}
    only_train = sorted(all_species - val_species - test_species)
    if only_train:
        print(f"\nВиды без val/test (одна запись, невозможно честно валидировать): {len(only_train)}")
        print("  " + ", ".join(only_train))

    print(f"\nCSV записаны в {output_dir}/")


if __name__ == "__main__":
    main()
