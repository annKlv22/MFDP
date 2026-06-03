import argparse
import csv
import random
from pathlib import Path


def collect_samples(processed_dir: Path) -> list[dict]:
    samples = []
    audio_exts = {".wav", ".mp3", ".flac"}
    for species_dir in sorted(processed_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name.replace("_", " ")
        for audio_file in sorted(species_dir.iterdir()):
            if audio_file.suffix.lower() not in audio_exts:
                continue
            samples.append({
                "filepath": str(audio_file),
                "label": species,
                "start": 0.0,
                "end": 5.0,
                "confidence": 1.0,
            })
    return samples


def stratified_split(samples: list[dict], val_ratio: float, seed: int = 42) -> tuple:
    from collections import defaultdict
    by_species = defaultdict(list)
    for s in samples:
        by_species[s["label"]].append(s)

    train, val = [], []
    rng = random.Random(seed)
    for species, items in by_species.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    return train, val


def write_csv(rows: list[dict], path: Path):
    fieldnames = ["filepath", "label", "start", "end", "confidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(name: str, samples: list[dict]):
    from collections import Counter
    counts = Counter(s["label"] for s in samples)
    print(f"\n{name} ({len(samples)} samples across {len(counts)} species):")
    for species, n in sorted(counts.items()):
        print(f"  {species:<40} {n:>4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/metadata")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(processed_dir)
    if not samples:
        print(f"No audio files found in {processed_dir}")
        return

    train, val = stratified_split(samples, args.val_split, args.seed)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    write_csv(train, train_path)
    write_csv(val, val_path)

    print_stats("TRAIN", train)
    print_stats("VAL", val)
    print(f"\nCSVs written to {output_dir}/")


if __name__ == "__main__":
    main()
