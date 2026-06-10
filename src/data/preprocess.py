"""Нарезка исходных аудио на 5-секундные сегменты.

Поддерживает два режима входной папки:
  1) НОВЫЙ (по подпапкам): data/audio/<Вид>/<файлы> -> вид = имя подпапки.
     Объединения видов (Воробей, Чайка и т.п.) задаются самой структурой папок.
  2) СТАРЫЙ (плоский): файлы 'XC123456 - Рус название - Genus species.ext' -> вид из имени файла.

Вид определяется автоматически: если во входной папке есть подпапки — режим 1, иначе режим 2.
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET_SR = 48000
SEGMENT_DURATION = 5.0
HOP_DURATION = 2.5
MIN_AMPLITUDE = 0.005
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

XC_PATTERN = re.compile(r"XC\d+\s+-\s+.+?\s+-\s+(.+?)\.(mp3|wav|ogg|flac)$", re.IGNORECASE)


def extract_species_from_name(filename: str):
    m = XC_PATTERN.match(filename)
    return m.group(1).strip() if m else None


def rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(array ** 2)))


def segment_audio(array: np.ndarray, sr: int):
    seg_len = int(SEGMENT_DURATION * sr)
    hop_len = int(HOP_DURATION * sr)
    segments = []
    start = 0
    while start + seg_len <= len(array):
        segments.append((start, array[start:start + seg_len]))
        start += hop_len
    remainder = array[start:]
    if len(remainder) >= seg_len // 2:
        padded = np.zeros(seg_len, dtype=array.dtype)
        padded[:len(remainder)] = remainder
        segments.append((start, padded))
    return segments


def process_file(audio_path: Path, species: str, out_dir: Path) -> int:
    import librosa
    import soundfile as sf
    try:
        array, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  Не удалось загрузить {audio_path.name}: {e}")
        return 0

    if len(array) < TARGET_SR:           # короче 1 секунды — пропускаем
        return 0

    species_dir = out_dir / species.replace(" ", "_")
    species_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, (_, seg) in enumerate(segment_audio(array, TARGET_SR)):
        if rms(seg) < MIN_AMPLITUDE:     # отсев тихих сегментов
            continue
        out_path = species_dir / f"{audio_path.stem}_seg{i:03d}.wav"
        if not out_path.exists():
            sf.write(str(out_path), seg, TARGET_SR, subtype="PCM_16")
        saved += 1
    return saved


def iter_inputs(in_dir: Path):
    """Выдаёт пары (audio_path, species)."""
    subdirs = [p for p in sorted(in_dir.iterdir()) if p.is_dir()]
    if subdirs:
        # Режим 1: вид = имя подпапки
        for sp_dir in subdirs:
            for f in sorted(sp_dir.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS:
                    yield f, sp_dir.name
    else:
        # Режим 2: вид из имени файла
        for f in sorted(in_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                sp = extract_species_from_name(f.name)
                if sp is None:
                    print(f"  ПРОПУСК (нет вида в имени): {f.name}")
                    continue
                yield f, sp


def main():
    parser = argparse.ArgumentParser(
        description="Нарезка аудио на 5-сек сегменты (вид = имя подпапки или из имени файла).")
    parser.add_argument("--input", required=True, help="Папка с аудио (подпапки по видам или плоские файлы)")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--clean", action="store_true",
                        help="очистить output перед нарезкой (убирает сегменты прошлого набора видов)")
    args = parser.parse_args()

    try:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        print("Нужны библиотеки: pip install librosa soundfile")
        sys.exit(1)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(iter_inputs(in_dir))
    print(f"Найдено {len(pairs)} аудиофайлов в {in_dir}")

    per_species = defaultdict(int)
    total = 0
    for audio_path, species in pairs:
        n = process_file(audio_path, species, out_dir)
        per_species[species] += n
        total += n

    print(f"\nГотово. {total} сегментов, {len(per_species)} видов:")
    for sp, n in sorted(per_species.items()):
        print(f"  {sp}: {n}")


if __name__ == "__main__":
    main()
