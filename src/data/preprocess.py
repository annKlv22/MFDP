"""
Preprocess audio files for BirdNET fine-tuning:
  - Convert to WAV mono 48kHz
  - Segment into 5-second chunks with 2.5s overlap
  - Skip near-silent segments
  - Output: data/processed/{Genus_species}/*.wav

Species is extracted from the Xeno-Canto filename:
  "XC123456 - Русское название - Genus species.mp3"  -> "Genus species"

Usage:
    python src/data/preprocess.py --input data/audio --output data/processed
    python src/data/preprocess.py --input data/audio_xc --output data/processed
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

TARGET_SR = 48000
SEGMENT_DURATION = 5.0
HOP_DURATION = 2.5
MIN_AMPLITUDE = 0.005

XC_PATTERN = re.compile(r"XC\d+\s+-\s+.+?\s+-\s+(.+?)\.(mp3|wav|ogg|flac)$", re.IGNORECASE)


def extract_species(filename: str) -> str | None:
    m = XC_PATTERN.match(filename)
    if m:
        return m.group(1).strip()
    return None


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
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("Install: pip install librosa soundfile")
        sys.exit(1)

    try:
        array, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  Cannot load {audio_path.name}: {e}")
        return 0

    if len(array) < TARGET_SR:
        return 0

    species_dir = out_dir / species.replace(" ", "_")
    species_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, (_, seg) in enumerate(segment_audio(array, TARGET_SR)):
        if rms(seg) < MIN_AMPLITUDE:
            continue
        out_path = species_dir / f"{audio_path.stem}_seg{i:03d}.wav"
        if not out_path.exists():
            sf.write(str(out_path), seg, TARGET_SR, subtype="PCM_16")
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with raw audio files")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_exts = {".mp3", ".wav", ".ogg", ".flac"}
    files = [p for p in in_dir.iterdir() if p.suffix.lower() in audio_exts]
    print(f"Found {len(files)} files in {in_dir}")

    total, skipped = 0, 0
    for audio_path in sorted(files):
        species = extract_species(audio_path.name)
        if species is None:
            print(f"  SKIP (no species in name): {audio_path.name}")
            skipped += 1
            continue
        n = process_file(audio_path, species, out_dir)
        total += n
        print(f"  {audio_path.name} -> {species} ({n} segments)")

    print(f"\nDone. {total} segments saved, {skipped} files skipped.")


if __name__ == "__main__":
    main()
