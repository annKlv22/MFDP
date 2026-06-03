import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 48000

# SNR range for noise overlay (dB)
SNR_RANGE = (5, 20)

# Pitch shift range (semitones)
PITCH_RANGE = (-2, 2)

# Time shift range (fraction of segment duration)
TIME_SHIFT_RANGE = (-0.3, 0.3)


def load_noise_files(noise_dir: Path) -> list[np.ndarray]:
    """Load background noise clips from a directory."""
    import librosa
    clips = []
    for p in noise_dir.rglob("*.wav"):
        try:
            arr, _ = librosa.load(str(p), sr=TARGET_SR, mono=True)
            clips.append(arr)
        except Exception:
            pass
    return clips


def add_noise(signal: np.ndarray, noise_clips: list[np.ndarray], snr_db: float) -> np.ndarray:
    if not noise_clips:
        # Fallback: Gaussian white noise
        noise = np.random.randn(len(signal)).astype(np.float32)
    else:
        clip = random.choice(noise_clips)
        # Tile noise to match signal length
        if len(clip) < len(signal):
            repeats = (len(signal) // len(clip)) + 1
            clip = np.tile(clip, repeats)
        start = random.randint(0, len(clip) - len(signal))
        noise = clip[start:start + len(signal)]

    sig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)
    return np.clip(signal + noise, -1.0, 1.0)


def time_shift(signal: np.ndarray, shift_frac: float) -> np.ndarray:
    shift_samples = int(shift_frac * len(signal))
    return np.roll(signal, shift_samples)


def pitch_shift(signal: np.ndarray, semitones: float) -> np.ndarray:
    try:
        import librosa
        return librosa.effects.pitch_shift(signal, sr=TARGET_SR, n_steps=semitones)
    except Exception:
        return signal


def augment_file(path: Path, noise_clips: list[np.ndarray], copies: int, rng: random.Random) -> int:
    try:
        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        print(f"  Cannot read {path.name}: {e}")
        return 0

    if sr != TARGET_SR:
        import librosa
        arr, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)

    saved = 0
    for i in range(copies):
        aug = arr.copy()

        # Random time shift
        shift = rng.uniform(*TIME_SHIFT_RANGE)
        aug = time_shift(aug, shift)

        # Random pitch shift ~50% of the time
        if rng.random() < 0.5:
            semitones = rng.uniform(*PITCH_RANGE)
            aug = pitch_shift(aug, semitones)

        # Always add noise
        snr = rng.uniform(*SNR_RANGE)
        aug = add_noise(aug, noise_clips, snr)

        out_path = path.parent / f"{path.stem}_aug{i}.wav"
        if not out_path.exists():
            sf.write(str(out_path), aug, TARGET_SR, subtype="PCM_16")
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--copies", type=int, default=3,
                        help="Augmented copies per original file")
    parser.add_argument("--noise-dir", default=None,
                        help="Directory with background noise WAV files (ESC-50 or UrbanSound)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-files-to-augment", type=int, default=0,
                        help="Only augment species with fewer than N files (0 = augment all)")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    rng = random.Random(args.seed)

    noise_clips = []
    if args.noise_dir:
        noise_clips = load_noise_files(Path(args.noise_dir))
        print(f"Loaded {len(noise_clips)} noise clips")
    else:
        print("No noise dir specified, using Gaussian noise as fallback")

    audio_exts = {".wav"}
    total_new = 0

    for species_dir in sorted(processed_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        originals = [p for p in species_dir.iterdir()
                     if p.suffix.lower() in audio_exts and "_aug" not in p.stem]

        if args.min_files_to_augment > 0 and len(originals) >= args.min_files_to_augment:
            continue

        species = species_dir.name.replace("_", " ")
        print(f"\n{species}: {len(originals)} originals -> {len(originals) * args.copies} new")

        for orig in originals:
            n = augment_file(orig, noise_clips, args.copies, rng)
            total_new += n

    print(f"\nAugmentation done. {total_new} new files created.")


if __name__ == "__main__":
    main()
