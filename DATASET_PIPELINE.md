# Dataset Pipeline — Regional Bird ID (SPb/Leningrad Oblast)

## Источники данных
- **Xeno-Canto** — 188 записей, скачаны вручную, вид кодируется в названии файла
  - Источник: https://xeno-canto.org/explore?query=box%3A58.827%2C28.743%2C60.868%2C34.522
  - Файлы лежат в `data/audio/` 

## Порядок запуска

### 1. Препроцессинг → 5-секундные сегменты WAV 48kHz
```powershell
python src/data/preprocess.py --input data/audio --output data/processed
```

### 2. Аугментация (особенно для редких видов с < 30 сегментами)
```powershell
python src/data/augment.py --processed-dir data/processed --copies 3 --min-files-to-augment 30
```

### 3. CSV для BirdNET
```powershell
python src/data/create_birdnet_csv.py --processed-dir data/processed --val-split 0.15
```

### 4. Сохранить через DVC
```powershell
dvc add data/audio data/audio_xc data/processed
git add data/audio.dvc data/audio_xc.dvc data/processed.dvc data/.gitignore
dvc push
git commit -m "Add XC audio + processed segments"
```

## Структура датасета
```
data/
  audio/              <- записи Xeno-Canto (вручную)
  processed/
    Cuculus_canorus/
      XC56032_seg000.wav
      ...
  metadata/
    train.csv
    val.csv
```

## Формат имён файлов Xeno-Canto
```
XC123456 - Русское название - Genus species.mp3
```
Скрипт `preprocess.py` автоматически извлекает `Genus species` из имени файла.

## По шуму
Шумоподавление не применяем — нарушает признаки пения.
Вместо этого аугментация при обучении: наложение шума, pitch/time shift.
