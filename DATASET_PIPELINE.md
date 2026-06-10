# Dataset Pipeline — Regional Bird ID (SPb/Leningrad Oblast)

## Источники данных
- **Xeno-Canto** — 173 записи (20 целевых видов), отобраны вручную по гео-боксу СПб/Ленобласти.
  - Источник: https://xeno-canto.org/explore?query=box%3A58.827%2C28.743%2C60.868%2C34.522
  - Лежат в `data/audio/<Вид>/` — **вид задаётся именем папки**; объединения «Воробей» (домовый+полевой)
    и «Чайка» (малая+озёрная+серебристая) реализованы как папки.
  - Под DVC отслеживается/пушится **только `data/audio`** (`processed`/`metadata` — производные, не пушатся).

## Порядок запуска
> На Colab весь пайплайн выполняет сам ноутбук `notebooks/bird_id_models.ipynb`.
> Локально нужен исправный Python с `librosa` (на повреждённом 3.14 не работает — используйте 3.12).

### 1. Препроцессинг → 5-сек сегменты WAV 48kHz (вид = имя папки)
```powershell
python src/data/preprocess.py --input data/audio --output data/processed --clean
```

### 2. CSV train/val/test без утечки
Сплит **по записям** (Xeno-Canto id), стратификация по виду, доли 70/15/15.
Сегменты одной записи не попадают в разные сплиты; проверка `assert_no_leakage`.
```powershell
python src/data/create_birdnet_csv.py --processed-dir data/processed --val-split 0.15 --test-split 0.15
```

### 3. Сохранить аудио через DVC
```powershell
dvc add data/audio
dvc push
git add data/audio.dvc
git commit -m "Update audio"
```

## Структура датасета
```
data/
  audio/                 <- записи Xeno-Canto по папкам-видам (под DVC)
    Большая_синица/
    Чайка/
    ...
  processed/             <- 5-сек сегменты (генерируются preprocess.py; НЕ под DVC)
    Большая_синица/
      XC1085132 - Большая синица - Parus major_seg000.wav
  metadata/              <- train/val/test.csv (генерируются create_birdnet_csv.py; НЕ под DVC)
```

## Аугментация
Отдельного скрипта нет. Для CNN аугментация делается **«на лету»** в ноутбуке
(SpecAugment: маскирование по частоте/времени + шум), только на train.
Эмбеддингам (AST/BirdNET) аугментация не нужна.

## По шуму
Шумоподавление не применяем — оно искажает признаки пения. Робастность к шуму даёт аугментация при обучении.
