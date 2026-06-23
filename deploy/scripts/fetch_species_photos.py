"""Скачивает фото 20 видов с Wikipedia (Wikimedia Commons) в app/static/species/<slug>.jpg.

Источник — лид-изображение статьи по латинскому названию (MediaWiki API, prop=pageimages).
Slug-имена совпадают с полем "photo" в app/seed.py и с photo_url в БД.
Лицензии — на страницах файлов Commons (в основном CC BY-SA / Public Domain); см. CREDITS.md.

Зависимостей нет (только стандартная библиотека). Запуск:
    python scripts/fetch_species_photos.py
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# (slug файла, заголовок статьи = латинское название)
SPECIES = [
    ("turdus_iliacus", "Turdus iliacus"),
    ("parus_major", "Parus major"),
    ("dendrocopos_major", "Dendrocopos major"),
    ("passer_domesticus", "Passer domesticus"),
    ("fringilla_coelebs", "Fringilla coelebs"),
    ("troglodytes_troglodytes", "Troglodytes troglodytes"),
    ("anas_platyrhynchos", "Anas platyrhynchos"),
    ("cuculus_canorus", "Cuculus canorus"),
    ("cyanistes_caeruleus", "Cyanistes caeruleus"),
    ("ficedula_hypoleuca", "Ficedula hypoleuca"),
    ("certhia_familiaris", "Certhia familiaris"),
    ("sitta_europaea", "Sitta europaea"),
    ("corvus_cornix", "Corvus cornix"),
    ("columba_livia", "Columba livia"),
    ("sturnus_vulgaris", "Sturnus vulgaris"),
    ("pyrrhula_pyrrhula", "Pyrrhula pyrrhula"),
    ("luscinia_luscinia", "Luscinia luscinia"),
    ("chroicocephalus_ridibundus", "Chroicocephalus ridibundus"),
    ("spinus_spinus", "Spinus spinus"),
    ("turdus_merula", "Turdus merula"),
]

OUT = Path(__file__).resolve().parents[1] / "app" / "static" / "species"
API = "https://en.wikipedia.org/w/api.php"
UA = "MFDP-BirdID/1.0 (educational MVP)"
THUMB_PX = 640


def _fetch(url: str, timeout: int) -> bytes:
    """GET с ретраями: 429 -> экспоненциальная пауза, прочие сбои -> короткий повтор."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                wait = 10 * (attempt + 1)
                print(f"       429 (лимит), пауза {wait}с...")
                time.sleep(wait)
                continue
            raise
        except Exception as exc:  # noqa: BLE001 — таймаут/сетевой сбой
            last = exc
            time.sleep(5 * (attempt + 1))
    raise last


def lead_image_url(latin: str) -> str | None:
    url = API + "?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "redirects": 1,
        "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": THUMB_PX,
        "titles": latin,
    })
    data = json.loads(_fetch(url, timeout=30))
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    return page.get("thumbnail", {}).get("source")


def download(url: str, dest: Path) -> int:
    data = _fetch(url, timeout=90)
    dest.write_bytes(data)
    return len(data)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    credits = []
    ok = 0
    for slug, latin in SPECIES:
        dest = OUT / f"{slug}.jpg"
        if dest.exists() and dest.stat().st_size > 3000:  # уже скачано — пропускаем (идемпотентно)
            ok += 1
            credits.append(f"- `{slug}.jpg` — *{latin}* — Wikipedia/Wikimedia Commons")
            print(f"[have] {slug}.jpg  ({dest.stat().st_size // 1024} КБ)")
            continue
        try:
            url = lead_image_url(latin)
            if not url:
                print(f"[skip] нет изображения: {latin}")
                continue
            size = download(url, dest)
            ok += 1
            credits.append(f"- `{slug}.jpg` — *{latin}* — {url}")
            print(f"[ok]   {slug}.jpg  ({size // 1024} КБ)")
            time.sleep(1.5)  # вежливо к API
        except Exception as exc:  # noqa: BLE001
            print(f"[err]  {latin}: {exc}")

    (OUT / "CREDITS.md").write_text(
        "# Источники фотографий видов\n\n"
        "Изображения — лид-фото статей Wikipedia / Wikimedia Commons по латинскому названию вида.\n"
        "Лицензии указаны на страницах файлов Commons (в основном CC BY-SA и Public Domain).\n"
        "Для продакшена проверьте лицензию каждого файла и добавьте атрибуцию авторов.\n\n"
        + "\n".join(credits) + "\n",
        encoding="utf-8",
    )
    print(f"\nГотово: {ok}/{len(SPECIES)} фото. Источники записаны в {OUT / 'CREDITS.md'}")
    return 0 if ok == len(SPECIES) else 1


if __name__ == "__main__":
    sys.exit(main())
