# -*- coding: utf-8 -*-
import io, re, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, "src/data")
from create_birdnet_csv import assign_recording_splits

AUDIO = Path("data/audio")
EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

samples, files_per, recs_per = [], defaultdict(int), defaultdict(set)
for d in sorted(AUDIO.iterdir()):
    if not d.is_dir():
        continue
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in EXTS:
            m = re.search(r"XC\d+", f.name)
            rid = m.group(0) if m else f.stem
            samples.append({"filepath": str(f), "label": d.name, "rec_id": rid, "is_aug": False})
            files_per[d.name] += 1
            recs_per[d.name].add(rid)

split_of = assign_recording_splits(samples, val_ratio=0.15, test_ratio=0.15, seed=42)
per = defaultdict(lambda: defaultdict(set))
for s in samples:
    per[s["label"]][split_of[s["rec_id"]]].add(s["rec_id"])

recs = {sp: len(v) for sp, v in recs_per.items()}
nrec = sorted(recs.values())
tot_rec, tot_files = sum(recs.values()), sum(files_per.values())
split_tot = Counter()
for sp in per:
    for k, v in per[sp].items():
        split_tot[k] += len(v)
hist = Counter(nrec)

out = io.StringIO(); w = out.write
w(f"видов={len(recs)} записей={tot_rec} файлов={tot_files} (многофайловых записей={tot_files-tot_rec})\n")
w(f"записей_на_вид: min={min(nrec)} медиана={st.median(nrec)} среднее={tot_rec/len(recs):.2f} max={max(nrec)} дисбаланс={max(nrec)/min(nrec):.2f}x\n")
w(f"сплит_записей train/val/test={split_tot['train']}/{split_tot['val']}/{split_tot['test']}\n")
w(f"гистограмма(записей:видов)={dict(sorted(hist.items()))}\n")
w("| Вид | Записей | Файлов | tr | val | test |\n|---|--:|--:|--:|--:|--:|\n")
for sp in sorted(recs, key=lambda s: (-recs[s], s)):
    c = {k: len(per[sp][k]) for k in ("train", "val", "test")}
    flag = "" if files_per[sp] == recs[sp] else "  <-- файлов != записей"
    w(f"| {sp} | {recs[sp]} | {files_per[sp]} | {c['train']} | {c['val']} | {c['test']} |{flag}\n")
ids = defaultdict(set)
for s in samples:
    ids[split_of[s["rec_id"]]].add(s["rec_id"])
leak = len(ids["train"] & ids["val"]) + len(ids["train"] & ids["test"]) + len(ids["val"] & ids["test"])
w(f"утечка={leak}\n")
Path("notebooks/_recount.md").write_text(out.getvalue(), encoding="utf-8")
print("OK species=", len(recs), "records=", tot_rec, "files=", tot_files,
      "split=", dict(split_tot), "leak=", leak, "min/med/max=", min(nrec), st.median(nrec), max(nrec),
      "imbalance=%.2f" % (max(nrec)/min(nrec)))
