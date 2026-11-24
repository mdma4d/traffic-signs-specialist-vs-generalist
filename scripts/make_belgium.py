"""Build the other-country test set: Belgian signs from BelgiumTSC (test split).

Only Belgian classes with an exact counterpart among the 43 GTSRB classes are used. The mapping was
made by hand from the photos of every class. Left out: classes with no GTSRB counterpart, class 32
(all speed limits under one id), class 35 (different arrows under one id), and classes 1, 6, 14, 16
(similar but different signs).

Usage:  python scripts/make_belgium.py
Output: data/belgium/images/*.png, data/belgium/labels.csv (class_id is the GTSRB class id).
Also data/belgium_train/: the same classes from the BelgiumTSC training split (plus 11 and 36, which
are absent from the test split). Only the reject model uses it (bench/reject.py, modes oe_mix and up).

Downloads BelgiumTSC (78 MB + 174 MB) into data/raw_belgium/ on first run. Needs only Pillow.
"""
import csv
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw_belgium/Testing"
OUT = ROOT / "data/belgium"

# BelgiumTSC class id -> GTSRB class id
MAP = {
    0: 22,   # bumpy road
    2: 23,   # slippery road
    3: 19,   # dangerous curve left
    4: 20,   # dangerous curve right
    5: 21,   # double curve, first to the left
    7: 28,   # children crossing
    8: 29,   # bicycles crossing
    10: 25,  # road work
    13: 18,  # general danger
    17: 11,  # right of way at the next intersection
    19: 13,  # yield
    21: 14,  # stop
    22: 17,  # no entry
    25: 16,  # no trucks
    28: 15,  # no vehicles
    31: 9,   # no overtaking
    34: 35,  # ahead only
    37: 40,  # roundabout
    61: 12,  # priority road
}


# present only in the BelgiumTSC training split
TRAIN_ONLY = {11: 26,  # traffic signals
              36: 36}  # go straight or right


URL = "https://btsd.ethz.ch/shareddata/BelgiumTSC/"


def download():
    raw = ROOT / "data/raw_belgium"
    for part in ("Testing", "Training"):
        if (raw / part).exists():
            continue
        raw.mkdir(parents=True, exist_ok=True)
        name = f"BelgiumTSC_{part}.zip"
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(URL + name, raw / name)
        zipfile.ZipFile(raw / name).extractall(raw)


def main():
    download()
    export(SRC, OUT, MAP)
    export(ROOT / "data/raw_belgium/Training", ROOT / "data/belgium_train", {**MAP, **TRAIN_ONLY})


def export(src, out, mapping):
    OUT, SRC = out, src
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for be, de in sorted(mapping.items()):
        for f in sorted((SRC / f"{be:05d}").glob("*.ppm")):
            img = Image.open(f).convert("RGB")
            name = f"{de:02d}_be{be:02d}_{f.stem}.png"
            img.save(OUT / "images" / name)
            rows.append((name, de, img.width, img.height))
    with (OUT / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "class_id", "width", "height"])
        w.writerows(rows)
    print(f"{out.name}: {len(rows)} images, {len(mapping)} classes")


if __name__ == "__main__":
    main()
