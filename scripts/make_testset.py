"""Build the 430-image test set: N random images per GTSRB class.

Usage:  python scripts/make_testset.py
Output: data/testset/images/*.png, data/testset/labels.csv, data/classes.csv

Downloads the official GTSRB test split (89 MB) into data/raw/ on first run. Needs only Pillow.
The selection is deterministic (fixed seed), so the set is identical on every machine.
"""
import csv
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "data/raw/images/GTSRB/Final_Test/Images"
RAW_GT = ROOT / "data/raw/gt/GT-final_test.csv"
OUT = ROOT / "data/testset"
PER_CLASS = 10
URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
SEED = 42

# id, prompt name (fed to zero-shot models), display name (used in reports)
CLASSES = [
    (0, "speed limit 20 km/h", "Speed limit 20"),
    (1, "speed limit 30 km/h", "Speed limit 30"),
    (2, "speed limit 50 km/h", "Speed limit 50"),
    (3, "speed limit 60 km/h", "Speed limit 60"),
    (4, "speed limit 70 km/h", "Speed limit 70"),
    (5, "speed limit 80 km/h", "Speed limit 80"),
    (6, "end of speed limit 80 km/h", "End of speed limit 80"),
    (7, "speed limit 100 km/h", "Speed limit 100"),
    (8, "speed limit 120 km/h", "Speed limit 120"),
    (9, "no overtaking", "No overtaking"),
    (10, "no overtaking for trucks", "No overtaking for trucks"),
    (11, "right of way at the next intersection", "Right of way at the next intersection"),
    (12, "priority road", "Priority road"),
    (13, "yield", "Yield"),
    (14, "stop", "Stop"),
    (15, "no vehicles", "No vehicles"),
    (16, "no trucks", "No trucks"),
    (17, "no entry", "No entry"),
    (18, "general danger", "General danger"),
    (19, "dangerous curve to the left", "Dangerous curve left"),
    (20, "dangerous curve to the right", "Dangerous curve right"),
    (21, "double curve", "Double curve"),
    (22, "bumpy road", "Bumpy road"),
    (23, "slippery road", "Slippery road"),
    (24, "road narrows on the right", "Road narrows on the right"),
    (25, "road work", "Road work"),
    (26, "traffic signals ahead", "Traffic signals"),
    (27, "pedestrians", "Pedestrians"),
    (28, "children crossing", "Children crossing"),
    (29, "bicycles crossing", "Bicycles crossing"),
    (30, "beware of ice or snow", "Ice or snow"),
    (31, "wild animals crossing", "Wild animals"),
    (32, "end of all speed and overtaking limits", "End of all limits"),
    (33, "turn right ahead", "Turn right ahead"),
    (34, "turn left ahead", "Turn left ahead"),
    (35, "ahead only", "Ahead only"),
    (36, "go straight or right", "Go straight or right"),
    (37, "go straight or left", "Go straight or left"),
    (38, "keep right", "Keep right"),
    (39, "keep left", "Keep left"),
    (40, "roundabout mandatory", "Roundabout"),
    (41, "end of no overtaking", "End of no overtaking"),
    (42, "end of no overtaking for trucks", "End of no overtaking for trucks"),
]


def download() -> None:
    raw = ROOT / "data/raw"
    for name, target in (("GTSRB_Final_Test_GT.zip", raw / "gt"), ("GTSRB_Final_Test_Images.zip", raw / "images")):
        if target.exists():
            continue
        raw.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(URL + name, raw / name)
        zipfile.ZipFile(raw / name).extractall(target)


def main() -> None:
    download()
    by_class = defaultdict(list)
    with RAW_GT.open(newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            by_class[int(row["ClassId"])].append(row)

    assert sorted(by_class) == [c[0] for c in CLASSES], "unexpected set of classes"

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)

    rng = random.Random(SEED)
    rows = []
    for class_id in sorted(by_class):
        items = sorted(by_class[class_id], key=lambda r: r["Filename"])
        for row in rng.sample(items, PER_CLASS):
            src = RAW_IMAGES / row["Filename"]
            name = f"{class_id:02d}_{Path(row['Filename']).stem}.png"
            Image.open(src).convert("RGB").save(OUT / "images" / name)
            rows.append((name, class_id, row["Width"], row["Height"]))

    with (OUT / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class_id", "width", "height"])
        w.writerows(rows)

    with (ROOT / "data/classes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name_en", "name"])
        w.writerows(CLASSES)

    sizes = sorted(int(r[2]) for r in rows)
    print(f"images: {len(rows)}, classes: {len(by_class)}")
    print(f"width: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]} px")


if __name__ == "__main__":
    main()
