"""Build the images without signs for the reject model (bench/reject.py). Needs torchvision.

1. Road background: crops of full GTSDB frames (600 frames of German roads, 1360x800) that do not
   touch any labelled sign. Frames 0-399 go to training, 400-599 to evaluation, so an evaluation crop
   never comes from a frame the model has seen. Flat crops (sky, asphalt) are discarded.
2. Random objects: 2,000 images from the CIFAR-100 test split. Evaluation only.

Usage:  python scripts/make_background.py
Output: data/background/{train,test,objects}/images/*.png and labels.csv

Downloads GTSDB (1.1 GB) and CIFAR-100 (170 MB) on first run.
"""
import csv
import random
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw_gtsdb"
OUT = ROOT / "data/background"
URL = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip"
PER_FRAME = 10      # crops per frame
MIN_STD = 20        # "not flat" threshold: standard deviation of brightness
MARGIN = 10         # margin around a labelled sign, px


def save(part, rows):
    with (OUT / part / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "class_id"])
        w.writerows((name, 43) for name in rows)  # 43 is "unknown"
    print(part, len(rows), "images")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "TrainIJCNN2013.zip"
    if not zpath.exists():
        print("downloading GTSDB, 1.1 GB ...", flush=True)
        urllib.request.urlretrieve(URL, zpath)
    gt = next(iter(RAW.rglob("gt.txt")), None)
    if gt is None:
        zipfile.ZipFile(zpath).extractall(RAW)
        gt = next(RAW.rglob("gt.txt"))

    boxes = {}
    for line in gt.read_text().split():
        f, x1, y1, x2, y2, _ = line.split(";")
        boxes.setdefault(f, []).append((int(x1), int(y1), int(x2), int(y2)))

    random.seed(0)
    rows = {"train": [], "test": []}
    for part in rows:
        (OUT / part / "images").mkdir(parents=True, exist_ok=True)
    for frame in sorted(gt.parent.glob("*.ppm")):
        part = "train" if int(frame.stem) < 400 else "test"
        img = Image.open(frame).convert("RGB")
        made, tries = 0, 0
        while made < PER_FRAME and tries < 500:
            tries += 1
            s = random.randint(30, 120)  # same sizes as cropped signs
            x, y = random.randint(0, img.width - s), random.randint(0, img.height - s)
            if any(x < x2 + MARGIN and x + s > x1 - MARGIN and y < y2 + MARGIN and y + s > y1 - MARGIN
                   for x1, y1, x2, y2 in boxes.get(frame.name, [])):
                continue
            crop = img.crop((x, y, x + s, y + s))
            if ImageStat.Stat(crop.convert("L")).stddev[0] < MIN_STD:
                continue
            name = f"bg_{frame.stem}_{made}.png"
            crop.save(OUT / part / "images" / name)
            rows[part].append(name)
            made += 1
    for part, r in rows.items():
        save(part, r)

    import torchvision
    cifar = torchvision.datasets.CIFAR100(str(ROOT / "data/raw_cifar"), train=False, download=True)
    (OUT / "objects/images").mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(2000):
        cifar[i][0].save(OUT / "objects/images" / f"obj_{i:04d}.png")
        names.append(f"obj_{i:04d}.png")
    save("objects", names)


if __name__ == "__main__":
    main()
