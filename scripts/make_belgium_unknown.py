"""Build the set of "unknown" signs: Belgian signs that are not among the 43 GTSRB classes.

Used by the reject model (bench/reject.py): the right answer for these photos is "unknown".
The 36 classes are split alternately into two groups:
  A - sign types the model sees while training the "unknown" class (photos from the Training split);
  B - sign types the model never sees, kept for evaluation only (except in mode oe_all).
Left out: classes that have a GTSRB counterpart (see make_belgium.py, plus 11 and 36, which are
absent from the test split) and ambiguous ones: 6, 15, 16 (mirrored variants of GTSRB signs),
32 (all speed limits under one id), 35 (different arrows under one id).

Usage:  python scripts/make_belgium_unknown.py     (after scripts/make_belgium.py)
Output: data/belgium_unknown/{train,test}/images/*.png and labels.csv (filename, be_class, group).
"""
import csv
from pathlib import Path

from PIL import Image

from make_belgium import MAP

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw_belgium"
OUT = ROOT / "data/belgium_unknown"
SKIP = set(MAP) | {6, 11, 15, 16, 32, 35, 36}
UNKNOWN = [c for c in range(62) if c not in SKIP]
GROUP = {c: "AB"[i % 2] for i, c in enumerate(UNKNOWN)}
# Label error in BelgiumTSC: this track sits in the folder of class 18 but shows a class 17 sign
# ("Right of way at the next intersection", a known class). Found while inspecting confident errors.
MISLABELED = {"00018/02048"}


def main():
    for part, src in (("train", "Training"), ("test", "Testing")):
        (OUT / part / "images").mkdir(parents=True, exist_ok=True)
        for old in (OUT / part / "images").glob("*.png"):
            old.unlink()
        rows = []
        for c in UNKNOWN:
            for f in sorted((SRC / src / f"{c:05d}").glob("*.ppm")):
                if f"{c:05d}/{f.stem.split('_')[0]}" in MISLABELED:
                    continue
                name = f"be{c:02d}_{f.stem}.png"
                Image.open(f).convert("RGB").save(OUT / part / "images" / name)
                rows.append((name, c, GROUP[c]))
        with (OUT / part / "labels.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "be_class", "group"])
            w.writerows(rows)
        print(part, len(rows), "images:", {g: sum(r[2] == g for r in rows) for g in "AB"})
    print("A:", [c for c in UNKNOWN if GROUP[c] == "A"], "B:", [c for c in UNKNOWN if GROUP[c] == "B"])


if __name__ == "__main__":
    main()
