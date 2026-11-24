"""Multi-view agreement (test-time augmentation): an attempt to remove confident errors without retraining.

Usage:  python bench/tta.py [models/big_cnn_oe_bg.pt]
Output: results/tta.json

One saved model looks at each image 9 times: as is, zoomed in, zoomed out, rotated by +-8 degrees,
shifted horizontally and vertically, darker and brighter. The answer is accepted only if all 9
answers agree and the lowest of the 9 confidences reaches the threshold.
Compared with the plain single-view rule at the same share of accepted held-out German signs.
"""
import json
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

from common import RESULTS, ROOT, load_testset
from reject import KEEP, unknown_items

sys.path.insert(0, str(ROOT))
from predict import load  # noqa: E402

# (angle, x shift, y shift as fractions of the side, scale, brightness)
VIEWS = [(0, 0, 0, 1.0, 1.0), (0, 0, 0, 1.1, 1.0), (0, 0, 0, 0.9, 1.0), (8, 0, 0, 1.0, 1.0), (-8, 0, 0, 1.0, 1.0),
         (0, 0.06, 0, 1.0, 1.0), (0, 0, 0.06, 1.0, 1.0), (0, 0, 0, 1.0, 0.75), (0, 0, 0, 1.0, 1.3)]


class Views(torch.utils.data.Dataset):
    def __init__(self, items, size, norm):
        self.items, self.size, self.norm = items, size, norm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        out = []
        for angle, dx, dy, scale, bright in VIEWS:
            v = F.affine(img, angle, [int(dx * self.size), int(dy * self.size)], scale, 0.0,
                         interpolation=T.InterpolationMode.BILINEAR)
            out.append(self.norm(F.to_tensor(F.adjust_brightness(v, bright))))
        return torch.stack(out), label


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models/big_cnn_oe_bg.pt")
    dev = "cuda"
    model, tf, _ = load(path, dev)
    size, norm = tf.transforms[0].size[0], tf.transforms[2]

    root = str(ROOT / "data/gtsrb_train")
    samples = torchvision.datasets.GTSRB(root, split="train", download=True)._samples
    val_known = [(p, y) for p, y in samples if int(Path(p).name.split("_")[0]) % 10 == 9]
    sets = {
        "val_known": val_known,
        "gtsrb_test": list(torchvision.datasets.GTSRB(root, split="test", download=True)._samples),
        "belgium_known": load_testset(ROOT / "data/belgium"),
        "unknown_A": unknown_items("test", "A"),
        "unknown_B": unknown_items("test", "B"),
        "background": load_testset(ROOT / "data/background/test"),
        "objects": load_testset(ROOT / "data/background/objects"),
    }

    @torch.no_grad()
    def run(items):
        """Per image: (true class, answer of the first view, its confidence, do all views agree, min confidence)."""
        out = []
        for x, y in torch.utils.data.DataLoader(Views(items, size, norm), batch_size=128, num_workers=16):
            b, v = x.shape[:2]
            prob = model(x.flatten(0, 1).to(dev)).float().softmax(1)[:, :43].view(b, v, 43)
            score, pred = prob.max(2)
            agree = (pred == pred[:, :1]).all(1)
            out += list(zip(y.tolist(), pred[:, 0].tolist(), score[:, 0].tolist(), agree.tolist(),
                            score.min(1).values.tolist()))
        return out

    res = {k: run(v) for k, v in sets.items()}

    def stats(rows, accept):
        acc = [r for r in rows if accept(r)]
        ok = sum(r[0] == r[1] for r in acc)
        return {"accepted_correct": ok / len(rows), "accepted_wrong": (len(acc) - ok) / len(rows),
                "wrong_count": len(acc) - ok, "rejected": 1 - len(acc) / len(rows)}

    points = []
    n = len(res["val_known"])
    for keep in KEEP:
        cut = int((1 - keep) * n)
        t1 = sorted(r[2] for r in res["val_known"])[cut]
        # for agreement: images whose views disagree count as rejected (confidence -1)
        t9 = sorted((r[4] if r[3] else -1.0) for r in res["val_known"])[cut]
        points.append({"keep": keep, "rule": "single view", "threshold": t1,
                       **{k: stats(v, lambda r: r[2] >= t1) for k, v in res.items()}})
        points.append({"keep": keep, "rule": "9 views agree", "threshold": t9,
                       **{k: stats(v, lambda r: r[3] and r[4] >= t9) for k, v in res.items()}})
    # strictest rule: all 9 views agree and each is at least 0.9999 confident
    points.append({"keep": None, "rule": "9 views agree, each confidence >= 0.9999", "threshold": 0.9999,
                   **{k: stats(v, lambda r: r[3] and r[4] >= 0.9999) for k, v in res.items()}})

    (RESULTS / "tta.json").write_text(json.dumps({"model": Path(path).name, "views": len(VIEWS),
                                                   "sizes": {k: len(v) for k, v in res.items()},
                                                   "points": points, "rows": res}))
    for pt in points:
        print(f"keep {pt['keep']} | {pt['rule']} | thr {pt['threshold']:.6f} | " + " | ".join(
            f"{k}: ok {pt[k]['accepted_correct']:.1%} wrong {pt[k]['wrong_count']}" for k in res if k != "val_known"),
            flush=True)


if __name__ == "__main__":
    main()
