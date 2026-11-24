"""Classifier with a reject option: answers "this sign" or "unknown".

Usage:  python bench/reject.py <base|oe|oe_mix|oe_bg|oe_all> [model key from finetune.MODELS, default resnet18_tv]
Output: results/reject/<key>_<mode>.json and models/<key>_<mode>.pt

Modes (same network and training recipe):
  base   - plain 43-class classifier. Rejects when its confidence is below a threshold.
  oe     - same classifier with a 44th "unknown" output trained on Belgian signs of group A, which do
           not exist in GTSRB. Rejection uses the same confidence: probability of the best of 43 classes.
  oe_mix - like oe, plus Belgian signs of known classes in training (BelgiumTSC training split).
           Without them the network learns the shortcut "Belgian photo = unknown".
  oe_bg  - like oe_mix, plus road-scene crops without signs in "unknown" (GTSDB, frames 0-399).
  oe_all - like oe_bg, plus the sign types of group B in "unknown", i.e. all 36 foreign types.
           This is the variant to use. No unseen sign types are left to test it on: its unknown_B
           column holds other photos of types it has seen.

If data/background exists (scripts/make_background.py), images without signs are evaluated too:
road-scene crops from frames not used in training, and random objects (CIFAR-100).

The threshold is never tuned on evaluation data. It is chosen on held-out German signs (tracks kept
out of training) so that the model accepts a given share of them (99%, 97%, 95%, 90%).
Evaluation: GTSRB test (12,630), Belgian signs of known classes (940), unknown signs of group A
(same types, other photos) and group B (types the model has never seen, except in oe_all).
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as T

from common import RESULTS, ROOT, load_testset
from finetune import BATCH, MODELS, build

UNKNOWN = 43          # index of the "unknown" output
OE_REPEAT = 4         # how many times the scarce unknown signs are repeated in training
KEEP = [0.99, 0.97, 0.95, 0.90]  # share of held-out German signs the model must accept


class Photos(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def unknown_items(part, group):
    folder = ROOT / "data/belgium_unknown" / part
    with (folder / "labels.csv").open(newline="") as f:
        return [(folder / "images" / r["filename"], UNKNOWN) for r in csv.DictReader(f) if r["group"] == group]


def main():
    mode = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "resnet18_tv"
    assert mode in ("base", "oe", "oe_mix", "oe_bg", "oe_all")
    cfg = MODELS[key]
    size, epochs = cfg["size"], cfg.get("epochs", 6)
    dev = "cuda"
    torch.manual_seed(0)
    random.seed(0)

    model, (mean, std) = build(cfg, 43 if mode == "base" else 44)
    model.to(dev)
    train_tf = T.Compose([T.Resize((size, size)), T.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                          T.ColorJitter(0.4, 0.4, 0.3), T.ToTensor(), T.Normalize(mean, std)])
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])

    # German signs. Photos come in tracks of 30 frames of the same physical sign; every tenth track is
    # held out as a whole, otherwise the held-out part nearly duplicates training and the threshold
    # comes out too lenient.
    root = str(ROOT / "data/gtsrb_train")
    samples = torchvision.datasets.GTSRB(root, split="train", download=True)._samples
    is_val = lambda path: int(Path(path).name.split("_")[0]) % 10 == 9
    train_items = [(p, y) for p, y in samples if not is_val(p)]
    val_known = [(p, y) for p, y in samples if is_val(p)]
    test_known = [(p, y) for p, y in torchvision.datasets.GTSRB(root, split="test", download=True)._samples]

    unk = unknown_items("train", "A")
    random.shuffle(unk)
    cut = int(0.7 * len(unk))
    unk_train, val_unknown = unk[:cut], unk[cut:]
    if mode != "base":
        train_items = train_items + unk_train * OE_REPEAT
    if mode in ("oe_mix", "oe_bg", "oe_all"):
        train_items = train_items + load_testset(ROOT / "data/belgium_train") * OE_REPEAT
    if mode in ("oe_bg", "oe_all"):
        train_items = train_items + load_testset(ROOT / "data/background/train")
    if mode == "oe_all":
        train_items = train_items + unknown_items("train", "B") * OE_REPEAT

    loader = torch.utils.data.DataLoader(Photos(train_items, train_tf), batch_size=BATCH, shuffle=True,
                                         num_workers=16, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"[{key} {mode}] epoch {epoch + 1}/{epochs}", flush=True)
    train_minutes = (time.perf_counter() - t0) / 60
    del loader
    model.eval()

    @torch.no_grad()
    def run(items):
        """Per image: (true class, answer among the 43 classes, confidence in that answer)."""
        out = []
        for x, y in torch.utils.data.DataLoader(Photos(items, eval_tf), batch_size=256, num_workers=16):
            prob = model(x.to(dev)).float().softmax(1)[:, :43]
            score, pred = prob.max(1)
            out += list(zip(y.tolist(), pred.tolist(), score.tolist()))
        return out

    sets = {
        "val_known": run(val_known),
        "val_unknown": run(val_unknown),
        "gtsrb_test": run(test_known),
        "belgium_known": run(load_testset(ROOT / "data/belgium")),
        "unknown_A": run(unknown_items("test", "A")),
        "unknown_B": run(unknown_items("test", "B")),
    }
    if (ROOT / "data/background/test/labels.csv").exists():
        sets["background"] = run(load_testset(ROOT / "data/background/test"))
        sets["objects"] = run(load_testset(ROOT / "data/background/objects"))

    def stats(rows, thr):
        n = len(rows)
        acc = [r for r in rows if r[2] >= thr]
        ok = sum(t == p for t, p, _ in acc)
        return {"accepted_correct": ok / n, "accepted_wrong": (len(acc) - ok) / n, "rejected": 1 - len(acc) / n}

    points = [{"keep": 1.0, "threshold": 0.0, **{k: stats(v, 0.0) for k, v in sets.items()}}]
    val_scores = sorted(s for _, _, s in sets["val_known"])
    for keep in KEEP:
        thr = val_scores[int((1 - keep) * len(val_scores))]
        points.append({"keep": keep, "threshold": thr, **{k: stats(v, thr) for k, v in sets.items()}})

    result = {"name": f"{key}_{mode}", "mode": mode, "title": cfg["title"], "train_minutes": train_minutes,
              "sizes": {k: len(v) for k, v in sets.items()}, "points": points,
              "scores": {k: [[t, p, round(s, 6)] for t, p, s in v] for k, v in sets.items()}}
    out = RESULTS / "reject"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}_{mode}.json").write_text(json.dumps(result))
    # Weights plus everything predict.py needs to recognise a single image
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"key": key, "mode": mode, "size": size, "mean": mean, "std": std,
                "num_outputs": 43 if mode == "base" else 44,
                "thresholds": {str(pt["keep"]): pt["threshold"] for pt in points},
                "state_dict": model.state_dict()}, ROOT / "models" / f"{key}_{mode}.pt")
    for pt in points:
        print(f"[{key} {mode}] keep {pt['keep']:.0%}, threshold {pt['threshold']:.4f}: " + "; ".join(
            f"{k} correct {pt[k]['accepted_correct']:.1%} false {pt[k]['accepted_wrong']:.1%} rejected {pt[k]['rejected']:.1%}"
            for k in sets if not k.startswith("val_")), flush=True)


if __name__ == "__main__":
    main()
