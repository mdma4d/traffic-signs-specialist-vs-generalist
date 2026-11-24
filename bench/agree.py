"""Two-model agreement: an answer is accepted only if two different models name the same sign.

Usage:  python bench/agree.py <mode: oe_bg|oe_all> [key1 key2]   (default: big_cnn resnet18_tv)
Needs:  models/<key>_<mode>.pt for both keys (saved by bench/reject.py)
Output: results/agree_<mode>.json

The pair's confidence is the lower of the two; different answers mean rejection. Compared with each
model on its own at the same share of accepted held-out German signs.
"""
import json
import sys
from pathlib import Path

import torch
import torchvision

from common import RESULTS, ROOT, load_testset
from reject import KEEP, Photos, unknown_items

sys.path.insert(0, str(ROOT))
from predict import load  # noqa: E402


def main():
    mode = sys.argv[1]
    keys = sys.argv[2:4] if len(sys.argv) > 3 else ["big_cnn", "resnet18_tv"]
    dev = "cuda"
    root = str(ROOT / "data/gtsrb_train")
    samples = torchvision.datasets.GTSRB(root, split="train", download=True)._samples
    sets = {
        "val_known": [(p, y) for p, y in samples if int(Path(p).name.split("_")[0]) % 10 == 9],
        "gtsrb_test": list(torchvision.datasets.GTSRB(root, split="test", download=True)._samples),
        "belgium_known": load_testset(ROOT / "data/belgium"),
        "unknown_A": unknown_items("test", "A"),
        "unknown_B": unknown_items("test", "B"),
        "background": load_testset(ROOT / "data/background/test"),
        "objects": load_testset(ROOT / "data/background/objects"),
    }

    @torch.no_grad()
    def run(key):
        model, tf, _ = load(ROOT / "models" / f"{key}_{mode}.pt", dev)
        out = {}
        for name, items in sets.items():
            rows = []
            for x, y in torch.utils.data.DataLoader(Photos(items, tf), batch_size=256, num_workers=16):
                score, pred = model(x.to(dev)).float().softmax(1)[:, :43].max(1)
                rows += list(zip(y.tolist(), pred.tolist(), score.tolist()))
            out[name] = rows
        return out

    a, b = run(keys[0]), run(keys[1])
    # pair: (true class, first model's answer, pair confidence or -1 on disagreement)
    pair = {k: [(t, p1, min(s1, s2) if p1 == p2 else -1.0) for (t, p1, s1), (_, p2, s2) in zip(a[k], b[k])]
            for k in sets}

    def stats(rows, thr):
        acc = [r for r in rows if r[2] >= thr]
        ok = sum(r[0] == r[1] for r in acc)
        return {"accepted_correct": ok / len(rows), "accepted_wrong": (len(acc) - ok) / len(rows),
                "wrong_count": len(acc) - ok, "rejected": 1 - len(acc) / len(rows)}

    points = []
    for keep in KEEP:
        for rule, res in ((keys[0], a), (keys[1], b), ("both agree", pair)):
            thr = sorted(r[2] for r in res["val_known"])[int((1 - keep) * len(res["val_known"]))]
            points.append({"keep": keep, "rule": rule, "threshold": thr, **{k: stats(v, thr) for k, v in res.items()}})
    # strictest rule: both agree and both are at least 0.9999 confident
    points.append({"keep": None, "rule": "both agree, both confidences >= 0.9999", "threshold": 0.9999,
                   **{k: stats(v, 0.9999) for k, v in pair.items()}})

    (RESULTS / f"agree_{mode}.json").write_text(json.dumps(
        {"mode": mode, "models": keys, "sizes": {k: len(v) for k, v in sets.items()}, "points": points, "pair": pair}))
    for pt in points:
        print(f"keep {pt['keep']} | {pt['rule']} | thr {pt['threshold']:.6f} | " + " | ".join(
            f"{k}: ok {pt[k]['accepted_correct']:.1%} wrong {pt[k]['wrong_count']}" for k in sets if k != "val_known"),
            flush=True)


if __name__ == "__main__":
    main()
