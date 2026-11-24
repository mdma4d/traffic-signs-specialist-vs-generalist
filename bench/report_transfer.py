"""Collects results/transfer/*.json of Part 3 into a table: the same models on German and Belgian signs.

Usage: python bench/report_transfer.py     ->  results/transfer.md
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZS = {"zs_clip": "CLIP L/14, zero-shot", "zs_siglip2": "SigLIP 2 So400m, zero-shot"}


def main():
    with (ROOT / "data/classes.csv").open(newline="") as f:
        label = {int(r["class_id"]): r["name"] for r in csv.DictReader(f)}

    rows = []
    for p in (ROOT / "results/transfer").glob("*.json"):
        r = json.loads(p.read_text())
        if p.stem in ZS:  # zero-shot: preds already are the Belgian set
            rows.append(dict(title=ZS[p.stem], de=None, be=r["accuracy"], n=r["n"], preds=r["preds"],
                             params=r["params_m"], ms=r["ms_per_image"]))
        elif "accuracy_belgium" in r:
            title = r["title"] + (f", repeat {r['seed']}" if r.get("seed") else "")
            rows.append(dict(title=title, de=r["accuracy_full"], be=r["accuracy_belgium"], n=r["n_belgium"],
                             preds=r["preds_belgium"], params=r["params_m"], ms=r["ms_per_image"]))
    rows.sort(key=lambda r: -r["be"])

    n = rows[0]["n"]
    out = ["# Part 3: signs from another country", "",
           f"Models are trained on German signs (GTSRB) and tested on {n} Belgian images (BelgiumTSC) "
           "without any further training. \"Drop\" is German test accuracy minus Belgian accuracy.", "",
           "| Model | Parameters, M | Germany | Belgium | Errors | Drop, pp |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        err = sum(p["true"] != p["pred"] for p in r["preds"])
        de = f"{r['de']:.2%}" if r["de"] is not None else "—"
        drop = f"{100 * (r['de'] - r['be']):.1f}" if r["de"] is not None else "—"
        out.append(f"| {r['title']} | {r['params']:.1f} | {de} | {r['be']:.2%} | {err} | {drop} |")

    # Per-class accuracy: how many images of each sign and how the models cope
    per = defaultdict(list)
    total = Counter(p["true"] for p in rows[0]["preds"])
    trained = [r for r in rows if r["de"] is not None]
    for r in trained:
        hit = Counter(p["true"] for p in r["preds"] if p["true"] == p["pred"])
        for c in total:
            per[c].append(hit[c] / total[c])
    out += ["", "## Accuracy by sign (trained models)", "",
            "| Sign | Images | Worst model | Mean | Best model |", "|---|---|---|---|---|"]
    for c in sorted(total, key=lambda c: sum(per[c]) / len(per[c])):
        v = per[c]
        out.append(f"| {label[c]} | {total[c]} | {min(v):.0%} | {sum(v) / len(v):.0%} | {max(v):.0%} |")

    conf = Counter()
    for r in trained:
        conf.update((p["true"], p["pred"]) for p in r["preds"] if p["true"] != p["pred"])
    out += ["", f"## Most frequent errors (summed over {len(trained)} trained models)", ""]
    out += [f"- \"{label[t]}\" taken for \"{label[p]}\": {k} times" for (t, p), k in conf.most_common(10)]

    text = "\n".join(out) + "\n"
    (ROOT / "results/transfer.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
