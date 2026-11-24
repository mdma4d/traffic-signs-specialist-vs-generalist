"""Collects results/*.json of Part 1 into a table. Standard library only.

Usage: python bench/report.py     ->  results/report.md
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["clip", "siglip2", "vlm_number", "vlm_name", "classifier"]
TITLES = {"clip": "CLIP", "siglip2": "SigLIP 2", "vlm_number": "Vision LLM, answers with a number",
          "vlm_name": "Vision LLM, answers with the name", "classifier": "Classifier (ResNet-18)"}
SIZES = [("up to 35 px", 0, 35), ("36-60 px", 36, 60), ("61 px and up", 61, 10**6)]


def main():
    with (ROOT / "data/classes.csv").open(newline="") as f:
        label = {int(r["class_id"]): r["name"] for r in csv.DictReader(f)}
    label[-1] = "(answer not in the list)"
    with (ROOT / "data/testset/labels.csv").open(newline="") as f:
        width = {r["filename"]: int(r["width"]) for r in csv.DictReader(f)}

    results = {n: json.loads((ROOT / f"results/{n}.json").read_text()) for n in ORDER
               if (ROOT / f"results/{n}.json").exists()}
    names = list(results)

    out = ["# Part 1: zero-shot models vs a trained classifier", "",
           "Test set: 430 GTSRB images, 10 for each of the 43 classes. "
           "Time is measured one image at a time on an RTX 4090 Laptop GPU.", "",
           "| Approach | Model | Accuracy | Correct | ms/image | Training |",
           "|---|---|---|---|---|---|"]
    for n in names:
        r = results[n]
        correct = sum(p["true"] == p["pred"] for p in r["preds"])
        out.append(f"| {TITLES.get(n, n)} | {r['model']} | {r['accuracy']:.1%} | "
                   f"{correct}/{r['n']} | {r['ms_per_image']:.1f} | {r.get('training', '')} |")

    out += ["", "## Resources", "",
            "| Approach | Parameters, M | Weights, MB | Peak GPU memory, MB | Peak RAM, MB | Images per second |",
            "|---|---|---|---|---|---|"]
    for n in names:
        r = results[n]
        out.append(f"| {TITLES.get(n, n)} | {r['params_m']:.0f} | {r['weights_mb']:.0f} | "
                   f"{r['vram_peak_mb']:.0f} | {r['ram_peak_mb']:.0f} | {1000 / r['ms_per_image']:.0f} |")

    out += ["", "## Accuracy by image size", "",
            "| Approach | " + " | ".join(t for t, _, _ in SIZES) + " |", "|---|---|---|---|"]
    for n in names:
        cells = []
        for _, lo, hi in SIZES:
            ps = [p for p in results[n]["preds"] if lo <= width[p["filename"]] <= hi]
            cells.append(f"{sum(p['true'] == p['pred'] for p in ps) / len(ps):.0%} of {len(ps)}")
        out.append(f"| {TITLES.get(n, n)} | " + " | ".join(cells) + " |")

    for n in names:
        r = results[n]
        out += ["", f"## {TITLES.get(n, n)}", ""]
        if r.get("invalid_answers"):
            out += [f"Answers not in the list: {r['invalid_answers']}.", ""]

        per_class = Counter()
        for p in r["preds"]:
            per_class[p["true"]] += p["true"] == p["pred"]
        worst = sorted(label.keys() - {-1}, key=lambda c: (per_class[c], c))[:8]
        worst = [c for c in worst if per_class[c] < 10]
        if worst:
            out.append("Classes with the lowest accuracy (correct out of 10):")
            out += [f"- {label[c]}: {per_class[c]}" for c in worst]

        confusions = Counter((p["true"], p["pred"]) for p in r["preds"] if p["true"] != p["pred"])
        if confusions:
            out += ["", "Most frequent errors:"]
            out += [f"- \"{label[t]}\" taken for \"{label[p]}\": {k} times"
                    for (t, p), k in confusions.most_common(6)]

    text = "\n".join(out) + "\n"
    (ROOT / "results/report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
