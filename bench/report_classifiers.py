"""Collects results/classifiers/*.json of Part 2 into a table. Standard library only.

Usage: python bench/report_classifiers.py     ->  results/classifiers.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    rs = [json.loads(p.read_text()) for p in (ROOT / "results/classifiers").glob("*.json")]
    rs.sort(key=lambda r: (-r["accuracy_full"], r["ms_per_image"]))

    out = ["# Part 2: fine-tuned classifiers compared", "",
           f"Accuracy on the full GTSRB test split ({rs[0]['n_full']} images) and on the 430-image set. "
           "Speed and GPU memory: one image at a time on an RTX 4090 Laptop GPU.", "",
           "| Model | Input, px | Parameters, M | Accuracy | Errors | Accuracy on 430 | ms/image | "
           "GPU memory, MB | Weights, MB | Training, min | Training GPU memory, MB |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        title = r["title"] + (f", repeat {r['seed']}" if r.get("seed") else "")
        out.append(f"| {title} | {r['input_size']} | {r['params_m']:.1f} | {r['accuracy_full']:.2%} | "
                   f"{r['errors_full']} | {r['accuracy']:.1%} | {r['ms_per_image']:.1f} | "
                   f"{r['vram_peak_mb']:.0f} | {r['weights_mb']:.0f} | {r['train_minutes']:.1f} | "
                   f"{r['train_vram_peak_mb']:.0f} |")

    text = "\n".join(out) + "\n"
    (ROOT / "results/classifiers.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
