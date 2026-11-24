"""Draws docs/accuracy.svg, the summary chart in README.md. Standard library only.

Usage: python bench/chart.py

Numbers come from the result tables: zero-shot rows from results/report.md (430-image set),
trained rows from results/classifiers.md and results/transfer.md (full GTSRB test, mean over runs).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINED, ZERO = "#2a78d6", "#eb6834"
# (label, accuracy %, ms per image, GPU memory, colour)
ROWS = [
    ("Custom large CNN, 4.7M params", 99.5, "1.1 ms", "83 MB", TRAINED),
    ("CLIP B/16, fine-tuned, 86M", 99.5, "5.6 ms", "1.3 GB", TRAINED),
    ("ResNet-18, fine-tuned, 11M", 99.0, "1.7 ms", "185 MB", TRAINED),
    ("Custom small CNN, 0.3M", 99.0, "0.8 ms", "6 MB", TRAINED),
    ("SigLIP 2 So400m, 1.1B", 60.7, "20.7 ms", "2.2 GB", ZERO),
    ("Vision LLM Qwen3-VL-4B, 4.4B", 52.6, "199 ms", "8.6 GB", ZERO),
    ("CLIP L/14, 428M", 39.1, "11.3 ms", "0.8 GB", ZERO),
]
W, ROW, TOP, X0, BAR = 820, 34, 96, 260, 300
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = "font-family=\"-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\""


def main():
    h = TOP + ROW * len(ROWS) + 56
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" {FONT} '
         f'role="img" aria-label="Accuracy, speed and GPU memory of seven traffic sign recognition approaches">',
         f'<rect width="{W}" height="{h}" rx="10" fill="{SURFACE}"/>',
         f'<text x="24" y="34" font-size="17" font-weight="600" fill="{INK}">Traffic sign recognition: accuracy on GTSRB (43 classes)</text>',
         f'<rect x="24" y="50" width="12" height="12" rx="3" fill="{TRAINED}"/>'
         f'<text x="42" y="61" font-size="13" fill="{MUTED}">trained on signs</text>',
         f'<rect x="164" y="50" width="12" height="12" rx="3" fill="{ZERO}"/>'
         f'<text x="182" y="61" font-size="13" fill="{MUTED}">zero-shot, no training</text>']
    for label, x in (("Time per image", 700), ("GPU memory", 796)):
        o.append(f'<text x="{x}" y="{TOP - 12}" font-size="12" fill="{MUTED}" text-anchor="end">{label}</text>')
    bottom = TOP + ROW * len(ROWS)
    for pct in (0, 25, 50, 75, 100):
        x = X0 + BAR * pct / 100
        o.append(f'<line x1="{x}" y1="{TOP - 4}" x2="{x}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{x}" y="{bottom + 18}" font-size="11" fill="{MUTED}" text-anchor="middle">{pct}%</text>')
    for i, (label, acc, ms, mem, colour) in enumerate(ROWS):
        y = TOP + ROW * i
        w = BAR * acc / 100
        o.append(f'<text x="{X0 - 12}" y="{y + 21}" font-size="13" fill="{INK}" text-anchor="end">{label}</text>')
        # square at the baseline, rounded at the data end
        o.append(f'<path d="M{X0} {y + 7} h{w - 4} a4 4 0 0 1 4 4 v12 a4 4 0 0 1 -4 4 h{-(w - 4)} z" fill="{colour}"/>')
        o.append(f'<text x="{X0 + w + 8}" y="{y + 21}" font-size="13" font-weight="600" fill="{INK}">{acc}%</text>')
        o.append(f'<text x="700" y="{y + 21}" font-size="13" fill="{INK}" text-anchor="end">{ms}</text>')
        o.append(f'<text x="796" y="{y + 21}" font-size="13" fill="{INK}" text-anchor="end">{mem}</text>')
    o.append(f'<text x="24" y="{h - 12}" font-size="11" fill="{MUTED}">Trained models: full test split, 12,630 images. '
             f'Zero-shot models: 430-image subset. One image at a time on an RTX 4090 Laptop GPU.</text>')
    o.append("</svg>")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/accuracy.svg").write_text("\n".join(o) + "\n")
    print("docs/accuracy.svg")


if __name__ == "__main__":
    main()
