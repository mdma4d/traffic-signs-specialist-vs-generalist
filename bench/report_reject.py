"""Collects results/reject/*.json of Part 4 into a table. Standard library only.

Usage: python bench/report_reject.py     ->  results/reject.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODES = {"base": "confidence threshold", "oe": "\"unknown\" class", "oe_mix": "\"unknown\" class + Belgian known",
         "oe_bg": "\"unknown\" class + Belgian known + background",
         "oe_all": "same + all foreign types (group B is seen here)"}
ORDER = ["base", "oe", "oe_mix", "oe_bg", "oe_all"]


def main():
    rs = [json.loads(p.read_text()) for p in (ROOT / "results/reject").glob("*.json")]
    rs.sort(key=lambda r: (r["title"], ORDER.index(r["mode"])))
    # set sizes; group A shrank from 529 to 526 when a mislabelled track was removed before the last runs
    s = {k: "-".join(map(str, sorted({r["sizes"][k] for r in rs if k in r["sizes"]}))) or "0"
         for k in ("gtsrb_test", "belgium_known", "unknown_A", "unknown_B", "background", "objects")}

    out = ["# Part 4: classifier with a reject option", "",
           "The model answers with a sign or with \"unknown\". \"Correct\" and \"false\" are shares of all images in a set; "
           "the rest is rejected. The threshold is chosen on held-out German signs so as to keep the given share of them.", "",
           f"Sets: GTSRB test ({s['gtsrb_test']}), Belgian signs of known classes ({s['belgium_known']}), "
           f"unknown signs of group A, whose types are in the \"unknown\" training data ({s['unknown_A']}), "
           f"unknown signs of group B, unseen types ({s['unknown_B']}), road background crops "
           f"({s['background']}), random objects ({s['objects']}).", "",
           "| Model | Training | Keep | Germany: correct | Germany: false | Belgium: correct | Belgium: false | "
           "Unknown signs A: false | Unknown signs B: false | Road background: false | Objects: false |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        for pt in r["points"]:
            g, b = pt["gtsrb_test"], pt["belgium_known"]
            bg = (f"{pt['background']['accepted_wrong']:.1%} | {pt['objects']['accepted_wrong']:.1%}"
                  if "background" in pt else "n/a | n/a")
            keep = "all" if pt["keep"] == 1 else f"{pt['keep']:.0%}"
            out.append(f"| {r['title']} | {MODES[r['mode']]} | {keep} | {g['accepted_correct']:.1%} | "
                       f"{g['accepted_wrong']:.2%} | {b['accepted_correct']:.1%} | {b['accepted_wrong']:.2%} | "
                       f"{pt['unknown_A']['accepted_wrong']:.1%} | {pt['unknown_B']['accepted_wrong']:.1%} | "
                       f"{bg} |")

    text = "\n".join(out) + "\n"
    (ROOT / "results/reject.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
