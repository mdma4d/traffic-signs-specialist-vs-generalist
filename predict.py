"""Recognise a traffic sign in an image, or answer "unknown".

Usage: python predict.py sign1.png [sign2.png ...]
       python predict.py --keep 0.95 sign.png     stricter: fewer false answers, more rejections
       python predict.py --model models/big_cnn_oe_bg.pt sign.png

Input: a photo of a single sign cropped from the frame (with a small margin), as in GTSRB.
The model is trained by bench/reject.py. The default is the custom large CNN trained with an
"unknown" class on all available foreign signs and road background. Runs on CPU as well.

The confidence threshold was fixed at training time so that the model accepts a given share
(--keep) of held-out German signs: 0.99, 0.97 (default), 0.95 or 0.9.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bench"))
from common import load_classes  # noqa: E402
from finetune import MODELS, small_cnn  # noqa: E402


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=True)
    cfg = MODELS[ck["key"]]
    if cfg["timm"] is None:
        model = small_cnn(cfg.get("widths", (32, 64, 128)), ck["num_outputs"])
    else:
        import timm
        kwargs = {"img_size": cfg["size"]} if cfg.get("img_size_arg") else {}
        model = timm.create_model(cfg["timm"], pretrained=False, num_classes=ck["num_outputs"], **kwargs)
    model.load_state_dict(ck["state_dict"])
    tf = T.Compose([T.Resize((ck["size"], ck["size"])), T.ToTensor(), T.Normalize(ck["mean"], ck["std"])])
    return model.to(dev).eval(), tf, ck["thresholds"]


@torch.no_grad()
def recognize(model, tf, threshold, image, dev):
    """Returns (GTSRB class id or None, confidence in the best of the 43 classes)."""
    prob = model(tf(image.convert("RGB")).unsqueeze(0).to(dev)).float().softmax(1)[0, :43]
    score, pred = prob.max(0)
    return (int(pred) if float(score) >= threshold else None), float(score)


def main():
    ap = argparse.ArgumentParser(description="Recognise a traffic sign in an image, or answer \"unknown\".")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "models/big_cnn_oe_all.pt"))
    ap.add_argument("--keep", default="0.97", choices=["0.99", "0.97", "0.95", "0.9"])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tf, thresholds = load(args.model, dev)
    names = {cid: name for cid, _, name in load_classes()}
    for path in args.images:
        cls, score = recognize(model, tf, thresholds[args.keep], Image.open(path), dev)
        answer = "unknown" if cls is None else f"{names[cls]} (class {cls})"
        print(f"{path}: {answer}, confidence {score:.4f}")


if __name__ == "__main__":
    main()
