"""Zero-shot classification with CLIP and SigLIP 2.

The model compares the image with a text prompt for every class and picks the closest one.

Usage: python bench/zeroshot.py --name clip --model openai/clip-vit-large-patch14
"""
import argparse
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common import ROOT, load_classes, load_testset, save_result, start_inference_meter

PROMPT = "a photo of a '{}' traffic sign"


def as_tensor(out):
    # Depending on the transformers version, get_*_features returns a tensor or an object with pooler_output.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--testset", default="testset", help="dataset folder inside data/: testset or belgium")
    args = ap.parse_args()

    dev = "cuda"
    model = AutoModel.from_pretrained(args.model).to(dev, dtype=torch.float16).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    classes = load_classes()
    texts = [PROMPT.format(en) for _, en, _ in classes]
    if "siglip" in args.model.lower():
        tok = proc(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    else:
        tok = proc(text=texts, padding=True, return_tensors="pt")

    @torch.no_grad()
    def predict(path):
        img = Image.open(path).convert("RGB")
        px = proc(images=img, return_tensors="pt")["pixel_values"].to(dev, dtype=torch.float16)
        f = as_tensor(model.get_image_features(pixel_values=px))
        f = f / f.norm(dim=-1, keepdim=True)
        return int((f @ text_feats.T)[0].argmax())

    with torch.no_grad():
        text_feats = as_tensor(model.get_text_features(**tok.to(dev)))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    items = load_testset(ROOT / "data" / args.testset)
    predict(items[0][0])  # warm-up, so the first call does not distort the timing

    start_inference_meter()
    preds = []
    t0 = time.perf_counter()
    for path, true in items:
        preds.append({"filename": path.name, "true": true, "pred": classes[predict(path)][0]})
    seconds = time.perf_counter() - t0

    save_result(args.name, args.model, preds, seconds, net=model, training="none")


if __name__ == "__main__":
    main()
