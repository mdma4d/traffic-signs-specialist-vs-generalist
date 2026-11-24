"""Open vision LLM: show the image and the list of options, ask for one.

Two answer modes:
  number - options are numbered and the model answers with a number;
  name   - the model writes the option text (it tends to mix up numbers in a long list).

Usage: python bench/vlm.py --name vlm_name --answer name --model Qwen/Qwen3-VL-4B-Instruct
"""
import argparse
import difflib
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import load_classes, load_testset, save_result, start_inference_meter

MIN_SIDE = 224  # signs in the test set are small (from 26 px); upscale before feeding the model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vlm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--answer", choices=["number", "name"], default="number")
    args = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(args.model).to("cuda", dtype=torch.bfloat16).eval()
    proc = AutoProcessor.from_pretrained(args.model)

    classes = load_classes()
    ids = {cid for cid, _, _ in classes}
    by_name = {en.lower(): cid for cid, en, _ in classes}
    if args.answer == "number":
        options = "\n".join(f"{cid}. {en}" for cid, en, _ in classes)
        ending = "Answer with the option number only."
    else:
        options = "\n".join(f"- {en}" for _, en, _ in classes)
        ending = "Answer with the exact text of one option and nothing else."
    question = ("This is a cropped photo of a German traffic sign. Which sign is it?\n"
                f"Options:\n{options}\n{ending}")

    def parse(text):
        if args.answer == "number":
            m = re.search(r"\d+", text)
            pred = int(m.group()) if m else -1
            return pred if pred in ids else -1
        key = text.strip().strip("-. \"'").lower()
        if key in by_name:
            return by_name[key]
        close = difflib.get_close_matches(key, list(by_name), n=1, cutoff=0.8)
        return by_name[close[0]] if close else -1

    @torch.no_grad()
    def predict(path):
        img = Image.open(path).convert("RGB")
        scale = MIN_SIDE / min(img.size)
        if scale > 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": question}]}]
        inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=5 if args.answer == "number" else 24, do_sample=False)
        text = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse(text), text.strip()

    items = load_testset()
    print("sample model answer:", repr(predict(items[0][0])[1]))  # doubles as warm-up

    start_inference_meter()
    preds = []
    t0 = time.perf_counter()
    for i, (path, true) in enumerate(items):
        pred, raw = predict(path)
        preds.append({"filename": path.name, "true": true, "pred": pred, "raw": raw})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}")
    seconds = time.perf_counter() - t0

    invalid = sum(p["pred"] == -1 for p in preds)
    save_result(args.name, args.model, preds, seconds, net=model, training="none", invalid_answers=invalid,
                answer_mode=args.answer)


if __name__ == "__main__":
    main()
