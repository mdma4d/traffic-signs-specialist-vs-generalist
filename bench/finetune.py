"""Fine-tune one classifier on GTSRB and measure accuracy, speed and resources.

Usage:  python bench/finetune.py <key from MODELS>
Output: results/classifiers/<key>.json

Accuracy is measured twice: on the full GTSRB test split (12,630 images; on 430 images almost
every model scores 99%+ and they cannot be told apart) and on our 430-image test set.
Speed and GPU memory are measured on the 430-image set, one image at a time, as in Part 1.

Environment variables: SEED=1,2,... repeats a run with another seed (output <key>_seed<N>.json);
RESULTS_SUBDIR=transfer writes to results/transfer/ instead of results/classifiers/.
"""
import json
import os
import resource
import sys
import time

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as T

from common import RESULTS, ROOT, load_testset, model_stats, start_inference_meter

# size is the input side in pixels. Conv nets do fine at 96 px; transformers and hybrids need the
# size they were pretrained at (224 px) or they lose accuracy noticeably.
# frozen=True freezes the backbone and trains only the classification head.
MODELS = {
    "small_cnn":         dict(title="Custom small CNN (from scratch)", timm=None, size=48, lr=2e-3, epochs=15),
    # same design as the small one but wider and deeper: 4 blocks instead of 3, 64-512 channels, 64 px input
    "big_cnn":           dict(title="Custom large CNN (from scratch)", timm=None, size=64, lr=2e-3, epochs=30,
                              widths=(64, 128, 256, 512)),
    "mobilenetv3_small": dict(title="MobileNetV3-Small", timm="mobilenetv3_small_100", size=96, lr=1e-3),
    "mobilenetv3_large": dict(title="MobileNetV3-Large", timm="mobilenetv3_large_100", size=96, lr=1e-3),
    "resnet18":          dict(title="ResNet-18 (timm a1 weights)", timm="resnet18", size=96, lr=1e-3),
    "resnet18_tv":       dict(title="ResNet-18 (torchvision weights)", timm="resnet18.tv_in1k", size=96, lr=1e-3),
    "efficientnet_b0":   dict(title="EfficientNet-B0", timm="efficientnet_b0", size=96, lr=1e-3),
    "efficientnetv2_s":  dict(title="EfficientNetV2-S", timm="tf_efficientnetv2_s", size=96, lr=1e-3),
    "mobilevit_s":       dict(title="MobileViT-S", timm="mobilevit_s", size=224, lr=1e-3),
    "efficientformer":   dict(title="EfficientFormer-L1", timm="efficientformer_l1", size=224, lr=5e-4),
    "fastvit_t8":        dict(title="FastViT-T8", timm="fastvit_t8", size=224, lr=1e-3),
    "convnext_tiny":     dict(title="ConvNeXt-Tiny", timm="convnext_tiny", size=96, lr=2e-4),
    "convnextv2_tiny":   dict(title="ConvNeXtV2-Tiny", timm="convnextv2_tiny", size=96, lr=2e-4),
    "vit_small":         dict(title="ViT-S/16", timm="vit_small_patch16_224", size=224, lr=1e-4),
    "vit_base":          dict(title="ViT-B/16", timm="vit_base_patch16_224", size=224, lr=1e-4),
    "deit_small":        dict(title="DeiT-S/16", timm="deit_small_patch16_224", size=224, lr=1e-4),
    "swin_tiny":         dict(title="Swin-T", timm="swin_tiny_patch4_window7_224", size=224, lr=1e-4),
    "dinov2_small_head": dict(title="DINOv2-S, linear head only",
                              timm="vit_small_patch14_dinov2.lvd142m", size=224, lr=1e-3,
                              frozen=True, img_size_arg=True),
    "dinov2_small_ft":   dict(title="DINOv2-S, fully fine-tuned", timm="vit_small_patch14_dinov2.lvd142m",
                              size=224, lr=5e-5, img_size_arg=True),
    "dinov3_small_head": dict(title="DINOv3-S, linear head only",
                              timm="vit_small_patch16_dinov3.lvd1689m", size=224, lr=1e-3,
                              frozen=True),
    "siglip2_base_ft":   dict(title="SigLIP 2 B/16, fine-tuned", timm="vit_base_patch16_siglip_224.v2_webli",
                              size=224, lr=3e-5),
    "clip_base_ft":      dict(title="CLIP B/16, fine-tuned", timm="vit_base_patch16_clip_224.openai",
                              size=224, lr=3e-5),
}
BATCH = 128


def small_cnn(widths=(32, 64, 128), num_classes=43):
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                             nn.MaxPool2d(2))
    blocks = [block(cin, cout) for cin, cout in zip((3,) + tuple(widths), widths)]
    return nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                         nn.Linear(widths[-1], num_classes))


def build(cfg, num_classes=43):
    if cfg["timm"] is None:
        return small_cnn(cfg.get("widths", (32, 64, 128)), num_classes), ([0.5] * 3, [0.5] * 3)
    import timm  # imported lazily so that predict.py with a custom CNN does not need timm
    kwargs = {"img_size": cfg["size"]} if cfg.get("img_size_arg") else {}
    model = timm.create_model(cfg["timm"], pretrained=True, num_classes=num_classes, **kwargs)
    dc = timm.data.resolve_data_config({}, model=model)
    if cfg.get("frozen"):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    return model, (list(dc["mean"]), list(dc["std"]))


def main():
    key = sys.argv[1]
    cfg = MODELS[key]
    size, epochs, frozen = cfg["size"], cfg.get("epochs", 6), cfg.get("frozen", False)
    dev = "cuda"
    seed = int(os.environ.get("SEED", "0"))  # SEED=1,2,... repeats a run to estimate run-to-run variance
    torch.manual_seed(seed)
    out_name = key if seed == 0 else f"{key}_seed{seed}"

    model, (mean, std) = build(cfg)
    model.to(dev)

    train_tf = T.Compose([
        T.Resize((size, size)),
        T.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(0.4, 0.4, 0.3),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])

    root = str(ROOT / "data/gtsrb_train")
    train = torchvision.datasets.GTSRB(root, split="train", download=True, transform=train_tf)
    test = torchvision.datasets.GTSRB(root, split="test", download=True, transform=eval_tf)
    loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=16,
                                         drop_last=True, persistent_workers=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.eval() if frozen else model.train()
        total, hit = 0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += len(y)
            hit += int((out.argmax(1) == y).sum())
        print(f"[{key}] epoch {epoch + 1}/{epochs}: train accuracy {hit / total:.1%}", flush=True)
    train_seconds = time.perf_counter() - t0
    train_vram_mb = torch.cuda.max_memory_allocated() / 2**20
    del loader

    model.eval()

    # Accuracy on the full GTSRB test split
    full_hit, full_n = 0, 0
    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(test, batch_size=256, num_workers=16):
            full_hit += int((model(x.to(dev)).argmax(1).cpu() == y).sum())
            full_n += len(y)

    # Accuracy, speed and GPU memory on the 430-image set, one image at a time
    @torch.no_grad()
    def predict(path):
        x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
        return int(model(x).argmax(1))

    items = load_testset()
    for path, _ in items[:5]:
        predict(path)  # warm-up
    start_inference_meter()
    t0 = time.perf_counter()
    preds = [{"filename": p.name, "true": true, "pred": predict(p)} for p, true in items]
    seconds = time.perf_counter() - t0

    # Signs from another country (BelgiumTSC), if the set has been prepared with scripts/make_belgium.py.
    # The model still chooses among all 43 GTSRB classes.
    belgium = {}
    if (ROOT / "data/belgium/labels.csv").exists():
        be = [{"filename": p.name, "true": true, "pred": predict(p)}
              for p, true in load_testset(ROOT / "data/belgium")]
        belgium = {"accuracy_belgium": sum(p["true"] == p["pred"] for p in be) / len(be),
                   "n_belgium": len(be), "preds_belgium": be}

    result = {
        "name": out_name,
        "seed": seed,
        "title": cfg["title"],
        "model": cfg["timm"] or "custom architecture",
        "input_size": size,
        "frozen_backbone": frozen,
        "epochs": epochs,
        "accuracy_full": full_hit / full_n,
        "errors_full": full_n - full_hit,
        "n_full": full_n,
        "accuracy": sum(p["true"] == p["pred"] for p in preds) / len(preds),
        "n": len(preds),
        "ms_per_image": 1000 * seconds / len(preds),
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
        "train_vram_peak_mb": train_vram_mb,
        "train_minutes": train_seconds / 60,
        "ram_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        **model_stats(model),
        **belgium,
        "preds": preds,
    }
    # RESULTS_SUBDIR=transfer: retraining for the Belgian check, kept apart from the main comparison.
    out = RESULTS / os.environ.get("RESULTS_SUBDIR", "classifiers")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{out_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"[{key}] RESULT: {result['accuracy_full']:.2%} on {full_n} images ({result['errors_full']} errors), "
          f"{result['accuracy']:.1%} on 430, {result['ms_per_image']:.1f} ms/image, "
          f"GPU memory {result['vram_peak_mb']:.0f} MB, training {result['train_minutes']:.1f} min"
          + (f", Belgium {result['accuracy_belgium']:.2%}" if belgium else ""), flush=True)


if __name__ == "__main__":
    main()
