"""Plain classifier: ResNet-18 fine-tuned on the GTSRB training split.

This is the supervised reference for Part 1: a model that has seen 26,640 labelled signs.
The GTSRB training and test splits do not overlap.

Usage: python bench/classifier.py
"""
import time

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as T

from common import ROOT, load_testset, save_result, start_inference_meter

SIZE = 96
EPOCHS = 6
BATCH = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main():
    dev = "cuda"
    train_tf = T.Compose([
        T.Resize((SIZE, SIZE)),
        T.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(0.4, 0.4, 0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    train = torchvision.datasets.GTSRB(str(ROOT / "data/gtsrb_train"), split="train",
                                       download=True, transform=train_tf)
    loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True,
                                         num_workers=8, drop_last=True)
    print(f"training images: {len(train)}")

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 43)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(loader))
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        total, hit, loss_sum = 0, 0, 0.0
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
            loss_sum += float(loss) * len(y)
        print(f"epoch {epoch + 1}/{EPOCHS}: loss {loss_sum / total:.3f}, train accuracy {hit / total:.1%}")
    train_seconds = time.perf_counter() - t0
    train_vram_mb = torch.cuda.max_memory_allocated() / 2**20

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models/resnet18_gtsrb.pt")

    model.eval()

    @torch.no_grad()
    def predict(path):
        x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
        return int(model(x).argmax(1))

    items = load_testset()
    predict(items[0][0])  # warm-up

    start_inference_meter()
    preds = []
    t0 = time.perf_counter()
    for path, true in items:
        preds.append({"filename": path.name, "true": true, "pred": predict(path)})
    seconds = time.perf_counter() - t0

    save_result("classifier", "ResNet-18, fine-tuned on GTSRB", preds, seconds,
                net=model,
                training=f"{train_seconds / 60:.1f} min", train_vram_peak_mb=train_vram_mb)


if __name__ == "__main__":
    main()
