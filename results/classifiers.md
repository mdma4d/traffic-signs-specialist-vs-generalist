# Part 2: fine-tuned classifiers compared

Accuracy on the full GTSRB test split (12630 images) and on the 430-image set. Speed and GPU memory: one image at a time on an RTX 4090 Laptop GPU.

| Model | Input, px | Parameters, M | Accuracy | Errors | Accuracy on 430 | ms/image | GPU memory, MB | Weights, MB | Training, min | Training GPU memory, MB |
|---|---|---|---|---|---|---|---|---|---|---|
| Custom large CNN (from scratch) | 64 | 4.7 | 99.51% | 62 | 98.8% | 1.1 | 83 | 18 | 3.4 | 742 |
| CLIP B/16, fine-tuned | 224 | 85.8 | 99.50% | 63 | 99.3% | 5.6 | 1319 | 327 | 9.1 | 9825 |
| ResNet-18 (torchvision weights), repeat 1 | 96 | 11.2 | 99.34% | 83 | 99.1% | 1.7 | 185 | 43 | 0.5 | 495 |
| FastViT-T8 | 224 | 3.3 | 99.31% | 87 | 98.4% | 6.2 | 61 | 13 | 4.4 | 3928 |
| ViT-B/16 | 224 | 85.8 | 99.30% | 88 | 99.3% | 6.3 | 1319 | 327 | 9.1 | 9751 |
| SigLIP 2 B/16, fine-tuned | 224 | 92.9 | 99.29% | 90 | 98.8% | 6.1 | 1429 | 354 | 9.7 | 9984 |
| DINOv2-S, fully fine-tuned | 224 | 21.6 | 99.25% | 95 | 98.6% | 5.8 | 337 | 83 | 5.0 | 6411 |
| ConvNeXt-Tiny | 96 | 27.9 | 99.12% | 111 | 97.9% | 3.3 | 443 | 106 | 1.1 | 1934 |
| DeiT-S/16 | 224 | 21.7 | 99.08% | 116 | 98.8% | 3.8 | 339 | 83 | 3.5 | 4611 |
| ResNet-18 (torchvision weights), repeat 3 | 96 | 11.2 | 99.02% | 124 | 98.4% | 1.6 | 185 | 43 | 0.5 | 495 |
| Custom small CNN (from scratch) | 48 | 0.3 | 98.99% | 127 | 98.1% | 0.8 | 6 | 1 | 0.5 | 187 |
| MobileViT-S | 224 | 5.0 | 98.98% | 129 | 97.7% | 6.0 | 91 | 19 | 5.5 | 8976 |
| ResNet-18 (torchvision weights), repeat 2 | 96 | 11.2 | 98.95% | 133 | 98.6% | 1.6 | 185 | 43 | 0.5 | 495 |
| Swin-T | 224 | 27.6 | 98.84% | 146 | 97.0% | 8.4 | 457 | 105 | 6.6 | 9146 |
| ConvNeXtV2-Tiny | 96 | 27.9 | 98.80% | 152 | 97.2% | 4.3 | 444 | 106 | 2.2 | 3034 |
| ResNet-18 (torchvision weights) | 96 | 11.2 | 98.76% | 156 | 97.2% | 1.7 | 185 | 43 | 0.5 | 495 |
| EfficientFormer-L1 | 224 | 11.4 | 98.69% | 166 | 97.0% | 5.1 | 190 | 44 | 3.8 | 4977 |
| ViT-S/16 | 224 | 21.7 | 98.67% | 168 | 97.9% | 5.1 | 339 | 83 | 3.4 | 4611 |
| ResNet-18 (timm a1 weights) | 96 | 11.2 | 98.25% | 221 | 96.0% | 1.6 | 185 | 43 | 0.5 | 495 |
| EfficientNetV2-S | 96 | 20.2 | 98.06% | 245 | 96.3% | 8.2 | 322 | 77 | 1.7 | 2152 |
| MobileNetV3-Small | 96 | 1.6 | 96.72% | 414 | 93.5% | 2.7 | 25 | 6 | 0.4 | 251 |
| MobileNetV3-Large | 96 | 4.3 | 96.63% | 425 | 96.3% | 3.2 | 67 | 16 | 0.6 | 651 |
| EfficientNet-B0 | 96 | 4.1 | 96.49% | 443 | 92.3% | 3.8 | 65 | 15 | 0.9 | 1087 |
| DINOv3-S, linear head only | 224 | 21.6 | 65.69% | 4333 | 50.9% | 10.0 | 93 | 82 | 1.9 | 461 |
| DINOv2-S, linear head only | 224 | 21.6 | 65.60% | 4345 | 59.1% | 4.1 | 88 | 83 | 1.9 | 542 |
