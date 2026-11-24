# Part 3: signs from another country

Models are trained on German signs (GTSRB) and tested on 940 Belgian images (BelgiumTSC) without any further training. "Drop" is German test accuracy minus Belgian accuracy.

| Model | Parameters, M | Germany | Belgium | Errors | Drop, pp |
|---|---|---|---|---|---|
| CLIP B/16, fine-tuned | 85.8 | 99.50% | 98.19% | 17 | 1.3 |
| Custom large CNN (from scratch), repeat 2 | 4.7 | 99.48% | 97.77% | 21 | 1.7 |
| Custom large CNN (from scratch), repeat 1 | 4.7 | 99.49% | 97.66% | 22 | 1.8 |
| EfficientFormer-L1 | 11.4 | 98.69% | 97.02% | 28 | 1.7 |
| Custom large CNN (from scratch) | 4.7 | 99.51% | 96.81% | 30 | 2.7 |
| DeiT-S/16 | 21.7 | 99.08% | 96.81% | 30 | 2.3 |
| ViT-S/16 | 21.7 | 98.67% | 96.60% | 32 | 2.1 |
| SigLIP 2 B/16, fine-tuned | 92.9 | 99.29% | 96.60% | 32 | 2.7 |
| ConvNeXt-Tiny | 27.9 | 99.12% | 96.38% | 34 | 2.7 |
| Custom small CNN (from scratch), repeat 3 | 0.3 | 98.92% | 95.96% | 38 | 3.0 |
| ResNet-18 (torchvision weights) | 11.2 | 98.76% | 95.96% | 38 | 2.8 |
| Swin-T | 27.6 | 98.84% | 95.74% | 40 | 3.1 |
| MobileViT-S | 5.0 | 98.85% | 95.64% | 41 | 3.2 |
| Custom large CNN (from scratch), repeat 3 | 4.7 | 99.42% | 95.32% | 44 | 4.1 |
| ResNet-18 (torchvision weights), repeat 2 | 11.2 | 98.95% | 95.32% | 44 | 3.6 |
| Custom small CNN (from scratch), repeat 1 | 0.3 | 99.10% | 95.11% | 46 | 4.0 |
| ViT-B/16 | 85.8 | 99.30% | 94.89% | 48 | 4.4 |
| ConvNeXtV2-Tiny | 27.9 | 98.80% | 94.57% | 51 | 4.2 |
| Custom small CNN (from scratch) | 0.3 | 98.99% | 94.36% | 53 | 4.6 |
| Custom small CNN (from scratch), repeat 2 | 0.3 | 98.95% | 94.26% | 54 | 4.7 |
| ResNet-18 (torchvision weights), repeat 3 | 11.2 | 99.02% | 93.94% | 57 | 5.1 |
| ResNet-18 (torchvision weights), repeat 1 | 11.2 | 99.34% | 93.83% | 58 | 5.5 |
| DINOv2-S, fully fine-tuned | 21.6 | 99.37% | 93.30% | 63 | 6.1 |
| EfficientNetV2-S | 20.2 | 98.22% | 92.77% | 68 | 5.5 |
| FastViT-T8 | 3.3 | 99.31% | 91.70% | 78 | 7.6 |
| ResNet-18 (timm a1 weights) | 11.2 | 98.25% | 91.17% | 83 | 7.1 |
| MobileNetV3-Small | 1.6 | 96.72% | 90.96% | 85 | 5.8 |
| EfficientNet-B0 | 4.1 | 96.49% | 86.06% | 131 | 10.4 |
| MobileNetV3-Large | 4.3 | 96.63% | 85.32% | 138 | 11.3 |
| SigLIP 2 So400m, zero-shot | 1136.0 | — | 75.43% | 231 | — |
| DINOv3-S, linear head only | 21.6 | 65.69% | 74.47% | 240 | -8.8 |
| DINOv2-S, linear head only | 21.6 | 65.60% | 67.87% | 302 | -2.3 |
| CLIP L/14, zero-shot | 427.6 | — | 40.64% | 558 | — |

## Accuracy by sign (trained models)

| Sign | Images | Worst model | Mean | Best model |
|---|---|---|---|---|
| Bicycles crossing | 12 | 0% | 7% | 42% |
| Slippery road | 7 | 0% | 64% | 100% |
| Dangerous curve left | 6 | 0% | 66% | 100% |
| Children crossing | 90 | 2% | 72% | 98% |
| Dangerous curve right | 12 | 0% | 82% | 100% |
| No vehicles | 51 | 78% | 90% | 100% |
| General danger | 39 | 59% | 93% | 100% |
| Bumpy road | 6 | 0% | 94% | 100% |
| Road work | 28 | 46% | 96% | 100% |
| Roundabout | 31 | 58% | 96% | 100% |
| Double curve | 3 | 0% | 96% | 100% |
| No trucks | 3 | 0% | 96% | 100% |
| Right of way at the next intersection | 183 | 60% | 96% | 100% |
| Ahead only | 9 | 67% | 97% | 100% |
| No overtaking | 86 | 53% | 98% | 100% |
| Stop | 45 | 89% | 98% | 100% |
| Priority road | 105 | 91% | 99% | 100% |
| No entry | 61 | 95% | 99% | 100% |
| Yield | 163 | 96% | 100% | 100% |

## Most frequent errors (summed over 31 trained models)

- "Children crossing" taken for "Right of way at the next intersection": 365 times
- "Children crossing" taken for "Ice or snow": 169 times
- "Bicycles crossing" taken for "Children crossing": 153 times
- "Children crossing" taken for "Road work": 139 times
- "Right of way at the next intersection" taken for "Double curve": 105 times
- "No vehicles" taken for "No overtaking": 65 times
- "Bicycles crossing" taken for "Pedestrians": 62 times
- "Bicycles crossing" taken for "Road work": 47 times
- "Children crossing" taken for "Pedestrians": 45 times
- "Right of way at the next intersection" taken for "General danger": 41 times
