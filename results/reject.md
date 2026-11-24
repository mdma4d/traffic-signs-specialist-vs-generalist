# Part 4: classifier with a reject option

The model answers with a sign or with "unknown". "Correct" and "false" are shares of all images in a set; the rest is rejected. The threshold is chosen on held-out German signs so as to keep the given share of them.

Sets: GTSRB test (12630), Belgian signs of known classes (940), unknown signs of group A, whose types are in the "unknown" training data (526-529), unknown signs of group B, unseen types (457), road background crops (2000), random objects (2000).

| Model | Training | Keep | Germany: correct | Germany: false | Belgium: correct | Belgium: false | Unknown signs A: false | Unknown signs B: false | Road background: false | Objects: false |
|---|---|---|---|---|---|---|---|---|---|---|
| Custom large CNN (from scratch) | confidence threshold | all | 99.0% | 0.97% | 96.0% | 4.04% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom large CNN (from scratch) | confidence threshold | 99% | 96.7% | 0.09% | 85.1% | 0.00% | 14.9% | 10.3% | 4.0% | 4.7% |
| Custom large CNN (from scratch) | confidence threshold | 97% | 92.4% | 0.02% | 68.6% | 0.00% | 9.6% | 0.9% | 0.2% | 0.4% |
| Custom large CNN (from scratch) | confidence threshold | 95% | 89.8% | 0.02% | 62.1% | 0.00% | 6.6% | 0.4% | 0.1% | 0.2% |
| Custom large CNN (from scratch) | confidence threshold | 90% | 84.0% | 0.00% | 50.6% | 0.00% | 1.5% | 0.0% | 0.1% | 0.1% |
| Custom large CNN (from scratch) | "unknown" class | all | 99.4% | 0.55% | 96.3% | 3.72% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom large CNN (from scratch) | "unknown" class | 99% | 97.8% | 0.13% | 75.1% | 0.00% | 0.6% | 2.2% | 4.1% | 5.2% |
| Custom large CNN (from scratch) | "unknown" class | 97% | 94.6% | 0.04% | 61.1% | 0.00% | 0.4% | 0.4% | 0.2% | 0.2% |
| Custom large CNN (from scratch) | "unknown" class | 95% | 89.6% | 0.01% | 47.3% | 0.00% | 0.4% | 0.2% | 0.1% | 0.0% |
| Custom large CNN (from scratch) | "unknown" class | 90% | 80.8% | 0.00% | 33.7% | 0.00% | 0.2% | 0.0% | 0.1% | 0.0% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known | all | 99.2% | 0.75% | 99.7% | 0.32% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known | 99% | 98.5% | 0.29% | 99.5% | 0.11% | 0.6% | 7.4% | 14.1% | 18.8% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known | 97% | 94.1% | 0.04% | 98.1% | 0.11% | 0.6% | 0.9% | 0.7% | 1.4% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known | 95% | 91.0% | 0.02% | 95.4% | 0.11% | 0.6% | 0.7% | 0.2% | 0.7% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known | 90% | 84.2% | 0.00% | 91.4% | 0.00% | 0.6% | 0.2% | 0.1% | 0.3% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known + background | all | 99.4% | 0.58% | 99.6% | 0.43% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known + background | 99% | 96.8% | 0.16% | 98.9% | 0.21% | 0.0% | 2.6% | 0.1% | 0.3% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known + background | 97% | 94.6% | 0.10% | 98.5% | 0.00% | 0.0% | 2.2% | 0.0% | 0.2% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known + background | 95% | 92.3% | 0.05% | 97.8% | 0.00% | 0.0% | 2.2% | 0.0% | 0.1% |
| Custom large CNN (from scratch) | "unknown" class + Belgian known + background | 90% | 85.7% | 0.02% | 94.4% | 0.00% | 0.0% | 1.1% | 0.0% | 0.0% |
| Custom large CNN (from scratch) | same + all foreign types (group B is seen here) | all | 99.3% | 0.68% | 99.7% | 0.32% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom large CNN (from scratch) | same + all foreign types (group B is seen here) | 99% | 97.0% | 0.22% | 98.7% | 0.00% | 0.0% | 0.0% | 0.1% | 0.9% |
| Custom large CNN (from scratch) | same + all foreign types (group B is seen here) | 97% | 94.0% | 0.14% | 97.4% | 0.00% | 0.0% | 0.0% | 0.0% | 0.3% |
| Custom large CNN (from scratch) | same + all foreign types (group B is seen here) | 95% | 90.3% | 0.03% | 96.1% | 0.00% | 0.0% | 0.0% | 0.0% | 0.0% |
| Custom large CNN (from scratch) | same + all foreign types (group B is seen here) | 90% | 84.6% | 0.01% | 93.1% | 0.00% | 0.0% | 0.0% | 0.0% | 0.0% |
| Custom small CNN (from scratch) | confidence threshold | all | 98.9% | 1.15% | 95.6% | 4.36% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom small CNN (from scratch) | confidence threshold | 99% | 97.8% | 0.42% | 91.3% | 2.02% | 28.0% | 35.9% | 15.3% | 19.7% |
| Custom small CNN (from scratch) | confidence threshold | 97% | 95.2% | 0.06% | 84.5% | 0.11% | 10.6% | 9.8% | 4.4% | 4.9% |
| Custom small CNN (from scratch) | confidence threshold | 95% | 90.9% | 0.01% | 71.6% | 0.00% | 5.1% | 3.5% | 1.2% | 1.8% |
| Custom small CNN (from scratch) | confidence threshold | 90% | 83.0% | 0.00% | 50.4% | 0.00% | 1.5% | 1.5% | 0.3% | 0.4% |
| Custom small CNN (from scratch) | "unknown" class | all | 98.4% | 1.62% | 93.8% | 6.17% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom small CNN (from scratch) | "unknown" class | 99% | 97.1% | 0.46% | 60.0% | 0.11% | 0.6% | 2.0% | 8.8% | 12.0% |
| Custom small CNN (from scratch) | "unknown" class | 97% | 94.4% | 0.08% | 44.1% | 0.00% | 0.2% | 1.1% | 2.2% | 3.5% |
| Custom small CNN (from scratch) | "unknown" class | 95% | 90.6% | 0.02% | 32.2% | 0.00% | 0.2% | 0.4% | 0.7% | 1.1% |
| Custom small CNN (from scratch) | "unknown" class | 90% | 80.6% | 0.00% | 15.3% | 0.00% | 0.0% | 0.2% | 0.1% | 0.2% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known | all | 98.8% | 1.17% | 99.5% | 0.53% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known | 99% | 96.9% | 0.19% | 99.3% | 0.00% | 0.6% | 4.2% | 8.5% | 11.1% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known | 97% | 93.7% | 0.04% | 98.3% | 0.00% | 0.6% | 1.8% | 2.9% | 2.9% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known | 95% | 89.7% | 0.01% | 96.3% | 0.00% | 0.6% | 1.3% | 1.7% | 1.1% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known | 90% | 81.3% | 0.00% | 90.5% | 0.00% | 0.6% | 0.4% | 0.9% | 0.4% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known + background | all | 99.0% | 1.05% | 99.6% | 0.43% | 100.0% | 100.0% | 100.0% | 100.0% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known + background | 99% | 98.5% | 0.74% | 99.6% | 0.32% | 0.6% | 12.0% | 0.2% | 4.0% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known + background | 97% | 95.1% | 0.10% | 99.0% | 0.00% | 0.6% | 5.7% | 0.1% | 1.1% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known + background | 95% | 89.6% | 0.02% | 96.2% | 0.00% | 0.6% | 2.2% | 0.0% | 0.4% |
| Custom small CNN (from scratch) | "unknown" class + Belgian known + background | 90% | 82.6% | 0.00% | 91.0% | 0.00% | 0.6% | 1.3% | 0.0% | 0.2% |
| ResNet-18 (torchvision weights) | confidence threshold | all | 98.9% | 1.07% | 96.6% | 3.40% | 100.0% | 100.0% | 100.0% | 100.0% |
| ResNet-18 (torchvision weights) | confidence threshold | 99% | 98.1% | 0.40% | 93.9% | 1.81% | 47.3% | 43.5% | 6.2% | 14.3% |
| ResNet-18 (torchvision weights) | confidence threshold | 97% | 96.3% | 0.13% | 86.7% | 0.32% | 19.7% | 8.5% | 0.4% | 1.8% |
| ResNet-18 (torchvision weights) | confidence threshold | 95% | 93.5% | 0.05% | 74.9% | 0.00% | 8.7% | 5.7% | 0.1% | 0.4% |
| ResNet-18 (torchvision weights) | confidence threshold | 90% | 87.3% | 0.02% | 58.6% | 0.00% | 1.3% | 4.8% | 0.1% | 0.0% |
| ResNet-18 (torchvision weights) | "unknown" class | all | 99.0% | 0.97% | 95.4% | 4.57% | 100.0% | 100.0% | 100.0% | 100.0% |
| ResNet-18 (torchvision weights) | "unknown" class | 99% | 98.3% | 0.58% | 78.2% | 0.64% | 0.9% | 6.1% | 15.1% | 13.8% |
| ResNet-18 (torchvision weights) | "unknown" class | 97% | 94.7% | 0.25% | 53.6% | 0.00% | 0.2% | 2.8% | 0.9% | 1.1% |
| ResNet-18 (torchvision weights) | "unknown" class | 95% | 91.4% | 0.13% | 41.7% | 0.00% | 0.2% | 0.9% | 0.1% | 0.4% |
| ResNet-18 (torchvision weights) | "unknown" class | 90% | 85.5% | 0.06% | 27.0% | 0.00% | 0.0% | 0.2% | 0.0% | 0.1% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known | all | 99.0% | 1.04% | 100.0% | 0.00% | 100.0% | 100.0% | 100.0% | 100.0% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known | 99% | 98.3% | 0.54% | 99.9% | 0.00% | 0.6% | 10.5% | 15.9% | 22.3% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known | 97% | 97.1% | 0.25% | 99.3% | 0.00% | 0.6% | 10.3% | 3.6% | 7.2% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known | 95% | 94.9% | 0.09% | 97.9% | 0.00% | 0.6% | 8.8% | 0.9% | 1.9% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known | 90% | 88.3% | 0.01% | 94.4% | 0.00% | 0.6% | 4.4% | 0.1% | 0.4% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known + background | all | 99.0% | 1.02% | 100.0% | 0.00% | 100.0% | 100.0% | 100.0% | 100.0% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known + background | 99% | 98.3% | 0.51% | 100.0% | 0.00% | 0.0% | 9.0% | 0.1% | 3.5% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known + background | 97% | 96.3% | 0.23% | 98.7% | 0.00% | 0.0% | 5.3% | 0.0% | 0.3% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known + background | 95% | 92.4% | 0.13% | 96.0% | 0.00% | 0.0% | 2.2% | 0.0% | 0.0% |
| ResNet-18 (torchvision weights) | "unknown" class + Belgian known + background | 90% | 83.8% | 0.04% | 87.9% | 0.00% | 0.0% | 0.0% | 0.0% | 0.0% |
| ResNet-18 (torchvision weights) | same + all foreign types (group B is seen here) | all | 98.4% | 1.58% | 99.8% | 0.21% | 100.0% | 100.0% | 100.0% | 100.0% |
| ResNet-18 (torchvision weights) | same + all foreign types (group B is seen here) | 99% | 96.3% | 0.30% | 98.4% | 0.00% | 0.0% | 0.0% | 0.1% | 0.7% |
| ResNet-18 (torchvision weights) | same + all foreign types (group B is seen here) | 97% | 93.2% | 0.18% | 96.3% | 0.00% | 0.0% | 0.0% | 0.0% | 0.1% |
| ResNet-18 (torchvision weights) | same + all foreign types (group B is seen here) | 95% | 90.1% | 0.10% | 95.4% | 0.00% | 0.0% | 0.0% | 0.0% | 0.1% |
| ResNet-18 (torchvision weights) | same + all foreign types (group B is seen here) | 90% | 83.9% | 0.05% | 92.0% | 0.00% | 0.0% | 0.0% | 0.0% | 0.0% |
