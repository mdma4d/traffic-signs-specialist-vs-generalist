# Part 1: zero-shot models vs a trained classifier

Test set: 430 GTSRB images, 10 for each of the 43 classes. Time is measured one image at a time on an RTX 4090 Laptop GPU.

| Approach | Model | Accuracy | Correct | ms/image | Training |
|---|---|---|---|---|---|
| CLIP | openai/clip-vit-large-patch14 | 39.1% | 168/430 | 11.3 | none |
| SigLIP 2 | google/siglip2-so400m-patch14-384 | 60.7% | 261/430 | 20.7 | none |
| Vision LLM, answers with a number | Qwen/Qwen3-VL-4B-Instruct | 51.6% | 222/430 | 137.0 | none |
| Vision LLM, answers with the name | Qwen/Qwen3-VL-4B-Instruct | 52.6% | 226/430 | 198.6 | none |
| Classifier (ResNet-18) | ResNet-18, fine-tuned on GTSRB | 99.5% | 428/430 | 1.7 | 0.5 min |

## Resources

| Approach | Parameters, M | Weights, MB | Peak GPU memory, MB | Peak RAM, MB | Images per second |
|---|---|---|---|---|---|
| CLIP | 428 | 816 | 834 | 2893 | 88 |
| SigLIP 2 | 1136 | 2167 | 2209 | 5636 | 48 |
| Vision LLM, answers with a number | 4438 | 8464 | 8631 | 7000 | 7 |
| Vision LLM, answers with the name | 4438 | 8464 | 8616 | 6999 | 5 |
| Classifier (ResNet-18) | 11 | 43 | 198 | 2311 | 586 |

## Accuracy by image size

| Approach | up to 35 px | 36-60 px | 61 px and up |
|---|---|---|---|
| CLIP | 27% of 131 | 46% of 199 | 41% of 100 |
| SigLIP 2 | 38% of 131 | 67% of 199 | 78% of 100 |
| Vision LLM, answers with a number | 36% of 131 | 60% of 199 | 55% of 100 |
| Vision LLM, answers with the name | 34% of 131 | 60% of 199 | 61% of 100 |
| Classifier (ResNet-18) | 98% of 131 | 100% of 199 | 100% of 100 |

## CLIP

Classes with the lowest accuracy (correct out of 10):
- Speed limit 80: 0
- No overtaking: 0
- Right of way at the next intersection: 0
- Priority road: 0
- No vehicles: 0
- No trucks: 0
- Dangerous curve right: 0
- Double curve: 0

Most frequent errors:
- "No trucks" taken for "No overtaking for trucks": 10 times
- "End of all limits" taken for "No overtaking for trucks": 7 times
- "Pedestrians" taken for "Children crossing": 6 times
- "End of no overtaking" taken for "No overtaking for trucks": 5 times
- "Speed limit 80" taken for "End of speed limit 80": 4 times
- "No overtaking" taken for "No overtaking for trucks": 4 times

## SigLIP 2

Classes with the lowest accuracy (correct out of 10):
- Right of way at the next intersection: 0
- Dangerous curve right: 0
- End of all limits: 0
- Go straight or right: 0
- Keep left: 0
- End of no overtaking for trucks: 0
- End of no overtaking: 1
- Speed limit 80: 2

Most frequent errors:
- "Go straight or right" taken for "Go straight or left": 10 times
- "End of all limits" taken for "No entry": 7 times
- "Dangerous curve right" taken for "Dangerous curve left": 6 times
- "No overtaking for trucks" taken for "No overtaking": 5 times
- "Ahead only" taken for "Turn left ahead": 4 times
- "Keep left" taken for "Go straight or left": 4 times

## Vision LLM, answers with a number

Classes with the lowest accuracy (correct out of 10):
- End of speed limit 80: 0
- Right of way at the next intersection: 0
- Priority road: 0
- No vehicles: 0
- Road narrows on the right: 0
- End of all limits: 0
- Ahead only: 0
- Keep right: 0

Most frequent errors:
- "Priority road" taken for "Speed limit 30": 10 times
- "Ahead only" taken for "Go straight or right": 10 times
- "Dangerous curve left" taken for "Dangerous curve right": 8 times
- "Slippery road" taken for "Wild animals": 6 times
- "End of all limits" taken for "Speed limit 30": 6 times
- "Keep left" taken for "No overtaking": 6 times

## Vision LLM, answers with the name

Classes with the lowest accuracy (correct out of 10):
- End of speed limit 80: 0
- Right of way at the next intersection: 0
- Priority road: 0
- No vehicles: 0
- Road narrows on the right: 0
- End of all limits: 0
- Ahead only: 0
- Keep right: 0

Most frequent errors:
- "Ahead only" taken for "Go straight or right": 10 times
- "Dangerous curve right" taken for "Dangerous curve left": 8 times
- "Keep left" taken for "No overtaking": 8 times
- "Slippery road" taken for "Wild animals": 7 times
- "Road narrows on the right" taken for "Dangerous curve left": 7 times
- "Speed limit 120" taken for "Speed limit 30": 6 times

## Classifier (ResNet-18)

Classes with the lowest accuracy (correct out of 10):
- End of speed limit 80: 9
- Roundabout: 9

Most frequent errors:
- "End of speed limit 80" taken for "End of no overtaking": 1 times
- "Roundabout" taken for "Ahead only": 1 times
