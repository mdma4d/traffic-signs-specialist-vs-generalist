# Traffic sign recognition: comparing approaches

Run dates: 23–24 November 2026. Part 1: zero-shot approaches versus a classifier. Part 2: a comparison of 21 variants of a fine-tuned classifier. Part 3: a test on signs from another country and a large custom network. Part 4: a classifier with a reject option that can answer "unknown".

## Summary

Four open models were compared on the same set of 430 traffic sign photos.

| Approach | Trained on signs | Accuracy | Time per photo | GPU memory |
|---|---|---|---|---|
| ResNet-18 classifier | yes, 0.5 min | **99.5%** | **1.7 ms** | **0.2 GB** |
| SigLIP 2 | no | 60.7% | 20.7 ms | 2.2 GB |
| Vision LLM Qwen3-VL-4B | no | 51.6–52.6% | 137–199 ms | 8.6 GB |
| CLIP | no | 39.1% | 11.3 ms | 0.8 GB |

A small classifier fine-tuned on labelled signs wins on all three axes at once: it is more accurate, 7–117 times faster and 4–40 times lighter on GPU memory than any zero-shot approach. The best zero-shot approach, SigLIP 2, gets four photos out of ten wrong.

A correction following Part 2: the classifier's 99.5% was a lucky run. Four repeats of the same training gave 97.2% to 99.1% on these 430 photos, and 98.8% to 99.3% on all 12,630 test photos. This does not change the conclusion. Details are in the "Part 2" section.

Caveat: the classifier was trained on photos from the same dataset as the test. On signs from another country its accuracy is 1–7 points lower; this is measured in Part 3.

## Methodology

### Task

The input is an already cropped photo of a single sign, the output is one of 43 classes. Finding the sign in a frame (detection) was not tested: none of the compared approaches does that.

### Test set

- **Source:** the official test split of GTSRB (German Traffic Sign Recognition Benchmark), 12,630 photos, 43 classes.
- **Selection:** 10 random photos per class, 430 in total. The generator seed is fixed (42), and the set is reproduced by the script `scripts/make_testset.py`.
- **Photo quality:** real footage shot from a car. Width from 26 to 260 pixels, median 43. Dark, blurred and overexposed frames were not filtered out.
- **Balance:** the classes are equally represented, so accuracy does not depend on which signs occur more often.

With 430 photos, the margin of error on accuracy is about ±4.6 percentage points for the zero-shot approaches (95% confidence interval) and roughly 98.3–99.9% for the classifier.

### What was compared

| Approach | Model | How the answer is produced |
|---|---|---|
| CLIP | `openai/clip-vit-large-patch14` | The photo is compared with the text "a photo of a '…' traffic sign" for each class, and the most similar one is taken |
| SigLIP 2 | `google/siglip2-so400m-patch14-384` | Same as above |
| Vision LLM | `Qwen/Qwen3-VL-4B-Instruct` | The model is shown the photo and a list of 43 options. Two modes: answers with the option number and answers with the name |
| Classifier | ResNet-18, pretrained on ImageNet | Fine-tuned on the GTSRB training split, outputs the class directly |

The first three approaches have not seen a single labelled sign from GTSRB: they are given only the English class names. The classifier serves as the "ceiling": what training on labelled data gives.

### Fine-tuning the classifier

- **Data:** the GTSRB training split in the torchvision version, 26,640 photos. It does not overlap with the test split.
- **Parameters:** 96×96 input, 6 epochs, batch size 128, AdamW, learning rate up to 0.001 with a OneCycle schedule.
- **Augmentations:** small rotations, shifts, changes in scale, brightness and contrast. No mirror flips, because they change the meaning of signs.
- **Time:** 0.5 minutes on an RTX 4090 Laptop.

### What was measured and how

- **Quality:** the share of correct answers out of 430. Additionally: per-class accuracy, frequent errors and accuracy by photo size.
- **Speed:** average time per photo, including reading the file, preparing the image and running the model. Photos are fed one at a time, without batching. One warm-up run is made before the measurement. Loading the model into memory is not included in the measurement.
- **GPU memory:** peak memory allocated by PyTorch during recognition (`torch.cuda.max_memory_allocated`). This is the weights plus intermediate data; `nvidia-smi` would show a few hundred megabytes more.
- **RAM:** peak process memory over the whole run, including loading the model and, for the classifier, training as well.
- **Model size:** the number of parameters and the size of the weights in the form in which they sit in GPU memory.

### Hardware and software

- **Machine:** a laptop with an NVIDIA RTX 4090 Laptop (16 GB of GPU memory), 22 CPU threads, 61 GB of RAM, Ubuntu 26.04.
- **Software:** Python 3.14, PyTorch 2.15 (CUDA 13.2), torchvision 0.29, transformers 5.24.
- **Numerical precision:** CLIP and SigLIP 2: float16, Qwen3-VL: bfloat16, classifier: float32.

## Results

### Quality

| Approach | Accuracy | Correct out of 430 |
|---|---|---|
| Classifier | 99.5% | 428 |
| SigLIP 2 | 60.7% | 261 |
| Vision LLM, answers with the name | 52.6% | 226 |
| Vision LLM, answers with a number | 51.6% | 222 |
| CLIP | 39.1% | 168 |

Accuracy by photo size:

| Approach | up to 35 px (131 photos) | 36–60 px (199 photos) | 61 px and up (100 photos) |
|---|---|---|---|
| Classifier | 98% | 100% | 100% |
| SigLIP 2 | 38% | 67% | 78% |
| Vision LLM, answers with the name | 34% | 60% | 61% |
| Vision LLM, answers with a number | 36% | 60% | 55% |
| CLIP | 27% | 46% | 41% |

The zero-shot approaches lose a lot on small photos: SigLIP 2 is twice as accurate on signs larger than 60 pixels as on signs up to 35 pixels. The classifier is almost indifferent to size.

### Speed

| Approach | Time per photo | Photos per second |
|---|---|---|
| Classifier | 1.7 ms | 586 |
| CLIP | 11.3 ms | 88 |
| SigLIP 2 | 20.7 ms | 48 |
| Vision LLM, answers with a number | 137.0 ms | 7 |
| Vision LLM, answers with the name | 198.6 ms | 5 |

For video at 30 frames per second, only the classifier and CLIP are fast enough with headroom; SigLIP 2 barely makes it, and the vision LLM does not.

### Resources

| Approach | Parameters | Weights | Peak GPU memory | Peak RAM |
|---|---|---|---|---|
| Classifier | 11M | 43 MB | 198 MB | 2.3 GB |
| CLIP | 428M | 816 MB | 834 MB | 2.9 GB |
| SigLIP 2 | 1,136M | 2.2 GB | 2.2 GB | 5.6 GB |
| Vision LLM | 4,438M | 8.5 GB | 8.6 GB | 7.0 GB |

During training the classifier used up to 495 MB of GPU memory. The classifier's peak RAM includes training; for recognition alone it would be lower.

The classifier is the only one of the four models whose size makes it suitable for a phone or an embedded device.

### Where they go wrong

**Classifier**: 2 errors out of 430:
- "End of speed limit 80" taken for "End of no overtaking";
- "Roundabout" taken for "Ahead only".

**SigLIP 2** confuses left and right and does not recognise crossed-out signs:
- "Go straight or right" taken for "Go straight or left": 10 times out of 10;
- "End of all limits" taken for "No entry": 7 times;
- "Dangerous curve right" taken for "Dangerous curve left": 6 times.

**Vision LLM** makes similar errors, and in addition does not distinguish signs that are close in meaning:
- "Ahead only" taken for "Go straight or right": 10 times out of 10 in both modes;
- "Dangerous curve left" and "right" confused 8 times;
- "Slippery road" taken for "Wild animals": 6–7 times.

The answer mode has almost no effect on accuracy: 51.6% with a number versus 52.6% with the name, a difference within the margin of error. There were no answers outside the list in either mode. The two modes gave the same answer on 331 photos out of 430.

**CLIP** lumps everything with a truck and a strike-through into one class:
- "No trucks" taken for "No overtaking for trucks": 10 times out of 10;
- 8 classes out of 43 were not recognised even once.

The "Right of way at the next intersection" sign was not recognised by any zero-shot approach (0 out of 10 for all three). The likely cause is the name: in the dataset it is "right of way at the next intersection", and from such text it is hard to tell what the sign looks like. This is an assumption; it was not tested separately.

Full error lists for each approach are in `results/report.md`.

## Conclusions

1. **If labelled data is available, an ordinary classifier is unbeatable.** 99.5% at 1.7 ms and 0.2 GB of GPU memory; training took half a minute.
2. **Zero-shot approaches are not accurate enough for this task.** The best result is 60.7%. They are suitable as a quick prototype or for a rough first pass over data, but not as a finished solution.
3. **The vision LLM turned out to be the worst use of resources:** 4 times more GPU memory and 7–10 times slower than SigLIP 2, with lower accuracy.
4. **Small signs are the main difficulty for zero-shot models.** On signs up to 35 pixels all three give 27–38%.

## Limitations

- **The classifier's data advantage.** It was trained on GTSRB and tested on GTSRB: the same country, the same camera. On Ukrainian signs and under other shooting conditions the result will be worse, and this was not measured.
- **Zero-shot approaches depend on the wording.** One wording per class was used, with literal English names of German signs. Tuning the descriptions ("a red circle with the number 50" instead of "speed limit 50 km/h") may raise accuracy; this was not tested.
- **One model of each type.** A larger vision LLM might have done better, but Qwen3-VL-8B does not fit into 16 GB of GPU memory without compression.
- **Cropped signs only.** Working with a full road frame requires a detector and was not evaluated here.
- **Speed was measured on a single GPU.** On a CPU or on a Mac the ratios will hold, but the absolute numbers will differ.
- **Jev was not tested,** following the decision not to connect OpenRouter at this stage.

## Part 2. Comparing fine-tuned classifiers

Part 1 showed that training on labelled signs solves the task. Part 2 answers the question of which model to use for that: 21 variants are compared, from a custom network with 0.3M parameters to ViT-B with 86M.

### Methodology

- **Training:** identical for all: the same 26,640 photos, 6 epochs, batch size 128, AdamW, OneCycle schedule, the same augmentations. Only the learning rate differs (chosen by model type, from 0.00003 to 0.001). The custom small network is trained from scratch for 15 epochs.
- **Input size:** convolutional networks receive 96×96 photos; transformers and hybrids receive 224×224, the size they were pretrained on. The custom small network: 48×48.
- **Quality:** on all 12,630 GTSRB test photos. On 430 photos the models are almost indistinguishable; accuracy on 430 is given to link back to Part 1.
- **Speed and GPU memory:** as in Part 1: the set of 430 photos, one photo at a time, float32.
- **Run-to-run variance:** ResNet-18 was trained four times with different generator seeds. The other models were trained once each.

The test of two "frozen backbone" variants (DINOv2, DINOv3) is supplemented with full fine-tuning of DINOv2, and ResNet-18 was tested with two sets of pretrained weights.

### Run-to-run variance

Four trainings of ResNet-18 with identical settings:

| Run | Accuracy on 12,630 | Errors | Accuracy on 430 |
|---|---|---|---|
| 1 | 98.76% | 156 | 97.2% |
| 2 | 99.34% | 83 | 99.1% |
| 3 | 98.95% | 133 | 98.6% |
| 4 | 99.02% | 124 | 98.4% |

Mean 99.02%, range 0.58 percentage points. Hence the rule for reading the table below: **a difference between models of less than about 0.5 points means nothing**; it is the noise of a single run.

### Results

For ResNet-18 with torchvision weights, the mean over four runs is given.

| Model | Input | Parameters | Accuracy | Time per photo | GPU memory | Training | GPU memory during training |
|---|---|---|---|---|---|---|---|
| CLIP B/16, fine-tuned | 224 | 85.8M | 99.50% | 5.6 ms | 1,319 MB | 9.1 min | 9.8 GB |
| FastViT-T8 | 224 | 3.3M | 99.31% | 6.2 ms | 61 MB | 4.4 min | 3.9 GB |
| ViT-B/16 | 224 | 85.8M | 99.30% | 6.3 ms | 1,319 MB | 9.1 min | 9.8 GB |
| SigLIP 2 B/16, fine-tuned | 224 | 92.9M | 99.29% | 6.1 ms | 1,429 MB | 9.7 min | 10.0 GB |
| DINOv2-S, fully fine-tuned | 224 | 21.6M | 99.25% | 5.8 ms | 337 MB | 5.0 min | 6.4 GB |
| ConvNeXt-Tiny | 96 | 27.9M | 99.12% | 3.3 ms | 443 MB | 1.1 min | 1.9 GB |
| DeiT-S/16 | 224 | 21.7M | 99.08% | 3.8 ms | 339 MB | 3.5 min | 4.6 GB |
| **ResNet-18 (torchvision weights)** | 96 | 11.2M | **99.02%** | **1.7 ms** | 185 MB | 0.5 min | 0.5 GB |
| **Custom small CNN, from scratch** | 48 | 0.3M | **98.99%** | **0.8 ms** | **6 MB** | 0.5 min | 0.2 GB |
| MobileViT-S | 224 | 5.0M | 98.98% | 6.0 ms | 91 MB | 5.5 min | 9.0 GB |
| Swin-T | 224 | 27.6M | 98.84% | 8.4 ms | 457 MB | 6.6 min | 9.1 GB |
| ConvNeXtV2-Tiny | 96 | 27.9M | 98.80% | 4.3 ms | 444 MB | 2.2 min | 3.0 GB |
| EfficientFormer-L1 | 224 | 11.4M | 98.69% | 5.1 ms | 190 MB | 3.8 min | 5.0 GB |
| ViT-S/16 | 224 | 21.7M | 98.67% | 5.1 ms | 339 MB | 3.4 min | 4.6 GB |
| ResNet-18 (timm a1 weights) | 96 | 11.2M | 98.25% | 1.6 ms | 185 MB | 0.5 min | 0.5 GB |
| EfficientNetV2-S | 96 | 20.2M | 98.06% | 8.2 ms | 322 MB | 1.7 min | 2.2 GB |
| MobileNetV3-Small | 96 | 1.6M | 96.72% | 2.7 ms | 25 MB | 0.4 min | 0.3 GB |
| MobileNetV3-Large | 96 | 4.3M | 96.63% | 3.2 ms | 67 MB | 0.6 min | 0.7 GB |
| EfficientNet-B0 | 96 | 4.1M | 96.49% | 3.8 ms | 65 MB | 0.9 min | 1.1 GB |
| DINOv3-S, linear head only | 224 | 21.6M | 65.69% | 10.0 ms | 93 MB | 1.9 min | 0.5 GB |
| DINOv2-S, linear head only | 224 | 21.6M | 65.60% | 4.1 ms | 88 MB | 1.9 min | 0.5 GB |

The full table with individual runs is in `results/classifiers.md`.

### What follows from this

1. **Accuracy tops out at 99.0–99.5%: eight variants fall into this band, and two more trail it by hundredths of a percent.** Given the 0.5-point variance, they cannot be told apart by quality on this test. The nominal leader, the fine-tuned CLIP B/16, is ahead of the average ResNet-18 by 0.48 points.
2. **Large models pay for these fractions of a percent many times over.** CLIP, ViT-B and SigLIP 2 are 3–4 times slower than ResNet-18, and use 7 times more GPU memory at inference and 20 times more during training.
3. **The most economical model is the custom small network.** 0.3M parameters, 6 MB of GPU memory, 0.8 ms per photo, training from scratch in half a minute, and 98.99%, on par with ResNet-18. For 43 classes of small images, ImageNet pretraining turned out to be unnecessary.
4. **MobileNetV3 and EfficientNet-B0 did worse than I expected:** 96.5–96.7%, 2.5 points below ResNet-18. This is more than the variance, so the gap is real under this training regime. A possible cause is the short training (6 epochs) and the small input; with longer training they may catch up, but this was not tested.
5. **A frozen DINO backbone with a single trainable layer does not work: 65.6%.** The same DINOv2 with full fine-tuning gives 99.25%. In other words, DINO features "out of the box" distinguish signs poorly, and my assumption that this variant is good when labels are scarce was not confirmed here. Caveat: the last layer was trained for the same 6 epochs; with longer training the result may be higher.
6. **Pretrained weights matter.** ResNet-18 with timm a1 weights gave 98.25%, lower than all four runs with torchvision weights (98.76–99.34%). This is a single run, so the conclusion is preliminary.
7. **On a GPU, speed is not determined by the number of parameters.** FastViT-T8 with 3.3M parameters takes 6.2 ms, ResNet-18 with 11.2M takes 1.7 ms. When photos are processed one at a time, the time goes into the number of layers and the input size. On a CPU or a phone the ordering of models may be different; this was not measured.

### Which model to choose

- **A server or a laptop with a GPU:** ResNet-18. Accuracy within the common band, the fastest of the pretrained models, trains in half a minute.
- **A weak device, a microcontroller:** the custom small network. The weights take about 1 MB.
- **Headroom for new shooting conditions:** fine-tuned CLIP, SigLIP 2 or DINOv2. Their advantage on this test is within the noise; the argument in their favour is stronger pretraining, but it can only be verified on data from a different source.

### Limitations of Part 2

- **One run per model,** except ResNet-18. The ordering of models inside the 99.0–99.5% band is random.
- **One training regime for all.** It was tuned for ResNet-18; lightweight mobile networks and transformers might benefit from other settings and more epochs.
- **The test comes from the same dataset as the training.** The main promised advantage of large pretrained models, robustness to new conditions, is not tested here. The test on signs from another country is in Part 3.
- **Speed on a GPU only.** Choosing a model for a phone requires a separate measurement on a CPU.

## Part 3. Signs from another country and a large custom network

### What was tested

Two questions. How much accuracy drops when a model is trained on German signs but runs on Belgian ones. And what is gained by enlarging a custom network trained from scratch.

### Methodology

- **Dataset.** BelgiumTSC, test split: cropped photos of signs shot in Belgium with a different camera. Out of 62 Belgian classes, the 19 that have an exact counterpart among the 43 GTSRB classes were taken: 940 photos. The mapping was built by hand from the photos of each class (`scripts/make_belgium.py`).
- **What was left out.** Classes without a counterpart in GTSRB; the class in which all speed limits sit under a single number; mirrored signs and signs that are similar but different.
- **The task is the same.** The model chooses from all 43 GTSRB classes and sees no Belgian photos during training.
- **Retraining.** The model weights from Part 2 were not saved, so each model was trained on GTSRB once more with the same recipe and immediately tested on both sets. The three fast models were trained four times each.
- **Large custom network.** The same design as the small one, but 4 blocks instead of 3, channels 64–512 instead of 32–128, 64 px input instead of 48, 30 epochs instead of 15. 4.7M parameters versus 0.3M. Trained from scratch, without pretrained weights.
- **Speed was not measured here:** other runs were going on at the same time. Model speed is in Part 2.

### The large custom network on the German test

| Model | Parameters | Accuracy, 12,630 photos | Time per photo | GPU memory | Training |
|---|---|---|---|---|---|
| Custom large CNN | 4.7M | 99.42–99.51% (4 runs) | 1.1 ms | 83 MB | 3.4 min |
| Custom small CNN | 0.3M | 98.92–99.10% (4 runs) | 0.8 ms | 6 MB | 0.5 min |
| ResNet-18 | 11.2M | 98.76–99.34% (4 runs) | 1.7 ms | 185 MB | 0.5 min |

The large network consistently gives about 99.5%, the same as the best result of Part 2 (fine-tuned CLIP B/16, 99.50%), while being 5 times faster and using 16 times less GPU memory.

### Run-to-run variance on Belgian signs

| Model | Belgium, 4 runs | Mean |
|---|---|---|
| Custom large CNN | 95.3% – 97.8% | 96.9% |
| Custom small CNN | 94.3% – 96.0% | 94.9% |
| ResNet-18 | 93.8% – 96.0% | 94.8% |

The variance reaches 2.5 points, four times more than on the German test. A difference between models of less than 2 points cannot be distinguished from this table.

### Results

One row per model; for the three models with repeats, the mean over four runs.

| Model | Parameters, M | Germany | Belgium | Drop, pp |
|---|---|---|---|---|
| CLIP B/16, fine-tuned | 85.8 | 99.50% | 98.2% | 1.3 |
| EfficientFormer-L1 | 11.4 | 98.69% | 97.0% | 1.7 |
| **Custom large CNN (mean)** | 4.7 | 99.48% | 96.9% | 2.6 |
| DeiT-S/16 | 21.7 | 99.08% | 96.8% | 2.3 |
| ViT-S/16 | 21.7 | 98.67% | 96.6% | 2.1 |
| SigLIP 2 B/16, fine-tuned | 92.9 | 99.29% | 96.6% | 2.7 |
| ConvNeXt-Tiny | 27.9 | 99.12% | 96.4% | 2.7 |
| Swin-T | 27.6 | 98.84% | 95.7% | 3.1 |
| MobileViT-S | 5.0 | 98.85% | 95.6% | 3.2 |
| Custom small CNN (mean) | 0.3 | 98.99% | 94.9% | 4.1 |
| ViT-B/16 | 85.8 | 99.30% | 94.9% | 4.4 |
| ResNet-18, torchvision weights (mean) | 11.2 | 99.02% | 94.8% | 4.3 |
| ConvNeXtV2-Tiny | 27.9 | 98.80% | 94.6% | 4.2 |
| DINOv2-S, fully fine-tuned | 21.6 | 99.37% | 93.3% | 6.1 |
| EfficientNetV2-S | 20.2 | 98.22% | 92.8% | 5.5 |
| FastViT-T8 | 3.3 | 99.31% | 91.7% | 7.6 |
| ResNet-18, timm a1 weights | 11.2 | 98.25% | 91.2% | 7.1 |
| MobileNetV3-Small | 1.6 | 96.72% | 91.0% | 5.8 |
| EfficientNet-B0 | 4.1 | 96.49% | 86.1% | 10.4 |
| MobileNetV3-Large | 4.3 | 96.63% | 85.3% | 11.3 |
| SigLIP 2 So400m, zero-shot | 1,136 | — | 75.4% | — |
| DINOv3-S, linear head only | 21.6 | 65.69% | 74.5% | −8.8 |
| DINOv2-S, linear head only | 21.6 | 65.60% | 67.9% | −2.3 |
| CLIP L/14, zero-shot | 428 | — | 40.6% | — |

The full table with individual runs, per-sign accuracy and the list of errors is in `results/transfer.md`.

### Where they go wrong on Belgian signs

- **"Children crossing" is the main source of errors.** On average the models recognise 72% of the 90 such photos. The pictogram on the Belgian sign is different, and it is taken for "Right of way at the next intersection", "Ice or snow" and "Road work".
- **"Bicycles crossing" is hardly recognised at all:** on average 7% of 12 photos. The pictogram also differs from the German one.
- **Signs with an identical pictogram transfer well:** "Yield", "No entry", "Priority road", "Stop": 98–100%.

In other words, the accuracy drop comes mostly from signs that look different in Belgium, not from the different camera.

### What follows from this

- **All models lose on foreign signs.** In the upper half of the table the drop is 1.3–4.4 points, in the lower half up to 11.
- **Large pretrained models do not give clear robustness.** Fine-tuned CLIP is the best (98.2%), but ViT-B/16 of the same size is on par with ResNet-18 (94.9%), and DINOv2 is lower (93.3%). Given the 2.5-point variance, CLIP and the custom large network are indistinguishable.
- **The custom large network is in the top group on both sets** with 4.7M parameters.
- **Lightweight mobile networks sag the most:** MobileNetV3-Large and EfficientNet-B0 lose 10–11 points.
- **Zero-shot approaches remain far behind:** SigLIP 2 at 75.4%, CLIP at 40.6%. They do not depend on the country, but they do not reach the trained models either.

### Limitations of Part 3

- **Few photos of rare signs.** Seven classes out of 19 have fewer than 13 photos; 940 photos give a rough estimate.
- **One country.** Belgian signs are close to German ones. On signs of a different design (USA, China) the losses will be larger.
- **One run each** for all models except three. With a variance of 2.5 points, the order of rows in the middle of the table is random.

## Part 4. Classifier with a reject option

### Task

An ordinary classifier must choose one of the 43 classes and therefore gives false answers in two cases: the sign is poorly shot and is taken for another one, or the sign is not among the 43 classes. What is needed is a single model that answers either "this is sign such-and-such" or "unknown".

### Methodology

The model outputs an answer and its confidence in it. If the confidence is below the threshold, the answer is "unknown". Three ways of training the same network were compared:

- **Confidence threshold.** An ordinary 43-class classifier; nothing changes, only a threshold is added.
- **"Unknown" class.** The network has a 44th output, trained on Belgian signs that are absent from GTSRB. Rejection is based on the same confidence in the best of the 43 classes.
- **"Unknown" class + Belgian signs of known classes.** The same, but Belgian signs of known classes are also added to the training (1,773 photos from the BelgiumTSC training split).

Data:

- **Unknown signs.** 36 Belgian classes without a counterpart in GTSRB (parking, dead end, no turn and others), `scripts/make_belgium_unknown.py`. They are split alternately into two groups: the model sees group A types when training the "unknown" class, and never sees group B types.
- **Evaluation.** The German test (12,630 photos), Belgian signs of known classes (940), unknown signs of seen types (529, different photos) and unknown signs of unseen types (457).
- **Choosing the threshold.** Not on the evaluation data, but on held-out German signs: 2,070 photos from tracks that were not used in training. The threshold is set so that the model accepts a given share of them.

Three networks: ResNet-18, the custom small CNN and the custom large CNN. One run per variant. A fourth way of training, with background, is described below in the "Photos without signs" section.

### Results

Threshold: "keep 97%" (accepts 97% of held-out German signs). "Correct" and "false" are shares of all photos in the set; the rest are rejected.

| Model | Method | Germany: correct | Germany: false | Belgium: correct | Belgium: false | Unknown signs, seen types: false | Unknown signs, unseen types: false |
|---|---|---|---|---|---|---|---|
| ResNet-18 | no reject option | 98.9% | 1.07% | 96.6% | 3.40% | 100% | 100% |
| ResNet-18 | confidence threshold | 96.3% | 0.13% | 86.7% | 0.32% | 19.7% | 8.5% |
| ResNet-18 | "unknown" class | 94.7% | 0.25% | 53.6% | 0.00% | 0.2% | 2.8% |
| ResNet-18 | "unknown" class + Belgian signs of known classes | 97.1% | 0.25% | 99.3%* | 0.00% | 0.6% | 10.3% |
| Custom small CNN | confidence threshold | 95.2% | 0.06% | 84.5% | 0.11% | 10.6% | 9.8% |
| Custom small CNN | "unknown" class + Belgian signs of known classes | 93.7% | 0.04% | 98.3%* | 0.00% | 0.6% | 1.8% |
| Custom large CNN | confidence threshold | 92.4% | 0.02% | 68.6% | 0.00% | 9.6% | 0.9% |
| **Custom large CNN** | **"unknown" class + Belgian signs of known classes** | **94.1%** | **0.04%** | **98.1%*** | **0.11%** | **0.6%** | **0.9%** |

\* These models saw Belgian signs of known classes during training (different photos), so here this is not a test on a new country.

All thresholds (99%, 97%, 95%, 90%) and all twelve variants are in `results/reject.md`.

### What follows from this

- **A single reject model works.** The best variant is the custom large CNN with the "unknown" class and mixed training: 0.04% false accepts on German signs, 0.11% on Belgian ones and 0.9% on unknown signs of unseen types. The price is rejecting 5.9% of German photos.
- **Rejection is a trade-off.** Without a reject option, ResNet-18 is wrong on 1.07% of German photos. To bring this down to 0.13%, 3.6% of photos have to be rejected, most of them correct.
- **A plain threshold is poor at catching unknown signs.** ResNet-18 with a threshold takes 8.5–19.7% of unknown signs for known ones: on unfamiliar input the network is often confident for no reason.
- **The "unknown" class learns what it was shown.** Trained only on unknown Belgian signs, it also started rejecting known signs shot in Belgium: 53.6% correct answers instead of 86.7%. The network learned the feature "Belgian photo" rather than "unknown sign". Adding Belgian signs of known classes to the training fixed this.
- **Protection is weaker on unseen types than on seen ones.** For ResNet-18 in the mixed variant: 10.3% false on unseen types versus 0.6% on seen ones. For the custom networks the gap is smaller.

### Photos without signs

A separate test: what the model answers on photos that contain no sign at all.

- **Road background crops.** 2,000 crops from full frames of German roads (GTSDB) that do not touch any labelled sign, 30–120 px in size, like cropped signs. Uniform crops (sky, asphalt) were discarded. `scripts/make_background.py`.
- **Random objects.** 2,000 photos from CIFAR-100: animals, vehicles, furniture. No model saw them during training.
- **Fourth way of training:** "unknown" class + Belgian signs of known classes + background. 4,000 background crops, taken from frames other than the evaluation ones, were added to "unknown".

The threshold is the same: "keep 97%" (accepts 97% of held-out German signs). The table shows the share of photos on which the model named some sign.

| Model | Method | Road background: false | Objects: false | Unknown signs, unseen types: false | Germany: correct | Germany: false |
|---|---|---|---|---|---|---|
| ResNet-18 | confidence threshold | 0.4% | 1.8% | 8.5% | 96.3% | 0.13% |
| ResNet-18 | "unknown" class + Belgian signs of known classes | 3.6% | 7.2% | 10.3% | 97.1% | 0.25% |
| ResNet-18 | same + background | 0.0% | 0.3% | 5.3% | 96.3% | 0.23% |
| Custom small CNN | confidence threshold | 4.4% | 4.9% | 9.8% | 95.2% | 0.06% |
| Custom small CNN | "unknown" class + Belgian signs of known classes | 2.9% | 2.9% | 1.8% | 93.7% | 0.04% |
| Custom small CNN | same + background | 0.1% | 1.1% | 5.7% | 95.1% | 0.10% |
| Custom large CNN | confidence threshold | 0.2% | 0.4% | 0.9% | 92.4% | 0.02% |
| Custom large CNN | "unknown" class + Belgian signs of known classes | 0.7% | 1.4% | 0.9% | 94.1% | 0.04% |
| **Custom large CNN** | **same + background** | **0.0%** | **0.2%** | **2.2%** | **94.6%** | **0.10%** |

- **Background in the training removes false accepts on background almost completely:** 0.0–0.1% for all three networks. For the large CNN, not one of the 2,000 road crops was taken for a sign.
- **This also carries over to things the model has not seen:** on random objects, 0.2–1.1% instead of 1.4–7.2%.
- **Without background in the training, the "unknown" class does not protect against background.** For ResNet-18 it is even worse than a plain threshold: 3.6% versus 0.4%. The model learned to reject unknown signs, not "everything that is not a sign".
- **Large networks with a plain threshold already reject background reasonably well** (0.2–0.4%); the small one does not (4.4%).
- **On unknown signs of unseen types the numbers fluctuate** from 0.9% to 5.7% with no clear pattern. This is 4–26 photos out of 457 with a single run; the differences between methods are unreliable here.

The final variant is the custom large CNN trained on German and Belgian signs with an "unknown" class made of unknown signs and background. At the 97% threshold: 94.6% correct answers on the German test, 0.10% false, 5.3% rejected; 0.0% false on background, 0.2% on random objects, 0.6–2.2% on unknown signs.

### Ready-to-use model

The weights are saved in `models/big_cnn_oe_all.pt` (18 MB). This is the variant trained on all unknown sign types; see the section "Training on all unknown types and two-model agreement". The previous variant is next to it, `models/big_cnn_oe_bg.pt`. Recognition of one or several photos is done with `predict.py`:

```bash
python predict.py data/testset/images/14_00251.png
```

The answer is the English sign name or "unknown", plus the confidence. The `--keep` option selects the threshold: 0.99 rejects less often and errs more often, 0.9 the other way round; the default is 0.97. The input must be a photo of a single sign cropped from the frame; the script cannot find a sign in a full frame. torch and torchvision are required; a GPU is not.

A check on the 4090 (with the previous variant): 11 German and 15 Belgian photos of the "Stop" sign were recognised correctly, and 2 unknown Belgian signs, 10 background crops and 5 random objects got the answer "unknown".

### Can zero false accepts be achieved

Tested on the saved final model, without retraining.

**With a threshold: no.** The model gave 13 false answers with a confidence of 1.0, that is, indistinguishable from the most confident correct ones. What these photos are:

| How many | What is in the photo | What the model answered | What it means |
|---|---|---|---|
| 3 | "Right of way at the next intersection", which in BelgiumTSC sits in the folder of another sign | "Right of way at the next intersection" | A labelling error in the dataset; the model is right |
| 8 | "3.5 t weight limit" | "Speed limit 30" | The model did not see this sign type during training, and it looks almost like "30" |
| 2 | A blurred "Speed limit 60" | "Speed limit 80" | A genuine error on a bad photo |

Allowing for the labelling error, there are no genuine false answers on unknown signs of seen types: 0 out of 526.

**Multi-view (test-time augmentation) agreement on a single photo did not help.** The model looked at each photo 9 times (as is, larger, smaller, rotated by ±8°, shifted, darker, lighter) and answered only if all 9 answers matched (`bench/tta.py`). The number of false answers at the same share of accepted held-out signs:

| Keep | Rule | Germany: correct | Germany: false | Unknown signs, unseen types: false | Objects: false |
|---|---|---|---|---|---|
| 97% | single view | 94.6% | 12 | 10 | 4 |
| 97% | 9 views agree | 93.2% | 12 | 10 | 3 |
| 95% | single view | 92.3% | 6 | 10 | 2 |
| 95% | 9 views agree | 91.2% | 8 | 8 | 2 |
| 90% | single view | 85.7% | 2 | 5 | 0 |
| 90% | 9 views agree | 84.8% | 2 | 7 | 1 |

The number of false answers is the same, correct answers drop by 1–1.5 points, and it runs 9 times slower. Confident errors are stable: the model makes the same mistake on all views of the photo.

**What follows from this.**

- **False answers on unknown signs are cured with data, not with a threshold.** On types that the model saw in the "unknown" class there are no genuine false answers. All confident errors on unknown signs are a single unseen type that resembles a known one. It has to be shown during training, as do the other look-alike types.
- **False answers on poorly shot known signs remain.** The strictest threshold leaves 2 out of 12,630 while rejecting 14% of photos. They cannot be removed from a single photo: a blurred "60" really does look like "80".
- **Zero cannot be proven by a test.** 0 false accepts on 2,000 photos only means that the true rate is most likely below 0.15%.
- **Agreement of two different models is tested in the next section.** Agreement across adjacent video frames was not tested; in practice it is the strongest, because on the next frame a blurred sign is usually seen better.

### Training on all unknown types and two-model agreement

Two next steps after the error analysis.

- **All unknown types in the training.** The 18 types of group B were also added to the "unknown" class, that is, all 36 unknown Belgian types (mode `oe_all`). After this no unseen types remain for evaluation: the unknown signs in the evaluation are different photos of the same types.
- **Two-model agreement.** The custom large CNN and ResNet-18 are trained identically; an answer is accepted only if both named the same sign (`bench/agree.py`). The threshold for the pair is chosen the same way, by the share of accepted held-out German signs.
- **Labelling fixed.** A track of three photos with a known sign that sat in the folder of an unknown one was removed from the evaluation: there are now 526 unknown signs of seen types.

Number of false answers (set size in brackets):

| Training | Rule | Keep | Germany: correct | Germany: false (12,630) | Belgium: false (940) | Unknown signs: false (983) | Background: false (2,000) | Objects: false (2,000) |
|---|---|---|---|---|---|---|---|---|
| without group B types | large CNN | 97% | 94.6% | 12 | 0 | 10 | 0 | 4 |
| without group B types | both agree | 97% | 95.5% | 19 | 0 | 2 | 0 | 4 |
| without group B types | large CNN | 95% | 92.3% | 6 | 0 | 10 | 0 | 2 |
| without group B types | both agree | 95% | 92.0% | 8 | 0 | 1 | 0 | 0 |
| all unknown types | large CNN | 97% | 94.0% | 18 | 0 | 0 | 0 | 6 |
| all unknown types | both agree | 97% | 92.9% | 18 | 0 | 0 | 0 | 0 |
| **all unknown types** | **large CNN** | **95%** | **90.3%** | **4** | **0** | **0** | **0** | **0** |
| all unknown types | both agree | 95% | 90.6% | 13 | 0 | 0 | 0 | 0 |
| all unknown types | large CNN | 90% | 84.6% | 1 | 0 | 0 | 0 | 0 |

- **Unknown signs: zero false out of 983** when their types are shown during training, at any threshold. The hypothesis from the error analysis was confirmed.
- **Two-model agreement helps precisely on the unseen.** On unknown signs of unseen types: 2 false instead of 10; on random objects: 0 instead of 6. Two networks err differently on unfamiliar input.
- **On poorly shot known signs agreement does not help:** the number of false answers on the German test is the same or higher (18 versus 18, 13 versus 4). On a blurred photo both models make the same mistake.
- **One model, 95% threshold:** zero false on all sets except the German test, where 4 out of 12,630 (0.03%) remain, with 90.3% correct answers and 9.7% rejected. This is the best that was achieved with a single model.
- **Caveats.** One run per variant: 12 and 18 false for the same network in two trainings is variance, not an effect. The zero on unknown signs was obtained on types the model has seen; on a type that is in none of the sets, it may err just as confidently as it did earlier on "3.5 t".

### Limitations of Part 4

- **One run each.** Run-to-run variance was not measured; differences of 1–2 points may be random.
- **Groups A and B are similar.** Some of the "unseen" types are variants of seen ones (different parking and no-parking signs). For entirely dissimilar signs the numbers may be worse.
- **Background is cropped at random.** In a real system the classifier will receive not random crops but whatever the detector took for a sign: round and triangular objects, advertising, street lamps. There will be more false accepts on those.
- **The background is German.** The crops for training and evaluation are taken from different frames, but from the same dataset and the same camera.
- **The threshold is chosen on German signs.** On a different camera the rejection rate will be different: for the ordinary ResNet-18 with the same threshold, 13% of Belgian signs are rejected instead of 3.6%.

## Reproducing

Setup, data preparation and the commands for every part are in [README.md](README.md).
