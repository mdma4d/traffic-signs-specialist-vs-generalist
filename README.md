# Specialised versus General-Purpose Vision Models for Narrow-Domain Classification: An Empirical Study on Traffic Sign Recognition

## Abstract

Large general-purpose vision and vision-language models are increasingly proposed as drop-in replacements for task-specific classifiers. We test this claim on a narrow, well-defined task: closed-set recognition of 43 German traffic sign classes (GTSRB), followed by open-set recognition with a reject option. We compare three ways of using a large pretrained model (zero-shot inference, a linear probe on frozen features, and full fine-tuning) against compact convolutional networks trained from scratch, under one training recipe and one measurement protocol, on accuracy, latency and GPU memory. Zero-shot models with 0.4–4.4 billion parameters reach 39.1–60.7% accuracy; linear probes on frozen self-supervised features reach 65.6–65.7%; a 4.7M-parameter CNN trained from scratch in 3.4 minutes reaches 99.47 ± 0.04% (mean ± standard deviation over four seeds). The best fine-tuned large model (CLIP ViT-B/16, 85.8M parameters) is statistically indistinguishable from it (63 versus 62 errors on 12,630 images) at 18 times the parameters, 5 times the latency and 16 times the memory. For open-set recognition, the composition of the "unknown" training data, not the confidence threshold, determines the false-acceptance rate: the final model gives 4 false answers on 12,630 in-distribution images and none on 983 foreign signs, 2,000 background crops and 2,000 unrelated objects, at 90.3% coverage. Within the tested range, large general-purpose models do not replace a small specialised one. Code, per-image predictions and the trained model are included.

![Accuracy, latency and GPU memory of seven approaches](docs/accuracy.svg)

## Contents

1. [Introduction](#1-introduction)
2. [Methods](#2-methods)
3. [Results](#3-results)
4. [Discussion](#4-discussion)
5. [Threats to validity](#5-threats-to-validity)
6. [Using the trained model](#6-using-the-trained-model)
7. [Reproducing the results](#7-reproducing-the-results)
8. [References](#8-references)

A longer narrative report with every intermediate table is in [REPORT.md](REPORT.md).

## 1. Introduction

Contrastive vision-language models [5, 6], self-supervised vision backbones [7, 8] and vision LLMs [9] are trained on hundreds of millions to billions of images and transfer to many tasks without task-specific training. This raises a practical question for any narrow recognition problem: is a specialised model still worth training, or can a large general-purpose model take its place?

We study the question on traffic sign classification, a task with a fixed label set, abundant labelled data and strict latency requirements. A sign classifier in a vehicle must also decline to answer on inputs outside its label set, so we treat rejection as part of the task and not as an afterthought.

**Research questions.**

- **RQ1.** How accurate are large general-purpose models on this task without any task-specific training?
- **RQ2.** When the same labelled data are available to all models, does model size or large-scale pretraining improve accuracy over a compact network trained from scratch, and at what cost in latency and memory?
- **RQ3.** How do the trained models behave under distribution shift (signs of another country, another camera)?
- **RQ4.** What determines the false-acceptance rate of a classifier with a reject option: the confidence threshold or the training data?

**Contributions.**

- A controlled comparison of 22 trained classifiers (0.3M to 92.9M parameters) and three zero-shot models (428M to 4.4B parameters) under one protocol, reporting accuracy, single-image latency, peak GPU memory and training cost.
- Seed-to-seed variance estimates, which show that most published-style rankings inside the 99.0–99.5% band are not resolvable from single runs.
- An ablation of the "unknown" training data for open-set recognition, including two negative results (test-time augmentation and two-model agreement).
- A trained 18 MB model and scripts that rebuild every dataset and every number from public sources.

## 2. Methods

### 2.1 Task

**Closed-set task.** The input $`x`$ is an RGB crop containing one traffic sign with a small margin. The output is a class $`y \in \{1, \dots, 43\}`$ of the German Traffic Sign Recognition Benchmark [1]. Detection of signs in a full frame is out of scope.

**Open-set task.** The output space is extended to $`\{1, \dots, 43\} \cup \{\text{unknown}\}`$. The model must answer "unknown" for signs outside the 43 classes, for images without a sign, and may do so for in-distribution images it cannot classify reliably.

### 2.2 Data

| Set | Source | Role | Images |
|---|---|---|---|
| GTSRB train | [1] | training | 26,640 |
| GTSRB test | [1] | in-distribution test | 12,630 |
| GTSRB test, balanced subset | [1] | test for zero-shot models, 10 per class, fixed seed | 430 |
| GTSRB held-out tracks | [1] | threshold selection only (every tenth physical sign) | 2,070 |
| BelgiumTSC, known classes | [3] | shifted-distribution test: the 19 Belgian classes that map to GTSRB classes | 940 |
| BelgiumTSC, known classes, train | [3] | training of the open-set models | 1,773 |
| BelgiumTSC, foreign types | [3] | "unknown" signs: 36 types absent from GTSRB, split into groups A and B | 2,390 train / 983 test |
| GTSDB crops | [2] | road-scene crops without signs; training and test crops come from different frames | 4,000 train / 2,000 test |
| CIFAR-100 | [4] | unrelated objects, evaluation only | 2,000 |

GTSRB training images come in tracks of 30 frames of one physical sign. Held-out data are therefore split by track: a random image split would leave near-duplicates of training frames in the held-out part and bias the threshold.

### 2.3 Models

**Zero-shot (no training on signs).** CLIP ViT-L/14 [5] and SigLIP 2 So400m/14 at 384 px [6] embed the image and one text prompt per class, `a photo of a '<sign name>' traffic sign`, and predict the class with the highest cosine similarity:

```math
\hat{y}(x) = \arg\max_{k \in \{1..43\}} \frac{E_{img}(x) \cdot E_{txt}(t_k)}{\lVert E_{img}(x) \rVert \, \lVert E_{txt}(t_k) \rVert}
```

The vision LLM, Qwen3-VL-4B-Instruct [9], receives the image (upscaled to at least 224 px on the short side) and the list of 43 class names, and is asked to output one name; the answer is matched to the list.

**Linear probe.** DINOv2 ViT-S/14 [7] and DINOv3 ViT-S/16 [8] with the backbone frozen and only a linear classification layer trained.

**Full fine-tuning.** Eighteen ImageNet- or web-pretrained configurations from the `timm` library, from MobileNetV3-Small (1.6M parameters) to ViT-B/16 initialised from CLIP and SigLIP 2 weights (85.8M and 92.9M); see `MODELS` in [bench/finetune.py](bench/finetune.py) for the list.

**Compact CNNs trained from scratch.** A plain VGG-style network [15] without residual connections, attention or pretrained weights. One block is

```text
Conv 3x3 -> BatchNorm -> ReLU -> Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2
```

and the network is a stack of blocks followed by global average pooling, dropout (0.3) and one linear layer.

| | Small CNN | Large CNN |
|---|---|---|
| Input | 48×48 RGB | 64×64 RGB |
| Blocks and channels | 3: 32, 64, 128 | 4: 64, 128, 256, 512 |
| Feature map before global pooling | 6×6×128 | 4×4×512 |
| Parameters | 293,451 | 4,711,275 |
| Weights, float32 | 1.1 MB | 18 MB |
| Epochs | 15 | 30 |

In the large network 4,685,376 parameters are convolution weights, 3,840 belong to batch normalisation and 22,059 to the linear layer; the "unknown" output adds 513. The receptive field of the last convolution is 68 px, so every output feature covers the whole 64 px input.

### 2.4 Training

All trained models share one recipe; only the peak learning rate, the input size and the number of epochs differ.

- **Objective:** cross-entropy over $`K = 43`$ classes ($`K = 44`$ for the open-set models), without label smoothing or class weights:

  ```math
  L = -\frac{1}{B} \sum_{i=1}^{B} \log p_{y_i}(x_i), \qquad p_k(x) = \frac{e^{z_k(x)}}{\sum_{j=1}^{K} e^{z_j(x)}}, \qquad B = 128
  ```

- **Optimiser:** AdamW [13] with weight decay $`10^{-4}`$ and a one-cycle learning-rate schedule [14]. The peak rate is $`2 \cdot 10^{-3}`$ for the CNNs trained from scratch and between $`3 \cdot 10^{-5}`$ and $`10^{-3}`$ for pretrained models, set per model family.
- **Epochs:** 6 for pretrained models; 15 and 30 for the small and large CNN.
- **Precision:** bfloat16 autocast in training, float32 at inference.
- **Augmentation:** rotation up to ±10°, translation up to 10%, scale 0.9–1.1, brightness and contrast ±40%, saturation ±30%. Horizontal flips are excluded because they change the meaning of directional signs.
- **Input size:** 96 px for pretrained convolutional networks, 224 px for transformers and hybrids (their pretraining resolution), 48 and 64 px for the custom CNNs. Crops are resized to a square without preserving the aspect ratio.
- **Seeds:** seed 0 for every model; seeds 1–3 additionally for ResNet-18 and both custom CNNs.

### 2.5 Open-set recognition

The open-set models add a 44th "unknown" output trained on auxiliary outliers, following the outlier-exposure approach [10]. The confidence score is the maximum softmax probability [11] among the 43 sign classes, with the softmax taken over all 44 outputs, so that probability assigned to "unknown" lowers it:

```math
s(x) = \max_{k \le 43} p_k(x), \qquad
\text{answer}(x) = \begin{cases} \arg\max_{k \le 43} p_k(x) & \text{if } s(x) \ge \tau \\ \text{unknown} & \text{otherwise} \end{cases}
```

The threshold $`\tau_q`$ is the empirical $`(1-q)`$ quantile of $`s`$ on the 2,070 held-out German images, so that a fraction $`q \in \{0.99, 0.97, 0.95, 0.90\}`$ of them is accepted. No test data are used to set it.

Five training variants of the same network isolate the effect of the outlier data:

| Variant | Additional training data |
|---|---|
| `base` | none; 43 outputs, threshold on confidence only |
| `oe` | + foreign sign types of group A, labelled "unknown" |
| `oe_mix` | + Belgian images of the known classes, labelled with their class |
| `oe_bg` | + road-scene crops without signs, labelled "unknown" |
| `oe_all` | + foreign sign types of group B, labelled "unknown" |

The final model (`oe_all`, large CNN) is trained on 43,734 images, 28% of them labelled "unknown":

| Source | Label | Images | Repeats per epoch | In training |
|---|---|---|---|---|
| GTSRB train without held-out tracks | 43 signs | 24,570 | 1 | 24,570 |
| BelgiumTSC, known classes | 43 signs | 1,773 | 4 | 7,092 |
| BelgiumTSC, foreign types, group A | unknown | 868 | 4 | 3,472 |
| BelgiumTSC, foreign types, group B | unknown | 1,150 | 4 | 4,600 |
| GTSDB crops without signs | unknown | 4,000 | 1 | 4,000 |

Two further rules were evaluated on top of a trained model: accepting an answer only if nine augmented views of the image agree (test-time augmentation), and only if two independently trained networks agree.

### 2.6 Evaluation protocol

- **Accuracy.** Top-1 accuracy on the full GTSRB test split for trained models and on the balanced 430-image subset for zero-shot models. For a set of $`n`$ images the standard error and 95% interval are

  ```math
  SE = \sqrt{\frac{p(1-p)}{n}}, \qquad p \pm 1.96 \, SE
  ```

  which gives ±0.12 points at $`p = 0.995`$, $`n = 12630`$; ±4.6 points at $`p = 0.6`$, $`n = 430`$; and ±1.1 points at $`p = 0.97`$, $`n = 940`$.
- **Run-to-run variance.** Sample standard deviation of accuracy over four seeds, for the three models trained with several seeds.
- **Open-set metrics.** Each image is answered correctly, answered falsely, or rejected; the three shares sum to one. We report coverage (the share answered) and selective risk (false answers divided by answers) [12]. For a set containing no known signs, every answer is false.
- **Zero-count bound.** When 0 errors are observed in $`n`$ trials, the 95% upper bound on the true rate is $`3/n`$ (rule of three).
- **Latency.** Wall-clock time per image at batch size 1 on the GPU, after warm-up, including file decoding, resizing and normalisation. This is request latency, not batched throughput.
- **Memory.** Peak GPU memory allocated by PyTorch during inference (weights and activations). Zero-shot encoders run in float16, the vision LLM in bfloat16, trained classifiers in float32.
- **Hardware and software.** One RTX 4090 Laptop GPU (16 GB); Python 3.14, PyTorch 2.15, torchvision 0.29, transformers 5.24.

## 3. Results

### 3.1 RQ1: zero-shot models

| Model | Parameters | Trained on signs | Accuracy, 430 images | Latency | GPU memory |
|---|---|---|---|---|---|
| Large CNN (ours) | 4.7M | yes | 98.8% | 1.1 ms | 83 MB |
| ResNet-18 | 11.2M | yes | 97.2–99.5% | 1.7 ms | 0.2 GB |
| SigLIP 2 So400m | 1,136M | no | 60.7% | 20.7 ms | 2.2 GB |
| Qwen3-VL-4B | 4,438M | no | 52.6% | 199 ms | 8.6 GB |
| CLIP ViT-L/14 | 428M | no | 39.1% | 11.3 ms | 0.8 GB |

The best zero-shot model is 38 points below the large CNN on the same images, with 240 times more parameters. The vision LLM has 940 times more parameters, is 180 times slower and misclassifies every second image. The gap is an order of magnitude larger than the ±4.6-point sampling interval; the order of the zero-shot models among themselves is less certain.

Errors are systematic: the zero-shot models confuse mirrored signs (left and right), fail on crossed-out ("end of restriction") signs, and degrade with resolution. On signs up to 35 px wide all three score 27–38%.

### 3.2 RQ2: trained classifiers

| Model | Parameters | Input | Accuracy | Errors of 12,630 | Latency | GPU memory | Training |
|---|---|---|---|---|---|---|---|
| Large CNN, from scratch | 4.7M | 64 px | 99.51% | 62 | 1.1 ms | 83 MB | 3.4 min |
| CLIP ViT-B/16, fine-tuned | 85.8M | 224 px | 99.50% | 63 | 5.6 ms | 1,319 MB | 9.1 min |
| FastViT-T8 | 3.3M | 224 px | 99.31% | 87 | 6.2 ms | 61 MB | 4.4 min |
| ViT-B/16 | 85.8M | 224 px | 99.30% | 88 | 6.3 ms | 1,319 MB | 9.1 min |
| SigLIP 2 ViT-B/16, fine-tuned | 92.9M | 224 px | 99.29% | 90 | 6.1 ms | 1,429 MB | 9.7 min |
| DINOv2 ViT-S/14, fine-tuned | 21.7M | 224 px | 99.25% | 95 | 5.8 ms | 337 MB | 5.0 min |
| ResNet-18, mean of 4 seeds | 11.2M | 96 px | 99.02% | 124 | 1.7 ms | 185 MB | 0.5 min |
| Small CNN, from scratch | 0.3M | 48 px | 98.99% | 127 | 0.8 ms | 6 MB | 0.5 min |
| DINOv3 ViT-S/16, linear probe | 21.6M | 224 px | 65.69% | 4,333 | 10.0 ms | 93 MB | 1.9 min |
| DINOv2 ViT-S/14, linear probe | 21.7M | 224 px | 65.60% | 4,345 | 4.1 ms | 88 MB | 1.9 min |

All 22 models are listed in [results/classifiers.md](results/classifiers.md).

**Variance.**

| Model | Seeds | Accuracy, mean ± s.d. | Range |
|---|---|---|---|
| Large CNN | 4 | 99.47 ± 0.04% | 99.42–99.51% |
| Small CNN | 4 | 98.99 ± 0.08% | 98.92–99.10% |
| ResNet-18 | 4 | 99.02 ± 0.24% | 98.76–99.34% |

**Findings.**

- **Accuracy saturates at 99.0–99.5%,** and eight models fall in this band. With a sampling interval of ±0.12 points and a seed standard deviation of up to 0.24 points, single-run differences below about half a point are not significant.
- **Size and pretraining do not improve accuracy.** The large CNN (4.7M, random initialisation) and CLIP ViT-B/16 (85.8M, pretrained on 400M image-text pairs) differ by one error in 12,630.
- **The cost differs by an order of magnitude.** Relative to the large CNN, fine-tuned CLIP ViT-B/16 has 18.2 times the parameters, 5.1 times the latency, 15.9 times the inference memory and 13.2 times the training memory (9.8 GB against 0.7 GB).
- **Frozen general-purpose features are insufficient.** Linear probes reach 66%, while the same DINOv2 backbone fine-tuned end to end reaches 99.25%. The accuracy comes from training on the task, not from the pretrained representation.

### 3.3 RQ3: distribution shift

Models trained on German signs only were evaluated on 940 Belgian images of the 19 shared classes, without further training.

| Model | Runs | GTSRB test | BelgiumTSC | Drop |
|---|---|---|---|---|
| CLIP ViT-B/16, fine-tuned | 1 | 99.50% | 98.2% | 1.3 |
| Large CNN | 4 | 99.47% | 96.9 ± 1.1% | 2.6 |
| Small CNN | 4 | 98.99% | 94.9 ± 0.8% | 4.1 |
| ResNet-18 | 4 | 99.02% | 94.8 ± 1.1% | 4.3 |
| SigLIP 2 So400m, zero-shot | 1 | n/a | 75.4% | n/a |

Every model loses accuracy: the best lose 1–4 points, lightweight mobile networks up to 11. The loss is concentrated in classes whose pictograms differ between the countries: "Children crossing" is recognised in 72% of cases and "Bicycles crossing" in 7%, while classes with identical artwork transfer at 98–100%. The large CNN ranges from 95.3% to 97.8% across seeds, and CLIP was trained once, so the 1.3-point difference between them is comparable to one standard deviation of the seed spread and is not significant. This is the only experiment in which large-scale pretraining may give an advantage; the data do not establish it.

### 3.4 RQ4: open-set recognition

**Ablation of the "unknown" training data** (large CNN, $`q = 0.95`$; counts of false answers unless noted):

| Variant | GTSRB correct | GTSRB false | Belgian known, correct | Foreign A (≈526) | Foreign B (457) | Background (2,000) | Objects (2,000) |
|---|---|---|---|---|---|---|---|
| `base` | 89.8% | 2 | 62.1% | 35 | 2 | 2 | 5 |
| `oe` | 89.6% | 1 | 47.3% | 2 | 1 | 1 | 0 |
| `oe_mix` | 91.0% | 2 | 95.4% | 3 | 3 | 4 | 13 |
| `oe_bg` | 92.3% | 6 | 97.8% | 0 | 10 | 0 | 2 |
| `oe_all` | 90.3% | 4 | 96.1% | 0 | 0 * | 0 | 0 |

\* In `oe_all` the types of group B are part of training; the test images are different photos of types the model has seen.

- **A threshold alone is not sufficient.** The `base` model still accepts 35 foreign signs at a threshold of 0.99996, and answers only 62% of the Belgian images of known classes.
- **Outliers from one domain create a shortcut.** With Belgian foreign signs as the only outliers (`oe`), the network learns "Belgian photo means unknown": coverage of known Belgian signs falls to 47%. Adding Belgian images of known classes (`oe_mix`) restores it to 95%.
- **Each outlier source removes the errors of its own kind.** Background crops remove false answers on background; the group B types remove the 10 confident errors on group B. Before that, a "3.5 t weight limit" sign was read as "Speed limit 30".
- **In-distribution errors do not improve.** False answers on GTSRB stay between 1 and 6 across variants; outlier exposure addresses unfamiliar input, not ambiguous known signs.

**Final model** (`oe_all`, $`q = 0.95`$, $`\tau = 0.99999`$):

| Test set | n | Correct | False answers | Rejected | 95% upper bound on false rate |
|---|---|---|---|---|---|
| GTSRB test | 12,630 | 90.3% | 4 (0.03%) | 9.7% | n/a |
| Belgian signs, known classes | 940 | 96.1% | 0 | 3.9% | 0.32% |
| Foreign signs | 983 | n/a | 0 | 100% | 0.31% |
| Road background crops | 2,000 | n/a | 0 | 100% | 0.15% |
| Unrelated objects | 2,000 | n/a | 0 | 100% | 0.15% |

**Risk–coverage trade-off** on the GTSRB test:

| $`q`$ | Threshold $`\tau`$ | Coverage | False answers | Selective risk |
|---|---|---|---|---|
| 1 (no threshold) | 0 | 100% | 86 | 0.68% |
| 0.99 | 0.9953 | 97.3% | 28 | 0.23% |
| 0.97 | 0.99984 | 94.2% | 18 | 0.15% |
| 0.95 | 0.99999 | 90.3% | 4 | 0.035% |
| 0.90 | ≈ 1 | 84.6% | 1 | 0.009% |

Rejecting 10% of the images lowers the selective risk by a factor of about 20. The thresholds are extreme because softmax classifiers are overconfident: at $`q = 0.95`$ an answer is accepted only if about 0.001% of the probability is left to all other outputs.

**Negative results.** Test-time augmentation with nine views does not remove the remaining errors: on the `oe_bg` model at $`q = 0.97`$ the GTSRB false answers stay at 12, because the network repeats the same mistake on every view. Requiring the large CNN and ResNet-18 to agree, with the threshold recalibrated for the pair, reduces false answers on unseen foreign types (2 instead of 10 for `oe_bg` at $`q = 0.97`$) but not on known signs: on GTSRB the pair gives 19 instead of 12 (`oe_bg`, $`q = 0.97`$) and 13 instead of 4 (`oe_all`, $`q = 0.95`$). On a blurred known sign both networks make the same mistake.

## 4. Discussion

The answer to the title question is negative on all three routes. Used as is, large models are 38 points or more below a compact specialised network. Used as frozen feature extractors, they are 34 points below. Fully fine-tuned, the best of them ties with it, and at that point it is itself a specialised model: it needs the same labelled data and the same training procedure, and costs 5 times the latency and 16 times the memory.

Three observations explain the result. First, the task is visually simple and closed: 43 rigid, standardised objects. A network with a receptive field covering the input and 4.7M parameters already has enough capacity, so additional capacity has nothing to fit. Second, the task depends on fine distinctions that generic pretraining does not emphasise: mirror orientation, digits inside a speed limit, a diagonal stroke. These are exactly where zero-shot models fail. Third, the inputs are small (26 px and up), far from the resolution the large encoders were trained at.

For the open-set problem the practical conclusion is that false acceptances are controlled by data coverage. A network reliably rejects what it was shown as "unknown" and generalises only partly beyond it; the threshold then sets the operating point on the risk–coverage curve but cannot substitute for missing outlier data. The errors that remain are in-distribution ambiguities, such as a blurred "60" against "80", that cannot be resolved from one image. Agreement across consecutive video frames is the natural next step and was not tested.

## 5. Threats to validity

- **Range of large models.** The largest model tested has 4.4B parameters and runs locally. Frontier multimodal models available through an API were not evaluated. They may be more accurate zero-shot; their latency and cost per image are further from 1 ms, not closer.
- **Prompting.** One prompt wording per class was used for the zero-shot models. Prompt engineering may raise their accuracy; it is unlikely to close a 38-point gap but this was not measured.
- **Data regime.** The task has 26,640 labelled training images. With few examples per class the balance between pretrained and from-scratch models may differ.
- **Single runs.** Most models and all open-set variants were trained once. Adding the group B types to training changed the GTSRB false answers at $`q = 0.97`$ from 12 to 18, although these types have nothing to do with German signs. With one run per variant such a change cannot be separated from seed variance, so the reported 4 could as well be 8.
- **Seen outlier types.** Zero false answers on foreign signs was measured on types present in training (in other photos). On a type absent from every set the model may be as confidently wrong as it was on the "3.5 t" sign.
- **Outlier realism.** Background crops are random. In a full system the classifier receives what a detector mistook for a sign (round objects, adverts, lamps), which is harder.
- **One shifted domain.** Only Belgian signs were tested, and they are close to the German ones.
- **Hardware.** Latency was measured on one GPU at batch size 1; CPU and embedded targets were not measured.
- **Hyperparameters.** Learning rates were set per model family without a search, and pretrained models were trained for 6 epochs. Longer schedules may change the ranking inside the 99.0–99.5% band, which we already treat as unresolved.

## 6. Using the trained model

The final model ships with the repository (`models/big_cnn_oe_all.pt`, 18 MB). It needs only PyTorch and runs on a CPU.

```bash
pip install torch torchvision pillow
```

```bash
python predict.py path/to/sign.png
```

The input is a photo of one sign cropped from the frame with a small margin, as in GTSRB. The script does not search for signs in a full frame. The answer is one of 43 sign names or `unknown`, with the confidence:

```text
data/testset/images/14_00251.png: Stop (class 14), confidence 1.0000
data/belgium_unknown/test/images/be27_00330_00001.png: unknown, confidence 0.0000
```

`--keep` sets how strict the model is. It is the share of clean held-out German signs the model accepts: `0.99`, `0.97` (default), `0.95` or `0.9`. Lower means fewer false answers and more rejections; the final-model table in Section 3.4 uses `--keep 0.95`.

To get sample images, build the test set (downloads 89 MB):

```bash
python scripts/make_testset.py
```

## 7. Reproducing the results

The code is organised in four parts that correspond to RQ1–RQ4. Parts 1–4 need Linux with an NVIDIA GPU: 16 GB of GPU memory covers everything, and 4 GB is enough for the custom CNNs and ResNet-18. The scripts use CUDA directly and do not run on CPU or Apple Silicon. Tested with Python 3.14, PyTorch 2.15 (CUDA 13.2), torchvision 0.29 and transformers 5.24.

The raw results of our runs are already in `results/`. Each run script overwrites them and regenerates its table; set `SKIP_DONE=1` to keep results that exist. A fresh copy of this repository reproduced the shipped numbers exactly on the same machine. On other GPUs or library versions expect small differences: training is seeded but not bit-exact across setups, and accuracy moves by about 0.5 points between runs on the German test and up to 2.5 on the Belgian one.

### 7.1 Set up

```bash
git clone https://github.com/mdma4d/traffic-signs-specialist-vs-generalist.git && cd traffic-signs-specialist-vs-generalist
```

```bash
bash setup.sh
```

`setup.sh` creates `.venv` and installs `requirements.txt`.

### 7.2 Prepare the data

All datasets are downloaded by the scripts; nothing has to be fetched by hand.

| Command | Downloads | Builds | Needed for |
|---|---|---|---|
| `.venv/bin/python scripts/make_testset.py` | GTSRB test, 89 MB | `data/testset/`, 430 images | all parts |
| `.venv/bin/python scripts/make_belgium.py` | BelgiumTSC, 252 MB | `data/belgium/`, `data/belgium_train/` | parts 3 and 4 |
| `.venv/bin/python scripts/make_belgium_unknown.py` | nothing | `data/belgium_unknown/` | part 4 |
| `.venv/bin/python scripts/make_background.py` | GTSDB 1.1 GB, CIFAR-100 170 MB | `data/background/` | part 4 |

The GTSRB training split (about 190 MB) is downloaded by torchvision on the first training run.

### 7.3 Run

| Part | Command | What it does | Rough time on an RTX 4090 Laptop |
|---|---|---|---|
| 1 | `bash run_part1_baselines.sh` | CLIP, SigLIP 2, Qwen3-VL-4B and ResNet-18 on 430 images | 10 min, plus about 12 GB of model downloads |
| 2 | `bash run_part2_classifiers.sh` | fine-tunes and measures 22 classifiers | 1.5 h |
| 3 | `bash run_part3_transfer.sh` | retrains them and tests on Belgian signs | 2 h |
| 4 | `bash run_part4_reject.sh` | trains every reject variant, checks agreement rules | 40 min |

Each script accepts model names to run a subset. The fastest way to check the headline numbers:

```bash
bash run_part2_classifiers.sh big_cnn small_cnn resnet18_tv
```

```bash
bash run_part4_reject.sh final
```

The first command trains the custom CNNs and ResNet-18 in about 5 minutes. The second retrains the shipped model in about 6 minutes, prints its numbers for every threshold and overwrites `models/big_cnn_oe_all.pt`.

### 7.4 Read the tables

| Part | Table | Raw results |
|---|---|---|
| 1 | [results/report.md](results/report.md) | `results/*.json` |
| 2 | [results/classifiers.md](results/classifiers.md) | `results/classifiers/*.json` |
| 3 | [results/transfer.md](results/transfer.md) | `results/transfer/*.json` |
| 4 | [results/reject.md](results/reject.md) | `results/reject/*.json`, `results/agree_*.json`, `results/tta.json` |

The tables are rebuilt from the JSON files by `bench/report*.py`, which need only the Python standard library. The JSON files hold the answer for every image, so any number can be recomputed without a GPU.

### 7.5 Repository layout

```text
predict.py                 recognise a sign in an image, or answer "unknown"
models/big_cnn_oe_all.pt   the final model
setup.sh, requirements.txt environment
run_part1..4_*.sh          one script per part of the study
scripts/                   dataset preparation (downloads included)
bench/zeroshot.py, vlm.py  zero-shot CLIP / SigLIP 2 and the vision LLM
bench/classifier.py        the ResNet-18 reference of part 1
bench/finetune.py          all classifiers of parts 2 and 3, including the custom CNNs
bench/reject.py            classifier with a reject option, part 4
bench/agree.py, tta.py     two-model and multi-view agreement
bench/report*.py           build the tables in results/*.md
bench/chart.py             draws docs/accuracy.svg
results/                   raw results and tables of our runs
REPORT.md                  full write-up
```

## 8. References

The datasets are not redistributed here; the scripts download them from their official sources. Check their terms before using them for anything beyond research. `models/big_cnn_oe_all.pt` was trained on GTSRB, BelgiumTSC and GTSDB crops.

1. J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32:323–332, 2012. (GTSRB)
2. S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing, C. Igel. Detection of traffic signs in real-world images: The German Traffic Sign Detection Benchmark. *IJCNN*, 2013. (GTSDB)
3. R. Timofte, K. Zimmermann, L. Van Gool. Multi-view traffic sign detection, recognition, and 3D localisation. *Machine Vision and Applications*, 25(3):633–647, 2014. (BelgiumTSC)
4. A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009. (CIFAR-100)
5. A. Radford et al. Learning transferable visual models from natural language supervision. *ICML*, PMLR 139, 2021. (CLIP)
6. M. Tschannen et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786, 2025.
7. M. Oquab et al. DINOv2: Learning robust visual features without supervision. arXiv:2304.07193, 2023.
8. O. Siméoni et al. DINOv3. arXiv:2508.10104, 2025.
9. Qwen Team. Qwen3-VL-4B-Instruct, model card. Hugging Face, `Qwen/Qwen3-VL-4B-Instruct`.
10. D. Hendrycks, M. Mazeika, T. Dietterich. Deep anomaly detection with outlier exposure. *ICLR*, 2019.
11. D. Hendrycks, K. Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*, 2017.
12. Y. Geifman, R. El-Yaniv. Selective classification for deep neural networks. *NeurIPS*, 2017.
13. I. Loshchilov, F. Hutter. Decoupled weight decay regularization. *ICLR*, 2019. (AdamW)
14. L. N. Smith, N. Topin. Super-convergence: Very fast training of neural networks using large learning rates. arXiv:1708.07120, 2017. (one-cycle schedule)
15. K. Simonyan, A. Zisserman. Very deep convolutional networks for large-scale image recognition. *ICLR*, 2015.

## License

The code is released under the MIT license, see [LICENSE](LICENSE).
