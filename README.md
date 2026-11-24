# Traffic sign recognition: what works, measured

A small CNN trained from scratch in three and a half minutes recognises German traffic signs with 99.5% accuracy in 1.1 ms per image, and it can answer "unknown" instead of guessing. Large general-purpose models used without training lose to it on accuracy, speed and memory at once.

This repository contains the code, the raw results and a trained model, so every number below can be reproduced.

![Accuracy, speed and GPU memory of seven approaches](docs/accuracy.svg)

- [Try the trained model](#try-the-trained-model)
- [What we did and what we found](#what-we-did-and-what-we-found)
- [Reproduce the results](#reproduce-the-results)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)

The full write-up with methodology, every table and all caveats is in [REPORT.md](REPORT.md).

## Try the trained model

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

`--keep` sets how strict the model is. It is the share of clean held-out German signs the model accepts: `0.99`, `0.97` (default), `0.95` or `0.9`. Lower means fewer false answers and more rejections; the "zero false answers" table below uses `--keep 0.95`.

To get sample images, build the test set (downloads 89 MB):

```bash
python scripts/make_testset.py
```

## What we did and what we found

The task is the same everywhere: the input is a cropped photo of one sign, the output is one of the 43 classes of the German benchmark [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html). Detecting signs in a frame is out of scope. Everything ran on a laptop with an RTX 4090 Laptop GPU.

### Step 1. Is training needed at all?

We first checked whether ready-made models can do the job when given only the sign names and no labelled examples, and compared them with a plain ResNet-18 fine-tuned on signs.

| Approach | Trained on signs | Accuracy (430 images) | Time per image | GPU memory |
|---|---|---|---|---|
| ResNet-18 classifier | yes, half a minute | 97.2–99.5% | 1.7 ms | 0.2 GB |
| SigLIP 2 So400m | no | 60.7% | 20.7 ms | 2.2 GB |
| Vision LLM, Qwen3-VL-4B | no | 52.6% | 199 ms | 8.6 GB |
| CLIP L/14 | no | 39.1% | 11.3 ms | 0.8 GB |

The best zero-shot model gets four images out of ten wrong. These models confuse left and right, fail on crossed-out signs and degrade badly on small images: on signs up to 35 pixels wide all three score 27–38%. The vision LLM is the most expensive option and not the most accurate one.

### Step 2. Which classifier?

Next we compared 21 trainable classifiers on the full test split of 12,630 images, from a custom network with 0.3M parameters to ViT-B with 86M.

- **Accuracy tops out at 99.0–99.5%.** Eight variants land in this band.
- **Differences inside the band are noise.** The same ResNet-18 training repeated four times gave 98.8% to 99.3%.
- **Large models pay several times over for fractions of a percent.** Fine-tuned CLIP B/16 is 3 times slower than ResNet-18 and uses 7 times more GPU memory.
- **Pretraining turned out to be optional.** A custom small network trained from scratch reached 99.0% with 6 MB of GPU memory and 0.8 ms per image.

We then scaled the custom network up to 4.7M parameters. It gives 99.4–99.5% in every run, the same as the best large model, at 1.1 ms per image and 83 MB of GPU memory.

### Step 3. What happens on signs from another country?

Models trained on German signs were tested on 940 Belgian images ([BelgiumTSC](https://btsd.ethz.ch/shareddata/)) shot with a different camera, without any further training.

| Model | Germany | Belgium |
|---|---|---|
| CLIP B/16, fine-tuned | 99.5% | 98.2% |
| Custom large CNN | 99.5% | 96.9% |
| Custom small CNN | 99.0% | 94.9% |
| ResNet-18 | 99.0% | 94.8% |
| SigLIP 2 So400m, zero-shot | n/a | 75.4% |

Every model loses accuracy: the best ones 1–4 points, lightweight mobile networks up to 11. Most of the loss comes from signs that are drawn differently in Belgium, not from the camera: "Children crossing" is recognised 72% of the time and "Bicycles crossing" 7%, while signs with identical artwork transfer at 98–100%. Run-to-run variance reaches 2.5 points here, so this table cannot separate CLIP from the custom large CNN.

### Step 4. A model that can say "unknown"

A plain classifier must pick one of 43 classes even when the sign is blurred, unfamiliar, or not a sign at all. We trained a single network to answer either with a sign or with "unknown" by adding a 44th output and a confidence threshold.

What mattered most was what the network was shown as "unknown":

- **A confidence threshold alone** lets foreign signs through. ResNet-18 confidently names one in five of them as a known sign.
- **Only foreign Belgian signs** in training make the network reject every Belgian photo, known signs included. Belgian photos of known signs had to go into training too.
- **Road-scene crops without signs** in training remove false answers on background and also on random objects the network never saw.
- **All foreign sign types** in training remove the last confident errors. Before that the network took a "3.5 t weight limit" sign for "Speed limit 30", because it had never seen one.

The final model at the strict threshold (`--keep 0.95`):

| Test set | Correct | False answers | Rejected |
|---|---|---|---|
| German test, 12,630 images | 90.3% | 4 images (0.03%) | 9.7% |
| Belgian signs of known classes, 940 | 96.1% | 0 | 3.9% |
| Foreign signs, 983 | n/a | 0 | 100% |
| Road background crops, 2,000 | n/a | 0 | 100% |
| Random objects, 2,000 | n/a | 0 | 100% |

The threshold is adjustable: at the default (`--keep 0.97`) the model rejects 6% of German test images and gives 18 false answers on them.

Two attempts to remove the remaining errors failed. Looking at each image nine times with small distortions does nothing, because the network makes the same mistake on every view. Requiring two different networks to agree helps on unfamiliar input (2 false answers instead of 10 on foreign signs of unseen types) but not on blurred known signs, where both make the same mistake.

### Takeaways

- **With labelled data, use a small trained classifier.** Zero-shot models are fine for a prototype, not for production.
- **Model size buys almost nothing here.** A 4.7M-parameter network matches models almost 20 times larger.
- **False answers are fixed with data, not with a threshold.** The network reliably rejects only what it was shown as "unknown" during training.
- **Some errors cannot be removed from a single image.** A blurred "60" does look like "80". The next lever is agreement across neighbouring video frames, which we did not test.

### What these results do not show

- **Zero false answers on foreign signs was measured on sign types the network had seen** (in other photos). On a type that is in none of the sets it may be as confidently wrong as it was on "3.5 t".
- **The reject models were trained once each.** The number of false answers on the German test moved from 12 to 18 between two trainings of the same network, so 4 could as well be 8.
- **Background crops are random.** In a real system the classifier receives whatever a detector took for a sign: round objects, adverts, street lamps. Expect more false answers on those.
- **Only one foreign country was tested,** and its signs are close to the German ones.
- **Speed was measured on a GPU only.**

## Reproduce the results

Parts 1–4 need Linux with an NVIDIA GPU: 16 GB of GPU memory covers everything, and 4 GB is enough for the custom CNNs and ResNet-18. The scripts use CUDA directly and do not run on CPU or Apple Silicon. Tested with Python 3.14, PyTorch 2.15 (CUDA 13.2), torchvision 0.29 and transformers 5.24.

The raw results of our runs are already in `results/`. Each run script overwrites them and regenerates its table; set `SKIP_DONE=1` to keep results that exist. A fresh copy of this repository reproduced the shipped numbers exactly on the same machine. On other GPUs or library versions expect small differences: training is seeded but not bit-exact across setups, and accuracy moves by about 0.5 points between runs on the German test and up to 2.5 on the Belgian one.

### 1. Set up

```bash
git clone <this repository> && cd <its folder>
```

```bash
bash setup.sh
```

`setup.sh` creates `.venv` and installs `requirements.txt`.

### 2. Prepare the data

All datasets are downloaded by the scripts; nothing has to be fetched by hand.

| Command | Downloads | Builds | Needed for |
|---|---|---|---|
| `.venv/bin/python scripts/make_testset.py` | GTSRB test, 89 MB | `data/testset/`, 430 images | all parts |
| `.venv/bin/python scripts/make_belgium.py` | BelgiumTSC, 252 MB | `data/belgium/`, `data/belgium_train/` | parts 3 and 4 |
| `.venv/bin/python scripts/make_belgium_unknown.py` | nothing | `data/belgium_unknown/` | part 4 |
| `.venv/bin/python scripts/make_background.py` | GTSDB 1.1 GB, CIFAR-100 170 MB | `data/background/` | part 4 |

The GTSRB training split (about 190 MB) is downloaded by torchvision on the first training run.

### 3. Run

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

### 4. Read the tables

| Part | Table | Raw results |
|---|---|---|
| 1 | [results/report.md](results/report.md) | `results/*.json` |
| 2 | [results/classifiers.md](results/classifiers.md) | `results/classifiers/*.json` |
| 3 | [results/transfer.md](results/transfer.md) | `results/transfer/*.json` |
| 4 | [results/reject.md](results/reject.md) | `results/reject/*.json`, `results/agree_*.json`, `results/tta.json` |

The tables are rebuilt from the JSON files by `bench/report*.py`, which need only the Python standard library. The JSON files hold the answer for every image, so any number can be recomputed without a GPU.

## Repository layout

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

The custom CNN is about ten lines of PyTorch: blocks of two 3×3 convolutions with batch norm and ReLU followed by max pooling, then global average pooling, dropout and a linear layer. See `small_cnn` in [bench/finetune.py](bench/finetune.py).

## Datasets

The datasets are not redistributed here; the scripts download them from their official sources. Check their terms before using them for anything beyond research.

- **GTSRB**, German Traffic Sign Recognition Benchmark: Stallkamp et al., 2012. Training and test images of 43 classes.
- **GTSDB**, German Traffic Sign Detection Benchmark: Houben et al., 2013. Full road frames, used here as a source of background crops.
- **BelgiumTSC**: Timofte, Zimmermann and Van Gool, 2014. Belgian signs, used as the other-country test and as "unknown" signs.
- **CIFAR-100**: Krizhevsky, 2009. Used as random non-sign objects.

`models/big_cnn_oe_all.pt` was trained on GTSRB, BelgiumTSC and GTSDB crops.

## License

The code is released under the MIT license, see [LICENSE](LICENSE).
