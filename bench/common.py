"""Shared helpers: reading a test set and writing results."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data/testset"
RESULTS = ROOT / "results"


def load_classes():
    """List of (class_id, prompt_name, display_name), sorted by class_id.

    prompt_name is the lowercase English name fed to zero-shot models; display_name is for reports.
    """
    with (ROOT / "data/classes.csv").open(newline="") as f:
        rows = [(int(r["class_id"]), r["name_en"], r["name"]) for r in csv.DictReader(f)]
    return sorted(rows)


def load_testset(folder=TESTSET):
    """List of (image path, class_id). folder is a dataset folder such as data/testset or data/belgium."""
    with (folder / "labels.csv").open(newline="") as f:
        return [(folder / "images" / r["filename"], int(r["class_id"])) for r in csv.DictReader(f)]


def start_inference_meter():
    """Call right before the inference loop: resets the peak GPU memory counter."""
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def model_stats(net):
    """Parameter count (millions) and weight size as stored in GPU memory (MB)."""
    params = sum(p.numel() for p in net.parameters())
    size = sum(p.numel() * p.element_size() for p in net.parameters())
    return {"params_m": params / 1e6, "weights_mb": size / 2**20}


def save_result(name, model, preds, infer_seconds, net=None, **extra):
    """preds: list of {filename, true, pred}. net: the model, used to measure its size."""
    import resource
    import torch
    correct = sum(p["true"] == p["pred"] for p in preds)
    result = {
        "name": name,
        "model": model,
        "n": len(preds),
        "accuracy": correct / len(preds),
        "ms_per_image": 1000 * infer_seconds / len(preds),
        # peak GPU memory during inference (weights + activations)
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
        # peak process RAM over the whole run (ru_maxrss is in KB on Linux)
        "ram_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        **(model_stats(net) if net is not None else {}),
        **extra,
        "preds": preds,
    }
    (RESULTS / f"{name}.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"[{name}] accuracy {result['accuracy']:.1%} ({correct}/{len(preds)}), "
          f"{result['ms_per_image']:.1f} ms/image, GPU memory {result['vram_peak_mb']:.0f} MB, "
          f"RAM {result['ram_peak_mb']:.0f} MB")
