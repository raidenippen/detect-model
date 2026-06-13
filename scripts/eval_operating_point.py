"""
Evaluate a trained model at its CALIBRATED operating point (not the raw 0.5).

benchmark_baselines.py reports flag rates at a fixed --threshold (0.5), which is
right for comparing raw baselines but not for judging our calibrated model: the
acceptance criteria (TRAINING.md section 4) are defined at the operating
threshold written to inference_config.json (temperature scaling + threshold_ai).

This scores val_real / val_ai / hard_negatives / hard_positives under the
benchmark degradations plus the social-media presets from degrade.py, applies
the calibration, and prints FPR/TPR at the operating point, including the share
of images landing in the uncertain band (cascade -> Hive in production).

Usage:
  python scripts/eval_operating_point.py models/<ts>/best
  python scripts/eval_operating_point.py models/<ts>/best --degradations clean jpeg70_half pinterest_feed
"""

import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from degrade import apply_preset, PRESETS
from benchmark_baselines import degrade as benchmark_degrade, list_images, DIRS

BENCH_DEGS = ("clean", "jpeg70", "half", "jpeg70_half")


def degraded(img: Image.Image, mode: str) -> Image.Image:
    if mode in BENCH_DEGS:
        return benchmark_degrade(img, mode)
    return apply_preset(img, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="e.g. models/<ts>/best (inference_config.json is read from its parent)")
    parser.add_argument("--limit", type=int, default=300, help="max images per val dir (hard sets always run in full)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--degradations", nargs="*",
                        default=["clean", "jpeg70", "half", "jpeg70_half", "pinterest_feed"],
                        choices=list(BENCH_DEGS) + list(PRESETS))
    args = parser.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    best = Path(args.model_dir)
    cfg = json.loads((best.parent / "inference_config.json").read_text())
    temperature, threshold = cfg["temperature"], cfg["threshold_ai"]
    band_lo, band_hi = cfg["uncertain_band"]
    ai_idx = next(int(k) for k, v in cfg["labels"].items() if v == "ai")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(best).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(best)

    @torch.no_grad()
    def score(images):
        probs = []
        for i in range(0, len(images), args.batch_size):
            batch = processor(images=images[i:i + args.batch_size], return_tensors="pt").to(device)
            p = torch.softmax(model(**batch).logits / temperature, dim=-1)[:, ai_idx]
            probs.extend(p.tolist())
        return probs

    print(f"model={best}  T={temperature}  threshold_ai={threshold}  band=[{band_lo}, {band_hi}]")
    results = {}
    for dir_name, (directory, true_label) in DIRS.items():
        limit = args.limit if dir_name.startswith("val_") else 10 ** 9
        files = list_images(directory, limit) if directory.exists() else []
        if not files:
            print(f"  {dir_name}: empty, skipping")
            continue
        for deg in args.degradations:
            images = []
            for f in tqdm(files, desc=f"  {dir_name}/{deg}", leave=False):
                try:
                    with Image.open(f) as img:
                        images.append(degraded(img.convert("RGB"), deg))
                except Exception:
                    continue
            probs = score(images)
            flagged = sum(p >= threshold for p in probs) / len(probs)
            uncertain = sum(band_lo <= p < band_hi for p in probs) / len(probs)
            metric = "FPR" if true_label == "real" else "TPR"
            results.setdefault(dir_name, {})[deg] = {
                "n": len(probs), metric: round(flagged, 4), "uncertain": round(uncertain, 4),
                "mean_cal_score": round(sum(probs) / len(probs), 4),
            }
            print(f"  {dir_name:<16} {deg:<16} n={len(probs):<5} {metric}@op={flagged:.4f}  uncertain={uncertain:.4f}")

    out = best.parent / "operating_point_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
