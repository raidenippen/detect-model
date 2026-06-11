"""
Benchmark baseline detectors before training anything. This is Step 2 (EDA &
baseline) and the bar our model has to beat.

What it does:
  - Runs each baseline HF image-classification model over:
      data/val_real        -> measures FPR (false positives on real photos)
      data/val_ai          -> measures TPR (recall on AI images)
      data/hard_negatives  -> FPR on the curated hard cases (concert photos,
                              bokeh portraits — the images that triggered this
                              project). Put the actual failing photos here.
  - Repeats every image under a degradation suite (JPEG-70, half-res, both) —
    production images arrive recompressed through CDNs/social media, so clean
    numbers alone are misleading.

Usage:
  python benchmark_baselines.py
  python benchmark_baselines.py --models NYUAD-ComNets/NYUAD_AI-generated_images_detector
  python benchmark_baselines.py --limit 500 --threshold 0.5

Output: a per-model, per-directory, per-degradation table of mean AI score and
flag rate, written to terminal and data/validation_report/baselines.json
"""

import io
import json
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent / "data"
DIRS = {
    "val_real":       (ROOT / "val_real", "real"),
    "val_ai":         (ROOT / "val_ai", "ai"),
    "hard_negatives": (ROOT / "hard_negatives", "real"),
    "hard_positives": (ROOT / "hard_positives", "ai"),  # deceptive AI baselines miss
}
REPORT = ROOT / "validation_report"
REPORT.mkdir(exist_ok=True)

# The models currently used by the Detect extension, the NYUAD candidate, and
# the Community Forensics detector. Loading failures are skipped with a warning
# — fix ids with --models if any of these have moved.
DEFAULT_MODELS = [
    "NYUAD-ComNets/NYUAD_AI-generated_images_detector",
    "Ateeqq/ai-vs-human-image-detector",
    "haywoodsloan/ai-image-detector-deploy",
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
]

AI_WORDS   = ("ai", "artificial", "fake", "generated", "deepfake", "synthetic",
              "dalle", "stable", "diffusion", "midjourney")
REAL_WORDS = ("real", "human", "authentic", "nature", "photo")

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def ai_score(predictions) -> float:
    """Map a list of {label, score} to P(ai), robust to different label schemes
    (binary ai/real, NYUAD's 3-class dalle/sd/real, etc.)."""
    ai_p, real_p = 0.0, 0.0
    for p in predictions:
        label = p["label"].lower()
        if any(w in label for w in AI_WORDS):
            ai_p += p["score"]
        elif any(w in label for w in REAL_WORDS):
            real_p += p["score"]
    if ai_p == 0.0 and real_p > 0.0:
        return 1.0 - real_p
    return ai_p


def degrade(img: Image.Image, mode: str) -> Image.Image:
    if mode == "clean":
        return img
    if mode in ("jpeg70", "jpeg70_half"):
        if mode == "jpeg70_half":
            img = img.resize((max(1, img.width // 2), max(1, img.height // 2)),
                             Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=70)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if mode == "half":
        return img.resize((max(1, img.width // 2), max(1, img.height // 2)),
                          Image.LANCZOS)
    raise ValueError(mode)


def list_images(d: Path, limit: int):
    files = []
    for pat in EXTS:
        files.extend(d.glob(pat))
    files.sort()
    return files[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=300,
                        help="Max images per directory (keep runtimes sane)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--degradations", nargs="*",
                        default=["clean", "jpeg70", "half", "jpeg70_half"])
    args = parser.parse_args()

    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)

    results = {}
    for model_id in args.models:
        print(f"\n=== {model_id} ===")
        try:
            clf = pipeline("image-classification", model=model_id, device=device)
        except Exception as e:
            print(f"  SKIPPED — failed to load: {e}")
            continue

        results[model_id] = {}
        for dir_name, (directory, true_label) in DIRS.items():
            files = list_images(directory, args.limit) if directory.exists() else []
            if not files:
                note = "MISSING — add curated hard cases here" if dir_name == "hard_negatives" else "empty"
                print(f"  {dir_name:<16}: {note}, skipping")
                continue

            for deg in args.degradations:
                scores = []
                for f in tqdm(files, desc=f"  {dir_name}/{deg}", leave=False):
                    try:
                        with Image.open(f) as img:
                            preds = clf(degrade(img.convert("RGB"), deg))
                        scores.append(ai_score(preds))
                    except Exception:
                        continue
                if not scores:
                    continue
                flag_rate = sum(s >= args.threshold for s in scores) / len(scores)
                mean = sum(scores) / len(scores)
                # For real dirs flag_rate IS the FPR; for ai dirs it's the TPR
                metric = "FPR" if true_label == "real" else "TPR"
                results[model_id].setdefault(dir_name, {})[deg] = {
                    "n": len(scores), "mean_ai_score": round(mean, 4),
                    metric: round(flag_rate, 4),
                }
                print(f"  {dir_name:<16} {deg:<12} n={len(scores):<5} "
                      f"mean={mean:.3f}  {metric}@{args.threshold}={flag_rate:.3f}")

    out = REPORT / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("\nReading the table:")
    print("  - val_real / hard_negatives: FPR should be LOW. The extension's current")
    print("    models are expected to fail hard on hard_negatives — that's the point.")
    print("  - val_ai: TPR should be HIGH, and should NOT collapse under jpeg70_half.")
    print("  - Our trained model must beat every row, especially degraded hard_negatives.")


if __name__ == "__main__":
    main()
