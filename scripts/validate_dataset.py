"""
Dataset validation script — run before training to confirm diversity and quality.

Checks:
  1. Image counts and source balance
  2. Duplicate detection via perceptual hashing
  3. Color distribution (brightness, saturation, contrast)
  4. Scene diversity via CLIP zero-shot classification
  5. Image statistics overlap (real vs AI distributions)
  6. Random sample grid for manual inspection

Usage:
  python validate_dataset.py

Output:
  - Prints a full report to terminal
  - Saves sample grids to data/validation_report/
  - Exits with code 1 if any hard failures are found (too few images, too many dupes)

Requirements:
  pip install Pillow numpy tqdm torch transformers matplotlib imagehash
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional imports — degrade gracefully if not installed
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("WARNING: imagehash not installed — skipping duplicate detection. pip install imagehash")

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("WARNING: torch/transformers not installed — skipping scene classification.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed — skipping visual report. pip install matplotlib")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent / "data"
REAL_DIR    = ROOT / "real"
AI_DIR      = ROOT / "ai"
VAL_REAL    = ROOT / "val_real"
VAL_AI      = ROOT / "val_ai"
REPORT_DIR  = ROOT / "validation_report"
REPORT_DIR.mkdir(exist_ok=True)

# Minimum acceptable counts — script exits with error if below these
MIN_REAL    = 3000
MIN_AI      = 3000
MAX_DUPE_PC = 0.05   # fail if >5% duplicates

# CLIP scene categories to check diversity
SCENE_CATEGORIES = [
    "a portrait photo of a person",
    "a concert or live music performance",
    "a landscape or nature photo",
    "a food or product photo",
    "a sports or action photo",
    "a street or urban photography",
    "a wedding or event photo",
    "a wildlife or animal photo",
    "an indoor scene",
    "an abstract or artistic photo",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_images(directory: Path, limit: int = 2000):
    """Load up to `limit` images from a directory, return list of (path, PIL image)."""
    paths = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
    random.shuffle(paths)
    paths = paths[:limit]
    images = []
    for p in tqdm(paths, desc=f"  Loading {directory.name}", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            images.append((p, img))
        except Exception:
            continue
    return images


def image_stats(img: Image.Image) -> dict:
    """Return brightness, contrast, saturation for an image."""
    arr = np.array(img).astype(np.float32)
    brightness = arr.mean() / 255.0
    contrast   = arr.std() / 255.0
    # Saturation: convert to HSV-like, measure S channel
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat  = np.where(cmax > 0, (cmax - cmin) / cmax, 0).mean()
    return {"brightness": brightness, "contrast": contrast, "saturation": sat}


def source_from_filename(name: str) -> str:
    """Infer collection source from filename prefix."""
    if name.startswith("oi_"):       return "open_images"
    if name.startswith("coco_"):     return "coco"
    if name.startswith("wiki_"):     return "wikimedia"
    if name.startswith("flickr_"):   return "flickr"
    if name.startswith("flux_"):     return "flux_local"
    if name.startswith("replicate_"): return "replicate"
    if name.startswith("dalle"):     return "dalle3"
    return "unknown"


# ---------------------------------------------------------------------------
# Check 1: Counts and source balance
# ---------------------------------------------------------------------------
def check_counts():
    print("\n── 1. Image Counts ─────────────────────────────────────────")
    results = {}
    hard_fail = False

    for label, directory in [("real", REAL_DIR), ("ai", AI_DIR),
                               ("val_real", VAL_REAL), ("val_ai", VAL_AI)]:
        count = len(list(directory.glob("*.jpg"))) + len(list(directory.glob("*.png")))
        results[label] = count
        status = "✓" if count > 0 else "✗ EMPTY"
        print(f"  {label:<12}: {count:>6} images  {status}")

    total_real = results["real"] + results["val_real"]
    total_ai   = results["ai"]   + results["val_ai"]
    print(f"\n  Total real (train+val): {total_real}")
    print(f"  Total AI   (train+val): {total_ai}")

    if total_real < MIN_REAL:
        print(f"  ✗ FAIL: Need at least {MIN_REAL} real images, have {total_real}")
        hard_fail = True
    if total_ai < MIN_AI:
        print(f"  ✗ FAIL: Need at least {MIN_AI} AI images, have {total_ai}")
        hard_fail = True

    ratio = total_real / total_ai if total_ai > 0 else 0
    if ratio < 0.7 or ratio > 1.4:
        print(f"  ⚠ WARNING: Class imbalance — real/AI ratio is {ratio:.2f} (want 0.7–1.4)")
    else:
        print(f"  ✓ Class balance: real/AI ratio {ratio:.2f}")

    return results, hard_fail


def check_source_balance():
    print("\n── 2. Source Balance ───────────────────────────────────────")
    for label, directory in [("real", REAL_DIR), ("ai", AI_DIR)]:
        paths   = list(directory.glob("*.jpg"))
        sources = Counter(source_from_filename(p.stem) for p in paths)
        total   = sum(sources.values())
        if total == 0:
            continue
        print(f"\n  {label}:")
        for source, count in sorted(sources.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            bar = "█" * int(pct / 2)
            warn = " ⚠ DOMINANT" if pct > 60 else ""
            print(f"    {source:<20}: {count:>5} ({pct:>5.1f}%) {bar}{warn}")


# ---------------------------------------------------------------------------
# Check 2: Duplicate detection
# ---------------------------------------------------------------------------
def check_duplicates(real_images, ai_images):
    print("\n── 3. Duplicate Detection ──────────────────────────────────")
    if not HAS_IMAGEHASH:
        print("  Skipped — imagehash not installed.")
        return False

    hard_fail = False
    for label, images in [("real", real_images), ("ai", ai_images)]:
        hashes = {}
        dupes  = 0
        for path, img in tqdm(images, desc=f"  Hashing {label}", leave=False):
            try:
                h = str(imagehash.phash(img))
                if h in hashes:
                    dupes += 1
                else:
                    hashes[h] = path
            except Exception:
                continue
        dupe_pct = dupes / len(images) if images else 0
        status   = "✓" if dupe_pct < MAX_DUPE_PC else "✗ FAIL"
        print(f"  {label}: {dupes} duplicates / {len(images)} images ({dupe_pct*100:.1f}%) {status}")
        if dupe_pct >= MAX_DUPE_PC:
            hard_fail = True

    return hard_fail


# ---------------------------------------------------------------------------
# Check 3: Image statistics distributions
# ---------------------------------------------------------------------------
def check_statistics(real_images, ai_images):
    print("\n── 4. Image Statistics (real vs AI distributions) ──────────")

    def compute_stats(images):
        stats = [image_stats(img) for _, img in tqdm(images, desc="  Computing stats", leave=False)]
        return {
            "brightness": [s["brightness"] for s in stats],
            "contrast":   [s["contrast"]   for s in stats],
            "saturation": [s["saturation"] for s in stats],
        }

    real_stats = compute_stats(real_images)
    ai_stats   = compute_stats(ai_images)

    print(f"\n  {'Metric':<12} {'Real mean':>10} {'AI mean':>10} {'Real std':>10} {'AI std':>10}")
    print(f"  {'-'*52}")
    for metric in ["brightness", "contrast", "saturation"]:
        r_mean = np.mean(real_stats[metric])
        a_mean = np.mean(ai_stats[metric])
        r_std  = np.std(real_stats[metric])
        a_std  = np.std(ai_stats[metric])
        diff   = abs(r_mean - a_mean)
        warn   = " ⚠ large gap" if diff > 0.15 else ""
        print(f"  {metric:<12} {r_mean:>10.3f} {a_mean:>10.3f} {r_std:>10.3f} {a_std:>10.3f}{warn}")

    # Save distribution plots
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, metric in enumerate(["brightness", "contrast", "saturation"]):
            axes[i].hist(real_stats[metric], bins=40, alpha=0.6, label="Real", color="green")
            axes[i].hist(ai_stats[metric],   bins=40, alpha=0.6, label="AI",   color="red")
            axes[i].set_title(metric.capitalize())
            axes[i].legend()
        plt.suptitle("Image Statistics Distribution: Real vs AI")
        plt.tight_layout()
        out = REPORT_DIR / "statistics_distribution.png"
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"\n  Saved distribution plot: {out}")

    return real_stats, ai_stats


# ---------------------------------------------------------------------------
# Check 4: Scene diversity via CLIP
# ---------------------------------------------------------------------------
def check_scene_diversity(real_images, ai_images):
    print("\n── 5. Scene Diversity (CLIP) ───────────────────────────────")
    if not HAS_CLIP:
        print("  Skipped — torch/transformers not installed.")
        return

    print("  Loading CLIP model...")
    try:
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device    = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model     = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  Failed to load CLIP: {e}")
        return

    def classify_batch(images, label):
        category_counts = Counter()
        sample = random.sample(images, min(500, len(images)))
        for _, img in tqdm(sample, desc=f"  Classifying {label}", leave=False):
            try:
                inputs  = processor(text=SCENE_CATEGORIES, images=img, return_tensors="pt", padding=True)
                inputs  = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                probs   = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                top_cat = SCENE_CATEGORIES[probs.argmax()]
                category_counts[top_cat] += 1
            except Exception:
                continue
        return category_counts

    print(f"\n  Real image scene distribution (sample of 500):")
    real_cats = classify_batch(real_images, "real")
    total_r   = sum(real_cats.values())
    for cat, count in real_cats.most_common():
        pct = 100 * count / total_r
        bar = "█" * int(pct / 2)
        print(f"    {cat[:45]:<45}: {pct:>5.1f}% {bar}")

    print(f"\n  AI image scene distribution (sample of 500):")
    ai_cats = classify_batch(ai_images, "AI")
    total_a = sum(ai_cats.values())
    for cat, count in ai_cats.most_common():
        pct = 100 * count / total_a
        bar = "█" * int(pct / 2)
        print(f"    {cat[:45]:<45}: {pct:>5.1f}% {bar}")

    # Check for missing categories
    missing = []
    for cat in SCENE_CATEGORIES:
        r_pct = 100 * real_cats.get(cat, 0) / total_r if total_r else 0
        if r_pct < 2.0:
            missing.append(cat)
    if missing:
        print(f"\n  ⚠ Under-represented in real images (< 2%):")
        for cat in missing:
            print(f"    - {cat}")
    else:
        print(f"\n  ✓ All scene categories represented in real images")


# ---------------------------------------------------------------------------
# Check 5: Random sample grid for manual inspection
# ---------------------------------------------------------------------------
def save_sample_grids(real_images, ai_images):
    if not HAS_MATPLOTLIB:
        print("\n── 6. Sample Grids ─────────────────────────────────────────")
        print("  Skipped — matplotlib not installed.")
        return

    print("\n── 6. Sample Grids ─────────────────────────────────────────")

    def save_grid(images, label, n=64):
        sample = random.sample(images, min(n, len(images)))
        cols   = 8
        rows   = len(sample) // cols + (1 if len(sample) % cols else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
        for i, (path, img) in enumerate(sample):
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(path.stem[:10], fontsize=6)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"{label} — random sample of {len(sample)}", fontsize=12)
        plt.tight_layout()
        out = REPORT_DIR / f"sample_{label}.png"
        plt.savefig(out, dpi=80)
        plt.close()
        print(f"  Saved: {out}")

    save_grid(real_images, "real")
    save_grid(ai_images,   "ai")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  detect-model Dataset Validation Report")
    print("=" * 60)

    # Check counts first — bail early if nothing collected yet
    count_results, count_fail = check_counts()
    check_source_balance()

    total_real = count_results.get("real", 0) + count_results.get("val_real", 0)
    total_ai   = count_results.get("ai",   0) + count_results.get("val_ai",   0)

    if total_real == 0 and total_ai == 0:
        print("\n  No images found. Run collect_real.py and collect_ai.py first.")
        sys.exit(1)

    # Load samples for deeper analysis (cap at 2000 each for speed)
    print("\n  Loading image samples for analysis...")
    real_images = load_images(REAL_DIR, limit=2000)
    ai_images   = load_images(AI_DIR,   limit=2000)

    dupe_fail = False
    if real_images or ai_images:
        dupe_fail = check_duplicates(real_images, ai_images)
        check_statistics(real_images, ai_images)
        check_scene_diversity(real_images, ai_images)
        save_sample_grids(real_images, ai_images)

    # Final summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    hard_fail = count_fail or dupe_fail
    if hard_fail:
        print("  ✗ HARD FAILURES — fix before training")
    else:
        print("  ✓ Dataset passes all hard checks")
    print(f"\n  Visual report saved to: {REPORT_DIR}/")
    print("  Review sample_real.png and sample_ai.png manually before training.")
    print("=" * 60)

    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
