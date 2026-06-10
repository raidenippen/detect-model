"""
Leakage probe — run BEFORE training. Go/no-go gate for the dataset.

Two checks, both using only LOW-LEVEL features (channel stats, sharpness,
FFT band energies, JPEG blockiness) computed on images preprocessed exactly
like training views (random downscale + crop + degradation chain):

  1. SOURCE probe (real images only): can a linear model tell which source a
     real photo came from (unsplash vs coco vs wikimedia...)? If yes, the
     detector can shortcut on source fingerprints instead of AI-ness, because
     our real and AI classes come from disjoint pipelines.

  2. SHORTCUT probe (real vs ai): can a linear model separate the classes
     from these 12 statistics alone? If yes, the dataset has a trivial
     low-level tell (resolution, compression, brightness) and the trained
     model's val numbers will be inflated.

Heuristics: chance*1.5 macro accuracy on (1), 70% on (2) -> WARN. These are
gates for attention, not proofs — read the per-source confusion before
deciding.

Usage:
  python leakage_probe.py --per-source 400
"""

import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import degrade

ROOT = Path(__file__).parent.parent / "data"
EXTS = (".jpg", ".jpeg", ".png", ".webp")

SOURCE_PREFIXES = [
    ("unsplash_", "unsplash"), ("pexels_", "pexels"), ("wiki_", "wikimedia"),
    ("coco_", "coco"), ("oi_", "open_images"), ("cfreal_", "cf_real"),
    ("flux_", "flux"), ("cf_", "community_forensics"), ("rep_", "replicate"),
    ("gptimg_", "gpt_image_1"), ("dalle", "dalle3"),
]


def source_of(name: str) -> str:
    for prefix, src in SOURCE_PREFIXES:
        if name.startswith(prefix):
            return src
    return "unknown"


def training_view(img: Image.Image, rng: random.Random, size=224) -> Image.Image:
    """Mirror the train.py pipeline: random shorter-side downscale, random
    crop to size, degradation chain with the same probability."""
    short = min(img.size)
    target = rng.randint(384, 768)
    if short > target:
        s = target / short
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                         Image.BICUBIC)
    if img.width > size and img.height > size:
        x = rng.randint(0, img.width - size)
        y = rng.randint(0, img.height - size)
        img = img.crop((x, y, x + size, y + size))
    else:
        img = img.resize((size, size), Image.BICUBIC)
    if rng.random() < 0.7:
        img = degrade.random_chain(img, rng)
    return img


def features(img: Image.Image) -> np.ndarray:
    """12 low-level stats: ch means/stds (6), gradient sharpness (1),
    FFT radial band energies (4), JPEG 8x8 blockiness (1)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    f = list(arr.mean(axis=(0, 1))) + list(arr.std(axis=(0, 1)))

    gx = np.diff(gray, axis=1)
    f.append(float(np.var(gx)))

    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    h, w = spec.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    total = spec.sum() + 1e-8
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        f.append(float(spec[(r >= lo) & (r < hi)].sum() / total))

    col_d = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    on_grid = col_d[7::8].mean() if len(col_d) >= 8 else 0.0
    off_grid = np.delete(col_d, np.s_[7::8]).mean() if len(col_d) >= 9 else 1e-8
    f.append(float(on_grid / (off_grid + 1e-8)))
    return np.array(f, dtype=np.float32)


def extract(paths, labels, rng, desc):
    X, y = [], []
    for p, lab in tqdm(list(zip(paths, labels)), desc=desc, leave=False):
        try:
            with Image.open(p) as img:
                view = training_view(img.convert("RGB"), rng)
            X.append(features(view))
            y.append(lab)
        except Exception:
            continue
    return np.stack(X), np.array(y)


def linear_probe(X, y, n_classes, epochs=300, seed=0):
    """Standardize features, 70/30 split, linear softmax probe. Returns
    (test accuracy, per-class accuracy dict)."""
    g = np.random.default_rng(seed)
    idx = g.permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr = int(len(X) * 0.7)
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(y, dtype=torch.long)

    torch.manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xt[:n_tr]), yt[:n_tr])
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = lin(Xt[n_tr:]).argmax(1).numpy()
    truth = y[n_tr:]
    acc = float((pred == truth).mean())
    per_class = {}
    for c in range(n_classes):
        mask = truth == c
        if mask.any():
            per_class[c] = round(float((pred[mask] == c).mean()), 3)
    return acc, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    real = [p for p in (ROOT / "real").iterdir() if p.suffix.lower() in EXTS]
    ai = [p for p in (ROOT / "ai").iterdir() if p.suffix.lower() in EXTS]
    if not real or not ai:
        raise SystemExit("Need images in data/real and data/ai first.")

    print("=" * 64)
    print("  Probe 1: real-image SOURCE separability (low-level stats)")
    print("=" * 64)
    by_source = {}
    for p in real:
        by_source.setdefault(source_of(p.name), []).append(p)
    by_source = {s: v for s, v in by_source.items()
                 if len(v) >= 50 and s != "unknown"}
    if len(by_source) < 2:
        print("  Fewer than 2 real sources with >=50 images — skipping probe 1.")
    else:
        sources = sorted(by_source)
        paths, labels = [], []
        for i, s in enumerate(sources):
            sample = rng.sample(by_source[s], min(args.per_source, len(by_source[s])))
            paths += sample
            labels += [i] * len(sample)
        X, y = extract(paths, labels, rng, "  features")
        acc, per_class = linear_probe(X, y, len(sources), seed=args.seed)
        chance = max(Counter(y).values()) / len(y)
        print(f"  sources: {sources}")
        print(f"  macro accuracy {acc:.3f} vs chance {chance:.3f}")
        for i, s in enumerate(sources):
            print(f"    {s:<22} recall {per_class.get(i, float('nan'))}")
        if acc > chance * 1.5:
            print("  ⚠ WARN: sources are separable from low-level stats. The model can")
            print("    shortcut on source fingerprints. Equalize pipelines or rebalance")
            print("    before training (see CLAUDE.md Data Policy).")
        else:
            print("  ✓ PASS: sources not strongly separable from low-level stats.")

    print()
    print("=" * 64)
    print("  Probe 2: real-vs-AI separability from 12 stats alone")
    print("=" * 64)
    n = min(len(real), len(ai), args.per_source * 4)
    paths = rng.sample(real, n) + rng.sample(ai, n)
    labels = [0] * n + [1] * n
    X, y = extract(paths, labels, rng, "  features")
    acc, per_class = linear_probe(X, y, 2, seed=args.seed)
    print(f"  accuracy {acc:.3f} (chance 0.500)")
    print(f"    real recall {per_class.get(0)}  ai recall {per_class.get(1)}")
    if acc > 0.70:
        print("  ⚠ WARN: classes separable from trivial statistics — the dataset has a")
        print("    low-level tell (resolution/compression/brightness). Val metrics from")
        print("    training on this will be inflated. Fix the data first.")
    else:
        print("  ✓ PASS: no strong trivial separation.")


if __name__ == "__main__":
    main()
