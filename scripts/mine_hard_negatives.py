"""
Mine hard-negative CANDIDATES from our own real-image pool.

A hard negative is a real photo that detectors score as AI. We have ~23k
verified-real images and the failing baseline detectors — so instead of
hunting for hard negatives manually, score the real pool with the baselines
and surface the images they are most wrong about.

Output: data/hard_negative_candidates/
  - top-K candidate images, copied (NOT moved), named with their mean AI score
  - candidates.csv (path, per-model scores)
  - contact_sheet_*.png grids for quick human review

Workflow after running:
  1. Review the contact sheets / candidate folder.
  2. For every image you confirm is plausibly-real-but-flagged, MOVE it to
     data/hard_negatives/ and DELETE the original from data/real/ (an eval
     image must never also be a training image). Anything that looks like it
     might actually be AI that slipped into a stock site: just delete it from
     data/real entirely — it's a label error either way.

Usage:
  python mine_hard_negatives.py --sample 4000 --top 150
"""

import csv
import random
import shutil
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent / "data"
REAL = ROOT / "real"
OUT  = ROOT / "hard_negative_candidates"
EXTS = (".jpg", ".jpeg", ".png", ".webp")

MINERS = [
    "NYUAD-ComNets/NYUAD_AI-generated_images_detector",
    "Ateeqq/ai-vs-human-image-detector",
]

AI_WORDS   = ("ai", "artificial", "fake", "generated", "deepfake", "synthetic",
              "dalle", "stable", "diffusion", "midjourney")
REAL_WORDS = ("real", "human", "authentic", "nature", "photo")


def ai_score(predictions) -> float:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=["real", "ai"], default="real",
                        help="real: surface real images scored most-AI (hard negatives). "
                             "ai: surface AI images scored most-real (hard positives / label errors).")
    parser.add_argument("--sample", type=int, default=4000)
    parser.add_argument("--top", type=int, default=150)
    parser.add_argument("--models", nargs="*", default=MINERS)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    global OUT
    pool_dir = ROOT / args.pool
    if args.pool == "ai":
        OUT = ROOT / "hard_positive_candidates"
    OUT.mkdir(exist_ok=True)
    paths = [p for p in pool_dir.iterdir() if p.suffix.lower() in EXTS]
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.sample]
    print(f"Scoring {len(paths)} {args.pool} images with {len(args.models)} models...")

    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)

    scores = {p: [] for p in paths}
    for model_id in args.models:
        try:
            clf = pipeline("image-classification", model=model_id, device=device)
        except Exception as e:
            print(f"  SKIP {model_id}: {e}")
            continue
        for p in tqdm(paths, desc=model_id.split("/")[-1][:30]):
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    img.thumbnail((1024, 1024), Image.BICUBIC)
                    scores[p].append(ai_score(clf(img)))
            except Exception:
                scores[p].append(float("nan"))
        del clf

    # real pool: rank by HIGHEST ai score (most wrongly flagged).
    # ai pool: rank by LOWEST ai score (most wrongly passed).
    sign = -1 if args.pool == "real" else 1
    ranked = sorted(
        ((p, s) for p, s in scores.items() if s and not any(x != x for x in s)),
        key=lambda kv: sign * (sum(kv[1]) / len(kv[1])),
    )[: args.top]

    with open(OUT / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "mean_ai_score", *[m.split("/")[-1] for m in args.models]])
        for p, s in ranked:
            mean = sum(s) / len(s)
            w.writerow([p.name, f"{mean:.4f}", *[f"{x:.4f}" for x in s]])
            shutil.copy2(p, OUT / f"{mean:.3f}_{p.name}")

    # contact sheets for fast review
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        per_sheet = 49
        for sheet_i in range(0, len(ranked), per_sheet):
            chunk = ranked[sheet_i: sheet_i + per_sheet]
            fig, axes = plt.subplots(7, 7, figsize=(21, 21))
            for ax in axes.flatten():
                ax.axis("off")
            for ax, (p, s) in zip(axes.flatten(), chunk):
                try:
                    with Image.open(p) as img:
                        img = img.convert("RGB")
                        img.thumbnail((300, 300))
                        ax.imshow(img)
                    ax.set_title(f"{sum(s)/len(s):.2f} {p.name[:18]}", fontsize=7)
                except Exception:
                    continue
            out = OUT / f"contact_sheet_{sheet_i // per_sheet + 1}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=70)
            plt.close()
            print(f"  wrote {out}")
    except ImportError:
        pass

    print(f"\n{len(ranked)} candidates in {OUT}/ (sorted by mean AI score; "
          f"filenames are prefixed with the score).")
    print("Review them, then for confirmed hard negatives:")
    print("  mv data/hard_negative_candidates/<score>_<name> data/hard_negatives/<name>")
    print("  rm data/real/<name>        # eval images must leave the training pool")


if __name__ == "__main__":
    main()
