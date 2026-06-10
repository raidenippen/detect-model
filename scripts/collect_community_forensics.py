"""
Pull a generator-stratified subset of CommunityForensics-Small into data/ai
(and optionally its paired real images into data/real).

Why: the Community Forensics paper (Park & Owens, CVPR 2025, arXiv:2411.04125)
showed that the NUMBER OF DISTINCT GENERATORS in training data matters more
than images per generator. Their Small release (~300k images, redistributable
licenses) covers thousands of generators — this is the diversity backbone of
our AI class. Our own Flux/Replicate/OpenAI generations layer current-model
coverage on top.

Images are saved as ORIGINAL BYTES (mostly PNG) — same no-re-encode policy as
the other collectors. NSFW-flagged rows are skipped.

Verified schema (Jun 2026): image_name, format, resolution, mode,
image_data (binary), model_name, nsfw_flag, prompt, real_source, subset,
split, label (1=generated, 0=real), architecture.

NOTE: rows stream grouped by generator, and skipping a row still downloads its
bytes — a full pass over the dataset is bandwidth-heavy (tens of GB). Ctrl-C
at any point keeps everything saved so far; re-running resumes the caps from
the files already on disk.

Usage:
  python collect_community_forensics.py --limit 20000 --per-generator 40
  python collect_community_forensics.py --limit 20000 --include-real

Requires: pip install datasets   (and HF_TOKEN in .env if the dataset gates access)
"""

import os
import re
import argparse
from pathlib import Path
from collections import Counter

from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT     = Path(__file__).parent.parent / "data"
AI_DIR   = ROOT / "ai"
REAL_DIR = ROOT / "real"
AI_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "OwensLab/CommunityForensics-Small"
EXPECTED_COLS = {"image_data", "format", "model_name", "nsfw_flag", "label"}
EXT_FOR_FORMAT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:60] or "unknown"


def existing_caps() -> Counter:
    """Resume support: count already-saved images per generator slug."""
    caps = Counter()
    for p in AI_DIR.glob("cf_*"):
        caps[p.stem[3:].rsplit("_", 1)[0]] += 1
    return caps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20000,
                        help="Max AI images to save this run")
    parser.add_argument("--per-generator", type=int, default=40,
                        help="Cap per generator — keeps the subset generator-diverse")
    parser.add_argument("--include-real", action="store_true",
                        help="Also save the dataset's real images to data/real")
    parser.add_argument("--real-limit", type=int, default=5000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets not installed. Run: pip install datasets")

    token = os.environ.get("HF_TOKEN")
    print(f"Streaming {DATASET} (split={args.split})...")
    ds = load_dataset(DATASET, split=args.split, streaming=True, token=token)

    missing = EXPECTED_COLS - set(ds.features)
    if missing:
        raise SystemExit(f"Dataset schema changed — missing columns {missing}. "
                         f"Got: {list(ds.features)}. Update this script.")

    per_gen = existing_caps()
    if per_gen:
        print(f"Resuming — {sum(per_gen.values())} images from {len(per_gen)} generators already on disk.")

    saved_ai = saved_real = skipped = 0
    pbar = tqdm(desc="CommunityForensics", unit="img")

    try:
        for row in ds:
            if saved_ai >= args.limit and (not args.include_real or saved_real >= args.real_limit):
                break
            pbar.update(1)
            if row["nsfw_flag"]:
                continue
            data = row["image_data"]
            ext = EXT_FOR_FORMAT.get(row["format"])
            if not data or ext is None:
                skipped += 1
                continue

            if row["label"] == 0:  # real
                if not args.include_real or saved_real >= args.real_limit:
                    continue
                out = REAL_DIR / f"cfreal_{slugify(row.get('real_source', ''))}_{saved_real:06d}{ext}"
                out.write_bytes(data)
                saved_real += 1
                continue

            if saved_ai >= args.limit:
                continue
            gen = slugify(row["model_name"])
            if per_gen[gen] >= args.per_generator:
                continue
            out = AI_DIR / f"cf_{gen}_{per_gen[gen]:04d}{ext}"
            out.write_bytes(data)
            per_gen[gen] += 1
            saved_ai += 1
    except KeyboardInterrupt:
        print("\nStopped early — everything saved so far is kept; re-running resumes.")

    pbar.close()
    print(f"\nSaved {saved_ai} AI images this run across {len(per_gen)} generators total "
          f"(cap {args.per_generator}/generator), {saved_real} real images, {skipped} skipped.")
    if per_gen:
        top = ", ".join(f"{g}:{n}" for g, n in per_gen.most_common(10))
        print(f"Top generators: {top}")


if __name__ == "__main__":
    main()
