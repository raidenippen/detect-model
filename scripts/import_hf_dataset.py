"""
Generic Hugging Face dataset importer for the AI image class.

Streams any HF image dataset, optionally stratified by a generator/model
column, and saves images into data/ai (original bytes when possible, PNG
otherwise) — same no-re-encode policy as the collectors.

Built for (all permissively licensed, ungated, verified Jun 2026):

  # Midjourney v6 (real MJ pulls, Jun 2024, MIT):
  python import_hf_dataset.py --dataset terminusresearch/midjourney-v6-520k-raw \
      --prefix mj_v6 --limit 4000

  # gpt-image-1 outputs (CC-BY-4.0). 'generate' keys are full regenerations:
  python import_hf_dataset.py --dataset UCSC-VLAA/GPT-Image-Edit-1.5M \
      --prefix gptimg_ds --limit 2500 --key-filter generate

  # DRAGON: 25 diffusion models incl. Kolors/Kandinsky3/Cascade (CC-BY-SA-4.0):
  python import_hf_dataset.py --dataset lesc-unifi/dragon --config Medium \
      --prefix dragon --limit 5000 --stratify model.txt --per-key 200

Images land in data/ai. Source tracking comes from the filename prefix.
"""

import io
import re
import argparse
from pathlib import Path
from collections import Counter

from PIL import Image
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
if not os.environ.get("HF_TOKEN"):
    os.environ.pop("HF_TOKEN", None)  # empty token crashes huggingface_hub

OUT = Path(__file__).parent.parent / "data" / "ai"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_COLS    = ["png", "jpg", "jpeg", "webp", "image", "img", "image_data"]
STRATIFY_COLS = ["model.txt", "model_name", "model", "generator", "version"]


def slugify(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:50] or "x"


def save_value(value, name_base: str) -> bool:
    """Save an image value that may be a PIL Image, raw bytes, or a dict
    holding bytes. Original bytes kept when the format is already sane."""
    try:
        if isinstance(value, dict) and "bytes" in value:
            value = value["bytes"]
        if isinstance(value, (bytes, bytearray)):
            img = Image.open(io.BytesIO(value))
            ext = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(img.format)
            if ext:
                (OUT / f"{name_base}{ext}").write_bytes(value)
                return True
            value = img  # unusual container format -> fall through to PNG
        if isinstance(value, Image.Image):
            value.convert("RGB").save(OUT / f"{name_base}.png", "PNG")
            return True
    except Exception:
        return False
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--prefix", required=True,
                   help="Filename prefix, e.g. mj_v6 -> mj_v6_*.png")
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--image-col", default=None)
    p.add_argument("--stratify", default=None,
                   help="Column to cap per distinct value (e.g. model.txt)")
    p.add_argument("--per-key", type=int, default=None)
    p.add_argument("--key-filter", default=None,
                   help="Only rows whose __key__ contains this substring")
    args = p.parse_args()

    from datasets import load_dataset
    print(f"Streaming {args.dataset}" + (f" [{args.config}]" if args.config else ""))
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    cols = list(ds.features.keys()) if ds.features else None
    if cols:
        print(f"Columns: {cols}")
    image_col = args.image_col
    stratify = args.stratify

    saved, skipped = 0, 0
    per_key = Counter()
    resumed = sum(1 for _ in OUT.glob(f"{args.prefix}_*"))
    if resumed:
        print(f"Note: {resumed} files with this prefix already exist; adding more.")

    pbar = tqdm(desc=args.prefix, unit="img")
    for row in ds:
        if saved >= args.limit:
            break
        pbar.update(1)

        if image_col is None:  # introspect on the first row if needed
            keys = list(row.keys())
            image_col = next((c for c in IMAGE_COLS if c in keys), None)
            if image_col is None:
                image_col = next((k for k, v in row.items()
                                  if isinstance(v, Image.Image)), None)
            if image_col is None:
                raise SystemExit(f"No image column found. Row keys: {keys} — pass --image-col.")
            if stratify is None:
                stratify = next((c for c in STRATIFY_COLS if c in keys), None)
            print(f"Using image_col={image_col!r}, stratify={stratify!r}")

        if args.key_filter and args.key_filter not in str(row.get("__key__", "")):
            continue

        key = slugify(row.get(stratify, "")) if stratify else ""
        if stratify and args.per_key and per_key[key] >= args.per_key:
            continue

        name = f"{args.prefix}_{key + '_' if key else ''}{resumed + saved:06d}"
        if save_value(row[image_col], name):
            saved += 1
            if stratify:
                per_key[key] += 1
        else:
            skipped += 1

    pbar.close()
    print(f"\nSaved {saved} images (prefix {args.prefix}_), skipped {skipped}.")
    if per_key:
        print("Per key:", dict(per_key.most_common(12)))


if __name__ == "__main__":
    main()
