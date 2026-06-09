"""
Collect real images from four sources for AI detection model training.

Sources:
  1. Open Images Dataset (Google) — via FiftyOne, filtered by category
  2. COCO Dataset — standard ML benchmark, diverse scenes
  3. Wikimedia Commons — CC-licensed, high quality
  4. Flickr Creative Commons — CC-BY/CC0 only, diverse photography

All images are resized to 384x384, saved as JPEG quality 90, labeled as real.
Run each source independently or all at once.

Usage:
  python collect_real.py --source all --limit 5000
  python collect_real.py --source open_images --limit 2000
  python collect_real.py --source coco --limit 2000
  python collect_real.py --source wikimedia --limit 500
  python collect_real.py --source flickr --limit 500

Requirements:
  pip install fiftyone Pillow requests tqdm flickrapi

Environment variables (copy .env.example to .env and fill in):
  FLICKR_API_KEY
  FLICKR_API_SECRET
"""

import os
import io
import sys
import time
import random
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE  = 384
JPEG_QUALITY = 90

# Hard-real-negative categories — these are the cases current models fail on.
# Prioritise professionally shot photography that looks "too clean" or "too perfect".
OPEN_IMAGES_LABELS = [
    "Person", "Musical instrument", "Concert", "Guitar", "Microphone",
    "Flower", "Dog", "Cat", "Food", "Car", "Skyscraper",
    "Portrait", "Wedding", "Sports", "Dance",
]

COCO_CATEGORIES = [
    "person", "dog", "cat", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "boat", "bird", "horse",
    "elephant", "giraffe", "zebra", "pizza", "cake",
]

WIKIMEDIA_CATEGORIES = [
    "Portrait photography",
    "Concert photography",
    "Wedding photography",
    "Street photography",
    "Landscape photography",
    "Wildlife photography",
    "Sports photography",
    "Food photography",
]

FLICKR_TAGS = [
    "portrait", "concert", "wedding", "street photography",
    "landscape", "wildlife", "sports", "food photography",
    "bokeh", "golden hour", "documentary", "photojournalism",
]


def save_image(img: Image.Image, dest_dir: Path, name: str) -> bool:
    try:
        img = img.convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        out = dest_dir / f"{name}.jpg"
        img.save(out, "JPEG", quality=JPEG_QUALITY)
        return True
    except Exception:
        return False


def image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def fetch_and_save(url: str, dest_dir: Path, prefix: str) -> bool:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "detect-model/1.0 (ML training dataset)"})
        resp.raise_for_status()
        data = resp.content
        img  = Image.open(io.BytesIO(data))
        name = f"{prefix}_{image_hash(data)}"
        return save_image(img, dest_dir, name)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Source 1: Open Images Dataset via FiftyOne
# ---------------------------------------------------------------------------
def collect_open_images(limit: int, dest_dir: Path):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("fiftyone not installed. Run: pip install fiftyone")
        return 0

    print(f"\n[Open Images] Downloading up to {limit} images across {len(OPEN_IMAGES_LABELS)} categories...")
    per_label = max(1, limit // len(OPEN_IMAGES_LABELS))
    saved = 0

    for label in OPEN_IMAGES_LABELS:
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="validation",
                label_types=["classifications"],
                classes=[label],
                max_samples=per_label,
                shuffle=True,
            )
            for sample in tqdm(dataset, desc=f"  {label}", leave=False):
                if saved >= limit:
                    break
                try:
                    img = Image.open(sample.filepath)
                    name = f"oi_{label.lower().replace(' ', '_')}_{image_hash(open(sample.filepath, 'rb').read())}"
                    if save_image(img, dest_dir, name):
                        saved += 1
                except Exception:
                    continue
            fo.delete_dataset(dataset.name)
        except Exception as e:
            print(f"  [Open Images] {label} failed: {e}")
            continue

    print(f"[Open Images] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 2: COCO Dataset
# ---------------------------------------------------------------------------
def collect_coco(limit: int, dest_dir: Path):
    print(f"\n[COCO] Downloading up to {limit} images...")

    # COCO 2017 validation annotations
    ann_url  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_file = Path("/tmp/coco_annotations.json")

    if not ann_file.exists():
        print("  Downloading COCO annotations...")
        try:
            import zipfile
            resp = requests.get(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                stream=True, timeout=60
            )
            zip_path = Path("/tmp/coco_ann.zip")
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extract("annotations/instances_val2017.json", "/tmp/")
            ann_file = Path("/tmp/annotations/instances_val2017.json")
        except Exception as e:
            print(f"  [COCO] Failed to download annotations: {e}")
            return 0

    import json
    ann_path = Path("/tmp/annotations/instances_val2017.json")
    if not ann_path.exists():
        print("  [COCO] Annotations not found.")
        return 0

    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in COCO_CATEGORIES}
    ann_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_map:
            ann_by_img.setdefault(ann["image_id"], []).append(cat_map[ann["category_id"]])

    images = [img for img in coco["images"] if img["id"] in ann_by_img]
    random.shuffle(images)
    images = images[:limit]

    saved = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(
                fetch_and_save,
                img["coco_url"],
                dest_dir,
                f"coco_{img['id']}"
            ): img for img in images
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="  COCO"):
            if future.result():
                saved += 1

    print(f"[COCO] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 3: Wikimedia Commons
# ---------------------------------------------------------------------------
def collect_wikimedia(limit: int, dest_dir: Path):
    print(f"\n[Wikimedia] Downloading up to {limit} images...")

    BASE   = "https://commons.wikimedia.org/w/api.php"
    saved  = 0
    per_cat = max(1, limit // len(WIKIMEDIA_CATEGORIES))

    for category in WIKIMEDIA_CATEGORIES:
        if saved >= limit:
            break
        params = {
            "action":       "query",
            "list":         "categorymembers",
            "cmtitle":      f"Category:{category}",
            "cmtype":       "file",
            "cmlimit":      per_cat,
            "format":       "json",
        }
        try:
            resp = requests.get(BASE, params=params, timeout=15)
            members = resp.json().get("query", {}).get("categorymembers", [])
            titles  = [m["title"] for m in members]

            for title in tqdm(titles, desc=f"  {category}", leave=False):
                if saved >= limit:
                    break
                try:
                    info_resp = requests.get(BASE, params={
                        "action":    "query",
                        "titles":    title,
                        "prop":      "imageinfo",
                        "iiprop":    "url|mime",
                        "format":    "json",
                    }, timeout=15)
                    pages = info_resp.json()["query"]["pages"]
                    for page in pages.values():
                        ii   = page.get("imageinfo", [{}])[0]
                        mime = ii.get("mime", "")
                        url  = ii.get("url", "")
                        if mime not in ("image/jpeg", "image/png", "image/webp"):
                            continue
                        name = f"wiki_{hashlib.md5(url.encode()).hexdigest()}"
                        if fetch_and_save(url, dest_dir, name):
                            saved += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"  [Wikimedia] {category} failed: {e}")
            continue

    print(f"[Wikimedia] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 4: Flickr Creative Commons (CC-BY and CC0 only)
# ---------------------------------------------------------------------------
def collect_flickr(limit: int, dest_dir: Path):
    api_key    = os.environ.get("FLICKR_API_KEY")
    api_secret = os.environ.get("FLICKR_API_SECRET")

    if not api_key or not api_secret:
        print("\n[Flickr] Skipping — FLICKR_API_KEY / FLICKR_API_SECRET not set in .env")
        return 0

    try:
        import flickrapi
    except ImportError:
        print("[Flickr] flickrapi not installed. Run: pip install flickrapi")
        return 0

    print(f"\n[Flickr] Downloading up to {limit} images...")
    flickr  = flickrapi.FlickrAPI(api_key, api_secret, format="parsed-json")
    saved   = 0
    per_tag = max(1, limit // len(FLICKR_TAGS))

    # CC licenses: 1=CC-BY-NC-SA, 2=CC-BY-NC, 4=CC-BY, 9=CC0 (public domain)
    # Only use 4 (CC-BY) and 9 (CC0) — safe for ML training
    SAFE_LICENSES = "4,9"

    for tag in FLICKR_TAGS:
        if saved >= limit:
            break
        try:
            resp   = flickr.photos.search(
                tags=tag,
                license=SAFE_LICENSES,
                media="photos",
                content_type=1,        # photos only, no screenshots
                sort="interestingness-desc",
                extras="url_l,url_o",  # large and original URL
                per_page=min(per_tag, 100),
                page=1,
            )
            photos = resp.get("photos", {}).get("photo", [])
            for photo in tqdm(photos, desc=f"  {tag}", leave=False):
                if saved >= limit:
                    break
                url = photo.get("url_o") or photo.get("url_l")
                if not url:
                    continue
                name = f"flickr_{photo['id']}"
                if fetch_and_save(url, dest_dir, name):
                    saved += 1
        except Exception as e:
            print(f"  [Flickr] {tag} failed: {e}")
            continue

    print(f"[Flickr] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["all", "open_images", "coco", "wikimedia", "flickr"])
    parser.add_argument("--limit",  type=int, default=5000,
                        help="Total images to collect (split across sources when --source=all)")
    args = parser.parse_args()

    existing = len(list(OUTPUT_DIR.glob("*.jpg")))
    print(f"Output: {OUTPUT_DIR}")
    print(f"Already collected: {existing} images")

    total = 0

    if args.source == "all":
        per_source = args.limit // 4
        total += collect_open_images(per_source, OUTPUT_DIR)
        total += collect_coco(per_source, OUTPUT_DIR)
        total += collect_wikimedia(per_source // 4, OUTPUT_DIR)  # smaller — slower API
        total += collect_flickr(per_source // 4, OUTPUT_DIR)
    elif args.source == "open_images":
        total += collect_open_images(args.limit, OUTPUT_DIR)
    elif args.source == "coco":
        total += collect_coco(args.limit, OUTPUT_DIR)
    elif args.source == "wikimedia":
        total += collect_wikimedia(args.limit, OUTPUT_DIR)
    elif args.source == "flickr":
        total += collect_flickr(args.limit, OUTPUT_DIR)

    final = len(list(OUTPUT_DIR.glob("*.jpg")))
    print(f"\nDone. Collected {total} new images. Total in directory: {final}")


if __name__ == "__main__":
    main()
