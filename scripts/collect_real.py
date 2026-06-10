"""
Collect real images for AI detection model training.

IMPORTANT: images are saved as ORIGINAL BYTES — no resizing, no re-encoding.
Resizing/JPEG happens at training time as augmentation, applied identically to
both classes. Re-encoding at collection time bakes label-correlated compression
artifacts into the dataset (the old 384x384 squash pipeline did exactly that —
that data now lives in data/real_legacy_384 and should not be trained on).

Sources (hard real negatives first — polished professional photography is the
failure mode this project exists to fix):
  1. Unsplash Lite dataset — 25k curated professional photos, no API key needed
  2. Pexels API — professional portraits, bokeh, product, concert (free key)
  3. Wikimedia Commons — Quality/Featured images, RECURSIVE category traversal
  4. Open Images (FiftyOne) — consumer-photo diversity slice
  5. COCO — consumer-photo diversity slice

Usage:
  python collect_real.py --source unsplash  --limit 8000
  python collect_real.py --source pexels    --limit 5000
  python collect_real.py --source wikimedia --limit 5000
  python collect_real.py --source coco      --limit 3000
  python collect_real.py --source open_images --limit 3000

Environment variables (.env):
  PEXELS_API_KEY   — free at https://www.pexels.com/api/
"""

import os
import io
import csv
import sys
import time
import random
import hashlib
import zipfile
import argparse
import shutil
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "detect-model/1.0 (ML training dataset; https://github.com/raidenippen/detect-model)"}

MIN_SIDE  = 384          # reject images smaller than this on the short side
MAX_BYTES = 25_000_000   # reject pathological downloads

EXT_FOR_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def save_original(data: bytes, prefix: str) -> bool:
    """Validate and save original bytes untouched. Returns True if saved."""
    if not data or len(data) > MAX_BYTES:
        return False
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))  # verify() invalidates the object
        if min(img.size) < MIN_SIDE:
            return False
        ext = EXT_FOR_FORMAT.get(img.format)
        if ext is None:
            return False
    except Exception:
        return False
    name = f"{prefix}_{hashlib.md5(data).hexdigest()}{ext}"
    out = OUTPUT_DIR / name
    if out.exists():
        return False  # exact duplicate
    out.write_bytes(data)
    return True


def fetch_and_save(url: str, prefix: str, timeout: int = 30) -> bool:
    try:
        resp = requests.get(url, timeout=timeout, headers=UA)
        resp.raise_for_status()
        return save_original(resp.content, prefix)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Source 1: Unsplash Lite dataset — 25k curated professional photos.
# This is the hard-negative core: pro portraits, bokeh, product, editorial.
# ---------------------------------------------------------------------------
UNSPLASH_LITE_URL = "https://unsplash.com/data/lite/latest"

def collect_unsplash(limit: int):
    print(f"\n[Unsplash Lite] Collecting up to {limit} images...")
    zip_path = Path("/tmp/unsplash_lite.zip")

    if not zip_path.exists():
        print("  Downloading Unsplash Lite dataset (~700MB, one-time)...")
        try:
            with requests.get(UNSPLASH_LITE_URL, stream=True, timeout=60, headers=UA) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except Exception as e:
            print(f"  Failed to download Unsplash Lite: {e}")
            print("  Manual fallback: download from https://unsplash.com/data and place at /tmp/unsplash_lite.zip")
            return 0

    urls = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            tsv_names = [n for n in z.namelist() if n.startswith("photos.tsv")]
            for tsv_name in tsv_names:
                with z.open(tsv_name) as f:
                    reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                    for row in reader:
                        u = row.get("photo_image_url")
                        if u:
                            urls.append(u)
    except Exception as e:
        print(f"  Failed to parse Unsplash Lite zip: {e}")
        return 0

    random.shuffle(urls)
    urls = urls[: int(limit * 1.2)]  # headroom for failures
    saved = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        # w=1920 serves a high-quality render; originals can be 50MP+
        futures = [ex.submit(fetch_and_save, f"{u}?w=1920&fm=jpg&q=92", "unsplash") for u in urls]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Unsplash"):
            if fut.result():
                saved += 1
                if saved >= limit:
                    break
    print(f"[Unsplash Lite] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 2: Pexels API — professional stock photography (hard negatives)
# ---------------------------------------------------------------------------
PEXELS_QUERIES = [
    "concert stage performance", "live music crowd", "portrait bokeh",
    "studio portrait", "wedding photography", "product photography",
    "food photography", "fashion editorial", "sports action",
    "street photography night", "golden hour portrait", "macro flower",
    "wildlife telephoto", "architecture dramatic sky", "dj festival lights",
]

def collect_pexels(limit: int):
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        print("\n[Pexels] Skipping — PEXELS_API_KEY not set in .env (free at https://www.pexels.com/api/)")
        return 0

    print(f"\n[Pexels] Collecting up to {limit} images...")
    headers = {**UA, "Authorization": api_key}
    per_query = max(1, limit // len(PEXELS_QUERIES))
    saved = 0

    for query in PEXELS_QUERIES:
        if saved >= limit:
            break
        page, got = 1, 0
        while got < per_query and saved < limit:
            try:
                resp = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers=headers,
                    params={"query": query, "per_page": 80, "page": page},
                    timeout=20,
                )
                resp.raise_for_status()
                photos = resp.json().get("photos", [])
                if not photos:
                    break
                for p in tqdm(photos, desc=f"  {query[:30]}", leave=False):
                    if got >= per_query or saved >= limit:
                        break
                    url = p.get("src", {}).get("large2x") or p.get("src", {}).get("original")
                    if url and fetch_and_save(url, "pexels"):
                        saved += 1
                        got += 1
                page += 1
                time.sleep(0.5)  # 200 req/hr free tier
            except Exception as e:
                print(f"  [Pexels] {query} failed: {e}")
                break

    print(f"[Pexels] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 3: Wikimedia Commons — recursive category traversal.
# The old version queried cmtype=file on top-level categories, which contain
# almost only subcategories — it collected 6 images total. This version walks
# subcategories breadth-first and also paginates flat curated categories
# (Quality images / Featured pictures), which are juried pro photography.
# ---------------------------------------------------------------------------
WIKI_API = "https://commons.wikimedia.org/w/api.php"

WIKIMEDIA_ROOTS = [
    ("Quality images", 0),                            # flat, ~300k juried files
    ("Featured pictures on Wikimedia Commons", 1),    # curated, shallow
    ("Portrait photographs", 3),
    ("Concert photography", 3),
    ("Wedding photography", 3),
    ("Sports photography", 3),
    ("Food photography", 3),
]

def _wiki_query(params):
    params = {"format": "json", **params}
    resp = requests.get(WIKI_API, params=params, headers=UA, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _wiki_fetch(url: str) -> bool:
    """Wikimedia throttles parallel downloads hard (HTTP 429) — fetch
    sequentially and honor Retry-After with backoff."""
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30, headers=UA)
            if resp.status_code == 429:
                wait = resp.headers.get("Retry-After")
                time.sleep(min(int(wait) if wait and wait.isdigit() else 5 * (attempt + 1), 60))
                continue
            resp.raise_for_status()
            return save_original(resp.content, "wiki")
        except Exception:
            time.sleep(2)
    return False

def _wiki_file_urls(titles):
    """Batched imageinfo lookup (50 titles/request). Returns list of URLs,
    using a 2048px thumb render when the original is larger than that."""
    urls = []
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        try:
            data = _wiki_query({
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": 2048,
            })
            for page in data.get("query", {}).get("pages", {}).values():
                ii = page.get("imageinfo", [{}])[0]
                if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                    continue
                url = ii.get("thumburl") if ii.get("width", 0) > 2048 else ii.get("url")
                if url:
                    urls.append(url)
        except Exception:
            continue
        time.sleep(0.1)
    return urls

def collect_wikimedia(limit: int):
    print(f"\n[Wikimedia] Collecting up to {limit} images (recursive)...")
    saved = 0
    per_root = max(1, limit // len(WIKIMEDIA_ROOTS))

    for root, max_depth in WIKIMEDIA_ROOTS:
        if saved >= limit:
            break
        root_target = min(per_root, limit - saved)
        titles, seen_cats = [], set()
        queue = deque([(f"Category:{root}", 0)])

        while queue and len(titles) < root_target * 2:
            cat, depth = queue.popleft()
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            cont = {}
            while len(titles) < root_target * 2:
                try:
                    data = _wiki_query({
                        "action": "query",
                        "list": "categorymembers",
                        "cmtitle": cat,
                        "cmtype": "file|subcat",
                        "cmlimit": 500,
                        **cont,
                    })
                except Exception:
                    break
                for m in data.get("query", {}).get("categorymembers", []):
                    if m["title"].startswith("Category:"):
                        if depth < max_depth:
                            queue.append((m["title"], depth + 1))
                    else:
                        titles.append(m["title"])
                cont = data.get("continue", {})
                if not cont:
                    break
                time.sleep(0.1)

        random.shuffle(titles)
        urls = _wiki_file_urls(titles[: int(root_target * 1.5)])
        got = 0
        for u in tqdm(urls, desc=f"  {root[:35]}"):
            if _wiki_fetch(u):
                saved += 1
                got += 1
                if got >= root_target:
                    break
            time.sleep(0.3)  # stay under Wikimedia's per-IP rate limit
        print(f"  [{root}] saved {got}")

    print(f"[Wikimedia] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 4: Open Images via FiftyOne (diversity slice — copy originals)
# ---------------------------------------------------------------------------
OPEN_IMAGES_LABELS = [
    "Person", "Musical instrument", "Guitar", "Microphone",
    "Flower", "Dog", "Cat", "Food", "Car", "Skyscraper",
]

def collect_open_images(limit: int):
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("fiftyone not installed. Run: pip install fiftyone")
        return 0

    print(f"\n[Open Images] Collecting up to {limit} images...")
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
                    if save_original(Path(sample.filepath).read_bytes(),
                                     f"oi_{label.lower().replace(' ', '_')}"):
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
# Source 5: COCO (diversity slice — original bytes from coco_url)
# ---------------------------------------------------------------------------
COCO_CATEGORIES = [
    "person", "dog", "cat", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "boat", "bird", "horse",
    "elephant", "giraffe", "zebra", "pizza", "cake",
]

def collect_coco(limit: int):
    print(f"\n[COCO] Collecting up to {limit} images...")
    ann_path = Path("/tmp/annotations/instances_val2017.json")

    if not ann_path.exists():
        print("  Downloading COCO annotations...")
        try:
            resp = requests.get(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                stream=True, timeout=120,
            )
            zip_path = Path("/tmp/coco_ann.zip")
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extract("annotations/instances_val2017.json", "/tmp/")
        except Exception as e:
            print(f"  [COCO] Failed to download annotations: {e}")
            return 0

    import json
    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in COCO_CATEGORIES}
    keep_ids = {a["image_id"] for a in coco["annotations"] if a["category_id"] in cat_map}
    images = [img for img in coco["images"] if img["id"] in keep_ids]
    random.shuffle(images)
    images = images[:limit]

    saved = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_and_save, img["coco_url"], "coco") for img in images]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  COCO"):
            if fut.result():
                saved += 1
    print(f"[COCO] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SOURCES = {
    "unsplash":    collect_unsplash,
    "pexels":      collect_pexels,
    "wikimedia":   collect_wikimedia,
    "open_images": collect_open_images,
    "coco":        collect_coco,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all", choices=["all", *SOURCES])
    parser.add_argument("--limit", type=int, default=5000,
                        help="Images to collect (per the chosen source; split across sources when --source=all)")
    args = parser.parse_args()

    existing = sum(1 for _ in OUTPUT_DIR.iterdir())
    print(f"Output: {OUTPUT_DIR}")
    print(f"Already collected: {existing} images")

    total = 0
    if args.source == "all":
        # Weight toward hard negatives: pro photography > consumer diversity
        weights = {"unsplash": 0.35, "pexels": 0.20, "wikimedia": 0.20,
                   "open_images": 0.125, "coco": 0.125}
        for name, fn in SOURCES.items():
            total += fn(int(args.limit * weights[name]))
    else:
        total += SOURCES[args.source](args.limit)

    final = sum(1 for _ in OUTPUT_DIR.iterdir())
    print(f"\nDone. Collected {total} new images. Total in directory: {final}")


if __name__ == "__main__":
    main()
