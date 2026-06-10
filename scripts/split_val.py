"""
Move ~10% of real and AI images into validation directories.

Unlike a plain random split, this groups near-duplicate images (perceptual-hash
clusters) and assigns whole clusters to one side. A random per-file split puts
near-identical images on both sides of the train/val boundary — burst shots,
same-prompt generations, COCO scene crops — and the resulting val metrics
measure memorization, not generalization.

Run once after collection is complete:
  python split_val.py
  python split_val.py --val-frac 0.10 --hamming 6
"""

import random
import shutil
import argparse
from pathlib import Path

ROOT     = Path(__file__).parent.parent / "data"
SPLITS   = [(ROOT / "real", ROOT / "val_real"), (ROOT / "ai", ROOT / "val_ai")]
EXTS     = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def list_images(d: Path):
    files = []
    for pat in EXTS:
        files.extend(d.glob(pat))
    return sorted(files)


def cluster_by_phash(paths, hamming_threshold: int):
    """Union near-duplicates (phash hamming distance <= threshold) into clusters.
    Returns a list of clusters (lists of paths)."""
    try:
        import imagehash
        import numpy as np
        from PIL import Image
    except ImportError:
        print("  WARNING: imagehash/numpy not installed — falling back to per-file split.")
        print("  pip install imagehash numpy   (strongly recommended)")
        return [[p] for p in paths]

    from tqdm import tqdm

    hashes, kept = [], []
    for p in tqdm(paths, desc="  phash", leave=False):
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img)  # 64-bit
            hashes.append(int(str(h), 16))
            kept.append(p)
        except Exception:
            kept.append(p)
            hashes.append(None)

    idx = [i for i, h in enumerate(hashes) if h is not None]
    arr = np.array([hashes[i] for i in idx], dtype=np.uint64)
    n = len(arr)

    # Union-find
    parent = list(range(len(kept)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Brute-force hamming in numpy chunks: 64-bit XOR then byte-LUT popcount.
    POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    CHUNK = 512
    for start in tqdm(range(0, n, CHUNK), desc="  near-dupes", leave=False):
        block = arr[start:start + CHUNK]
        xor = block[:, None] ^ arr[None, start:]  # upper triangle only
        dist = POP[xor.view(np.uint8)].reshape(xor.shape[0], xor.shape[1], 8).sum(axis=2)
        rows, cols = np.nonzero(dist <= hamming_threshold)
        for r, c in zip(rows, cols):
            i_global, j_global = start + int(r), start + int(c)
            if i_global < j_global:
                union(idx[i_global], idx[j_global])

    clusters = {}
    for i, p in enumerate(kept):
        clusters.setdefault(find(i), []).append(p)
    return list(clusters.values())


def split(src: Path, dest: Path, val_frac: float, hamming: int):
    dest.mkdir(exist_ok=True)
    images = list_images(src)
    if not images:
        print(f"  {src.name}: empty, skipping")
        return

    print(f"  {src.name}: clustering {len(images)} images...")
    clusters = cluster_by_phash(images, hamming)
    n_multi = sum(1 for c in clusters if len(c) > 1)
    print(f"  {src.name}: {len(clusters)} clusters ({n_multi} contain near-duplicates)")

    random.shuffle(clusters)
    target = int(len(images) * val_frac)
    moved = 0
    for cluster in clusters:
        if moved >= target:
            break
        for img in cluster:
            shutil.move(str(img), dest / img.name)
            moved += 1
    print(f"  {src.name}: moved {moved}/{len(images)} to {dest.name}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--hamming", type=int, default=6,
                        help="phash hamming distance treated as near-duplicate")
    args = parser.parse_args()

    print("Splitting validation set (near-duplicate-cluster aware)...")
    for src, dest in SPLITS:
        split(src, dest, args.val_frac, args.hamming)
    print("Done. The val set is FROZEN once training starts — never reshuffle it,")
    print("or metrics stop being comparable across model versions.")


if __name__ == "__main__":
    main()
