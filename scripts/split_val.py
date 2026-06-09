"""
Move 10% of real and AI images into validation directories.
Run once after collection is complete.

Usage:
  python split_val.py
"""

import random
import shutil
from pathlib import Path

ROOT     = Path(__file__).parent.parent / "data"
REAL_DIR = ROOT / "real"
AI_DIR   = ROOT / "ai"
VAL_REAL = ROOT / "val_real"
VAL_AI   = ROOT / "val_ai"

VAL_REAL.mkdir(exist_ok=True)
VAL_AI.mkdir(exist_ok=True)

VAL_SPLIT = 0.10

def split(src: Path, dest: Path):
    images = list(src.glob("*.jpg"))
    random.shuffle(images)
    n_val  = max(1, int(len(images) * VAL_SPLIT))
    for img in images[:n_val]:
        shutil.move(str(img), dest / img.name)
    print(f"  {src.name}: moved {n_val}/{len(images)} to {dest.name}/")

print("Splitting validation set...")
split(REAL_DIR, VAL_REAL)
split(AI_DIR,   VAL_AI)
print("Done.")
