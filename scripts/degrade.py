"""
Shared image degradation library — the SINGLE implementation used by both the
training loader (train.py) and evaluation (benchmark_baselines.py / eval
harness). Never reimplement these ops elsewhere: if train and eval degrade
differently, the robustness numbers stop meaning anything.

Two entry points:

  random_chain(img, rng)      — training-time augmentation: 1-3 randomly
                                sampled ops (JPEG/double-JPEG/WebP/rescale/
                                blur/noise/sharpen) in random order.
  apply_preset(img, name)     — DETERMINISTIC named platform pipelines
                                (twitter/instagram/whatsapp/discord_thumb/
                                screenshot) for the eval degradation ladder.

All functions take and return PIL RGB images. See docs/ROBUSTNESS.md §1.
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Primitive ops
# ---------------------------------------------------------------------------

def jpeg_cycle(img: Image.Image, quality: int, subsampling: int = 2) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def webp_cycle(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_max_dim(img: Image.Image, max_dim: int,
                   resample=Image.LANCZOS) -> Image.Image:
    if max(img.size) <= max_dim:
        return img
    scale = max_dim / max(img.size)
    return img.resize((max(1, round(img.width * scale)),
                       max(1, round(img.height * scale))), resample)


def down_up(img: Image.Image, factor: float, resample=Image.BICUBIC) -> Image.Image:
    """Lose resolution: downscale by `factor`, upscale back to original size.
    Simulates a thumbnail re-enlarged by a browser/CDN."""
    w, h = img.size
    small = img.resize((max(1, round(w * factor)), max(1, round(h * factor))), resample)
    return small.resize((w, h), resample)


def add_noise(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noise_rng = np.random.default_rng(rng.getrandbits(32))
    arr = arr + noise_rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Random chain ops (training augmentation)
# ---------------------------------------------------------------------------

def _op_jpeg(img, rng):
    return jpeg_cycle(img, rng.randint(40, 95), rng.choice([0, 1, 2]))

def _op_double_jpeg(img, rng):
    """Two JPEG passes with mismatched quality and an 8x8-grid misalignment —
    the signature of a re-shared social-media image."""
    img = jpeg_cycle(img, rng.randint(55, 95))
    dx, dy = rng.randint(0, 7), rng.randint(0, 7)
    if img.width > dx + 64 and img.height > dy + 64:
        img = img.crop((dx, dy, img.width, img.height))
    return jpeg_cycle(img, rng.randint(40, 90))

def _op_webp(img, rng):
    return webp_cycle(img, rng.randint(40, 90))

def _op_rescale(img, rng):
    kernel = rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
    return down_up(img, rng.uniform(0.4, 0.95), kernel)

def _op_blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.5)))

def _op_noise(img, rng):
    return add_noise(img, rng.uniform(2.0, 8.0), rng)

def _op_sharpen(img, rng):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=rng.randint(50, 150)))

# (op, sampling weight)
CHAIN_OPS = [
    (_op_jpeg,        0.28),
    (_op_double_jpeg, 0.15),
    (_op_webp,        0.10),
    (_op_rescale,     0.20),
    (_op_blur,        0.10),
    (_op_noise,       0.10),
    (_op_sharpen,     0.07),
]


def random_chain(img: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Apply 1-3 degradation ops in random order. The CALLER decides whether
    to degrade at all (e.g. with probability 0.7 per sample)."""
    rng = rng or random
    k = rng.choices([1, 2, 3], weights=[0.45, 0.35, 0.20])[0]
    ops = [op for op, _ in CHAIN_OPS]
    weights = [w for _, w in CHAIN_OPS]
    chosen = []
    pool_ops, pool_w = ops[:], weights[:]
    for _ in range(k):
        op = rng.choices(pool_ops, weights=pool_w)[0]
        i = pool_ops.index(op)
        pool_ops.pop(i)
        pool_w.pop(i)
        chosen.append(op)
    for op in chosen:
        img = op(img, rng)
    return img


# ---------------------------------------------------------------------------
# Deterministic platform presets (eval ladder). Approximations of real
# pipelines — keep stable across versions so metrics stay comparable.
# ---------------------------------------------------------------------------

def _preset_twitter(img):
    return jpeg_cycle(resize_max_dim(img, 2048), 85, subsampling=2)

def _preset_instagram(img):
    if img.width > 1080:
        scale = 1080 / img.width
        img = img.resize((1080, max(1, round(img.height * scale))), Image.LANCZOS)
    return jpeg_cycle(img, 80, subsampling=2)

def _preset_whatsapp(img):
    return jpeg_cycle(resize_max_dim(img, 1600), 72, subsampling=2)

def _preset_discord_thumb(img):
    return jpeg_cycle(resize_max_dim(img, 1024), 75, subsampling=2)

def _preset_screenshot(img):
    """Browser render at a viewport fraction, then re-shared as JPEG."""
    img = img.resize((max(1, round(img.width * 0.75)),
                      max(1, round(img.height * 0.75))), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.3))
    return jpeg_cycle(img, 92, subsampling=1)

PRESETS = {
    "twitter":       _preset_twitter,
    "instagram":     _preset_instagram,
    "whatsapp":      _preset_whatsapp,
    "discord_thumb": _preset_discord_thumb,
    "screenshot":    _preset_screenshot,
}


def apply_preset(img: Image.Image, name: str) -> Image.Image:
    if name in ("clean", "none"):
        return img
    return PRESETS[name](img.convert("RGB"))
