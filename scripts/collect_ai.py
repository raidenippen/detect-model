"""
Generate AI images for training data using local Flux (free) and optionally
Replicate API or OpenAI DALL-E 3 for generator diversity.

Generators:
  1. Flux (local, free) — runs on Apple Silicon MPS, ~30-60s/image
  2. Replicate API — Flux/SDXL via cloud GPU, ~$0.003/image
  3. OpenAI DALL-E 3 — optional, ~$0.04/image, for DALL-E-specific artifacts

Prompts are designed to cover the hard cases — portraits, concerts, professional
photography — so the model learns to distinguish these from real photos.

Usage:
  python collect_ai.py --source local --count 500
  python collect_ai.py --source replicate --count 1000
  python collect_ai.py --source dalle --count 200
  python collect_ai.py --source all --count 5000

Requirements:
  pip install diffusers transformers accelerate torch Pillow tqdm requests
"""

import os
import io
import sys
import time
import random
import hashlib
import argparse
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "ai"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE  = 384
JPEG_QUALITY = 90

# Prompts cover: portraits, concerts, professional photography, landscapes,
# product shots — the cases where current models produce the most false positives
# on real images. We want the model to learn the subtle differences.
PROMPTS = [
    # Portraits
    "professional portrait photo of a young woman, studio lighting, bokeh background, sharp focus",
    "headshot of a businessman, natural window light, shallow depth of field",
    "candid portrait of elderly man, street photography, Leica camera aesthetic",
    "portrait of a woman with curly hair, golden hour light, warm tones",
    "close up portrait of a child laughing, outdoor natural light, Canon 85mm",

    # Concerts and live events
    "concert photo of a rock band on stage, dramatic lighting, crowd in background",
    "female singer performing on stage with guitar, spotlight, bokeh crowd",
    "jazz musician playing trumpet, moody club lighting, shallow depth of field",
    "DJ at festival, colorful stage lights, wide angle shot",
    "orchestra performance, classical concert hall, dramatic lighting",

    # Professional photography styles
    "wedding photo of couple in golden hour light, romantic, editorial style",
    "fashion editorial photo of model in designer clothing, studio lighting",
    "sports photography, basketball player dunking, action shot, blurred crowd",
    "food photography, gourmet dish on wooden table, top down shot, natural light",
    "product photography, luxury watch on marble surface, soft studio lighting",

    # Landscapes and architecture
    "landscape photo of mountain range at sunset, dramatic clouds, golden hour",
    "aerial photography of coastline, blue ocean, natural colors",
    "street photography in Tokyo at night, neon lights, rain reflections",
    "architecture photo of modern glass building, dramatic sky, wide angle",
    "forest path in autumn, dappled light, photorealistic nature photography",

    # Wildlife and nature
    "wildlife photography of lion in savanna, telephoto lens, natural light",
    "macro photography of flower with dew drops, shallow depth of field",
    "bird in flight, wildlife photography, motion blur background",
    "underwater photography of coral reef, vivid colors, natural light",
    "dog portrait in outdoor park, shallow depth of field, happy expression",

    # Documentary and photojournalism style
    "documentary photo of busy market, candid street photography, natural light",
    "photojournalism style image of city street, black and white, high contrast",
    "travel photography in Indian village, colorful clothing, natural light",
    "environmental portrait of craftsman in workshop, dramatic side lighting",
    "reportage style photo of children playing, candid, natural light",
]

# Style suffixes to add diversity within each prompt
STYLE_SUFFIXES = [
    ", photorealistic, 8k, professional photography",
    ", shot on Canon EOS R5, natural light",
    ", Nikon Z9, 50mm lens, award winning photography",
    ", documentary photography style",
    ", National Geographic style photography",
    ", editorial photography",
    ", fine art photography",
    "",
]


def save_image(img: Image.Image, name: str) -> bool:
    try:
        img = img.convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        out = OUTPUT_DIR / f"{name}.jpg"
        img.save(out, "JPEG", quality=JPEG_QUALITY)
        return True
    except Exception as e:
        print(f"  save failed: {e}")
        return False


def random_prompt() -> str:
    base   = random.choice(PROMPTS)
    suffix = random.choice(STYLE_SUFFIXES)
    return base + suffix


# ---------------------------------------------------------------------------
# Source 1: Local Flux — auto-detects CUDA (Nvidia) or MPS (Apple Silicon)
# ---------------------------------------------------------------------------
def collect_local(count: int):
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError:
        print("diffusers not installed. Run: pip install diffusers transformers accelerate torch")
        return 0

    if torch.cuda.is_available():
        device     = "cuda"
        dtype      = torch.bfloat16  # bfloat16 is faster on Nvidia Ampere/Ada (30xx/40xx)
        speed_note = "~2-4s/image on 4090"
    elif torch.backends.mps.is_available():
        device     = "mps"
        dtype      = torch.float16
        speed_note = "~30-60s/image on Apple Silicon"
    else:
        print("[Local Flux] No GPU found (CUDA or MPS required). Use --source replicate instead.")
        return 0

    print(f"\n[Local Flux] Generating {count} images on {device.upper()} ({speed_note})...")
    print("  Loading Flux model (first run downloads ~24GB — be patient)...")

    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        if device == "mps":
            pipe.enable_attention_slicing()
    except Exception as e:
        print(f"  Failed to load Flux: {e}")
        return 0

    saved = 0
    pbar  = tqdm(total=count, desc="  Generating")

    try:
        while saved < count:
            prompt = random_prompt()
            try:
                result = pipe(
                    prompt,
                    num_inference_steps=4,   # schnell only needs 4 steps
                    guidance_scale=0.0,      # schnell doesn't use CFG
                    height=512,
                    width=512,
                )
                img  = result.images[0]
                name = f"flux_local_{hashlib.md5(prompt.encode()).hexdigest()[:12]}_{saved:05d}"
                if save_image(img, name):
                    saved += 1
                    pbar.update(1)
            except Exception as e:
                print(f"\n  generation error: {e}")
                time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n  Stopped early. Saved {saved} images.")

    pbar.close()
    print(f"[Local Flux] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 2: Replicate API
# ---------------------------------------------------------------------------
def collect_replicate(count: int):
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("\n[Replicate] Skipping — REPLICATE_API_TOKEN not set in .env")
        return 0

    print(f"\n[Replicate] Generating {count} images via API (~${count * 0.003:.2f} estimated)...")

    HEADERS = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    # Rotate between models for generator diversity
    MODELS = [
        "black-forest-labs/flux-schnell",
        "stability-ai/sdxl:39ed52f2319f9b7b4cc1b9d5af79838e",
    ]

    saved = 0
    pbar  = tqdm(total=count, desc="  Replicate")

    while saved < count:
        prompt = random_prompt()
        model  = random.choice(MODELS)

        try:
            # Start prediction
            resp = requests.post(
                f"https://api.replicate.com/v1/models/{model}/predictions"
                if "/" in model and ":" not in model
                else "https://api.replicate.com/v1/predictions",
                headers=HEADERS,
                json={
                    "version": model.split(":")[1] if ":" in model else None,
                    "input": {
                        "prompt": prompt,
                        "num_inference_steps": 4,
                        "width": 512,
                        "height": 512,
                    },
                },
                timeout=30,
            )
            pred = resp.json()
            pred_url = f"https://api.replicate.com/v1/predictions/{pred['id']}"

            # Poll for completion
            for _ in range(60):
                time.sleep(3)
                poll = requests.get(pred_url, headers=HEADERS, timeout=15).json()
                if poll["status"] == "succeeded":
                    output = poll.get("output")
                    img_url = output[0] if isinstance(output, list) else output
                    img_resp = requests.get(img_url, timeout=30)
                    img  = Image.open(io.BytesIO(img_resp.content))
                    name = f"replicate_{model.split('/')[0]}_{saved:05d}"
                    if save_image(img, name):
                        saved += 1
                        pbar.update(1)
                    break
                elif poll["status"] in ("failed", "canceled"):
                    break

        except Exception as e:
            print(f"\n  Replicate error: {e}")
            time.sleep(5)

    pbar.close()
    print(f"[Replicate] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 3: OpenAI DALL-E 3 (optional, for diversity)
# ---------------------------------------------------------------------------
def collect_dalle(count: int):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[DALL-E 3] Skipping — OPENAI_API_KEY not set in .env")
        return 0

    print(f"\n[DALL-E 3] Generating {count} images (~${count * 0.04:.2f} estimated)...")

    HEADERS = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    saved = 0
    pbar  = tqdm(total=count, desc="  DALL-E 3")

    while saved < count:
        prompt = random_prompt()
        try:
            resp = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=HEADERS,
                json={
                    "model":   "dall-e-3",
                    "prompt":  prompt,
                    "n":       1,
                    "size":    "1024x1024",
                    "quality": "standard",
                },
                timeout=60,
            )
            resp.raise_for_status()
            img_url  = resp.json()["data"][0]["url"]
            img_resp = requests.get(img_url, timeout=30)
            img      = Image.open(io.BytesIO(img_resp.content))
            name     = f"dalle3_{saved:05d}"
            if save_image(img, name):
                saved += 1
                pbar.update(1)
            time.sleep(1)  # rate limit
        except Exception as e:
            print(f"\n  DALL-E error: {e}")
            time.sleep(5)

    pbar.close()
    print(f"[DALL-E 3] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="local",
                        choices=["local", "replicate", "dalle", "all"])
    parser.add_argument("--count", type=int, default=500,
                        help="Number of images to generate")
    args = parser.parse_args()

    existing = len(list(OUTPUT_DIR.glob("*.jpg")))
    print(f"Output: {OUTPUT_DIR}")
    print(f"Already collected: {existing} images")

    total = 0

    if args.source == "all":
        # Split budget: 70% local (free), 20% replicate, 10% dalle
        local_count     = int(args.count * 0.70)
        replicate_count = int(args.count * 0.20)
        dalle_count     = int(args.count * 0.10)
        total += collect_local(local_count)
        total += collect_replicate(replicate_count)
        total += collect_dalle(dalle_count)
    elif args.source == "local":
        total += collect_local(args.count)
    elif args.source == "replicate":
        total += collect_replicate(args.count)
    elif args.source == "dalle":
        total += collect_dalle(args.count)

    final = len(list(OUTPUT_DIR.glob("*.jpg")))
    print(f"\nDone. Generated {total} new images. Total in directory: {final}")


if __name__ == "__main__":
    main()
