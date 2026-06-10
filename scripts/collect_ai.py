"""
Generate AI images for training data.

IMPORTANT: images are saved as LOSSLESS PNG at native generation resolution —
no resizing, no JPEG re-encode. Degradation (resize/JPEG/WebP) happens at
training time as augmentation, applied identically to both classes. Expect
~1.5-2.5MB per image on disk.

Diversity strategy (per the Community Forensics finding: number of distinct
generators matters more than images per generator):
  - The bulk of AI training data should come from CommunityForensics-Small
    (see collect_community_forensics.py). This script adds CURRENT-generation
    coverage on top: Flux schnell+dev locally, plus small API batches of
    commercial 2025-era models.
  - Prompts are built combinatorially from scene templates (~100k+ unique
    combinations) instead of a fixed list of 30.
  - Generation uses native resolutions and varied aspect ratios — never a
    single square size, which would become a class-correlated artifact.

Usage:
  python collect_ai.py --source local --model both --count 5000   # 4090 box
  python collect_ai.py --source replicate --count 1000
  python collect_ai.py --source openai --count 300
"""

import os
import io
import time
import random
import hashlib
import argparse
import base64
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# An empty HF_TOKEN= line in .env crashes huggingface_hub with an illegal
# "Bearer " header — treat empty as absent.
if not os.environ.get("HF_TOKEN"):
    os.environ.pop("HF_TOKEN", None)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ai"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Combinatorial prompt bank. Each template is (format string, slot options).
# Templates keep subject/setting semantically coherent; lighting and camera
# suffixes multiply diversity. ~10 templates x slots x suffixes >> 100k combos.
# ---------------------------------------------------------------------------
LIGHTING = [
    "studio lighting", "natural window light", "golden hour light",
    "dramatic side lighting", "soft overcast light", "neon lights",
    "spotlight", "dappled sunlight", "moody low-key lighting", "backlit",
]
CAMERA = [
    "shot on Canon EOS R5, 85mm f/1.4", "shot on Nikon Z9, 50mm",
    "shot on Sony A7IV, 35mm", "Leica aesthetic", "telephoto compression",
    "wide angle 24mm", "medium format look", "35mm film grain",
    "editorial photography", "documentary photography style", "",
]

TEMPLATES = [
    ("professional portrait of {subj}, {detail}, bokeh background", {
        "subj": ["a young woman with curly hair", "an elderly fisherman", "a businessman",
                 "a teenage athlete", "a chef in whites", "a violinist", "a construction worker",
                 "a bride", "twin sisters", "a man with a beard and glasses"],
        "detail": ["shallow depth of field", "sharp focus on eyes", "candid expression",
                   "laughing", "serious expression", "looking away from camera"],
    }),
    ("concert photo of {subj} on stage, {detail}", {
        "subj": ["a rock band", "a female singer with guitar", "a jazz trumpeter",
                 "a DJ at a festival", "an orchestra", "a rapper", "a drummer mid-solo",
                 "a country singer", "a metal band", "a gospel choir"],
        "detail": ["dramatic stage lighting, crowd in background", "spotlight, bokeh crowd",
                   "colorful lights, smoke", "silhouetted against strobes",
                   "crowd hands raised", "moody club lighting"],
    }),
    ("{style} photo of {subj}", {
        "style": ["wedding", "fashion editorial", "sports action", "photojournalism",
                  "travel", "reportage"],
        "subj": ["a couple at golden hour", "a model in designer clothing",
                 "a basketball player dunking", "a soccer player mid-kick",
                 "a busy street market", "children playing in a fountain",
                 "a protest march", "a craftsman in his workshop",
                 "commuters on a rainy platform", "a dancer mid-leap"],
    }),
    ("product photography of {subj}, {detail}", {
        "subj": ["a luxury watch", "a perfume bottle", "running shoes", "a leather handbag",
                 "a smartphone", "a whiskey bottle", "headphones", "a ceramic mug"],
        "detail": ["on marble surface, soft studio lighting", "floating with dramatic shadows",
                   "on dark slate, water droplets", "minimalist white background",
                   "surrounded by ingredients", "macro detail shot"],
    }),
    ("food photography of {subj}, {detail}", {
        "subj": ["a gourmet pasta dish", "a charcuterie board", "ramen with soft egg",
                 "a layered chocolate cake", "fresh sushi", "a burger with melted cheese",
                 "a colorful salad bowl", "croissants on a wooden board"],
        "detail": ["top down shot, natural light", "45 degree angle, steam rising",
                   "dark moody styling", "bright airy styling", "close macro crop"],
    }),
    ("landscape photo of {subj}, {detail}", {
        "subj": ["a mountain range at sunset", "a coastline from above", "rolling fog over hills",
                 "a desert with dunes", "a glacier lagoon", "autumn forest path",
                 "rice terraces", "a stormy seascape", "wildflower meadow"],
        "detail": ["dramatic clouds, golden hour", "long exposure water", "aerial drone view",
                   "misty morning", "starry night sky", "after rain, saturated colors"],
    }),
    ("street photography in {subj}, {detail}", {
        "subj": ["Tokyo at night", "New York in winter", "Havana", "Mumbai", "Paris",
                 "Seoul", "Mexico City", "London in the rain"],
        "detail": ["neon reflections on wet pavement", "candid pedestrians, motion blur",
                   "black and white, high contrast", "golden hour shadows",
                   "steam from a food cart", "umbrellas from above"],
    }),
    ("wildlife photography of {subj}, {detail}", {
        "subj": ["a lion in savanna grass", "a hummingbird at a flower", "a fox in snow",
                 "an eagle in flight", "elephants at a waterhole", "a leopard in a tree",
                 "a bear catching salmon", "penguins on ice"],
        "detail": ["telephoto lens, natural light", "motion blur background", "eye-level close-up",
                   "backlit at dawn", "rain, dramatic mood"],
    }),
    ("architecture photo of {subj}, {detail}", {
        "subj": ["a modern glass skyscraper", "a brutalist library", "a gothic cathedral interior",
                 "a spiral staircase", "a Japanese temple", "an art deco theater"],
        "detail": ["dramatic sky, wide angle", "symmetrical composition", "blue hour, lit windows",
                   "looking straight up", "minimalist with single person for scale"],
    }),
    ("macro photography of {subj}, {detail}", {
        "subj": ["a flower with dew drops", "a butterfly wing", "a spider web with rain",
                 "an eye close-up", "frost patterns on glass", "a bee on lavender"],
        "detail": ["shallow depth of field", "natural light", "dark background", "backlit"],
    }),
]

# Aspect-ratio buckets (Flux/SDXL-friendly). Never a single fixed square.
ASPECTS = [(1024, 1024), (1152, 896), (896, 1152), (1344, 768), (768, 1344)]
ASPECT_STRINGS = ["1:1", "4:3", "3:4", "16:9", "9:16"]


def random_prompt() -> str:
    tmpl, slots = random.choice(TEMPLATES)
    filled = tmpl.format(**{k: random.choice(v) for k, v in slots.items()})
    parts = [filled, random.choice(LIGHTING)]
    cam = random.choice(CAMERA)
    if cam:
        parts.append(cam)
    return ", ".join(parts)


def save_png(img: Image.Image, name: str) -> bool:
    try:
        img.convert("RGB").save(OUTPUT_DIR / f"{name}.png", "PNG")
        return True
    except Exception as e:
        print(f"  save failed: {e}")
        return False


def save_bytes(data: bytes, prefix: str) -> bool:
    """Save generator output bytes as-is when already lossless, else as PNG."""
    try:
        img = Image.open(io.BytesIO(data))
        name = f"{prefix}_{hashlib.md5(data).hexdigest()[:16]}"
        if img.format == "PNG":
            (OUTPUT_DIR / f"{name}.png").write_bytes(data)
            return True
        if img.format == "WEBP" and not getattr(img, "is_animated", False):
            (OUTPUT_DIR / f"{name}.webp").write_bytes(data)
            return True
        return save_png(img, name)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Source 1: Local Flux — schnell (fast) and dev (higher quality, what people
# actually share). Auto-detects CUDA / MPS.
# ---------------------------------------------------------------------------
def collect_local(count: int, model: str):
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError:
        print("diffusers not installed. Run: pip install diffusers transformers accelerate torch")
        return 0

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        print("[Local Flux] No GPU found (CUDA or MPS required). Use --source replicate instead.")
        return 0

    variants = ["schnell", "dev"] if model == "both" else [model]
    saved_total = 0

    for variant in variants:
        n = count // len(variants)
        repo = f"black-forest-labs/FLUX.1-{variant}"
        steps, guidance = (4, 0.0) if variant == "schnell" else (28, 3.5)
        print(f"\n[Local Flux-{variant}] Generating {n} images on {device.upper()} "
              f"({steps} steps — dev is ~7x slower than schnell)...")
        try:
            pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype)
            if device == "cuda":
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb < 30:
                    # Flux (12B transformer + T5-XXL) does not fit a 24GB card
                    # fully resident — offload swaps components to system RAM.
                    print(f"  {vram_gb:.0f}GB VRAM — using model CPU offload")
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(device)
            else:
                pipe = pipe.to(device)
                if device == "mps":
                    pipe.enable_attention_slicing()
        except Exception as e:
            print(f"  Failed to load {repo}: {e}")
            if variant == "dev":
                print("  NOTE: FLUX.1-dev is a gated model. Accept the license at")
                print("  https://huggingface.co/black-forest-labs/FLUX.1-dev and run:")
                print("    hf auth login")
            continue

        saved = 0
        pbar = tqdm(total=n, desc=f"  flux-{variant}")
        try:
            while saved < n:
                prompt = random_prompt()
                w, h = random.choice(ASPECTS)
                try:
                    img = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance,
                               height=h, width=w).images[0]
                    name = f"flux_{variant}_{hashlib.md5(prompt.encode()).hexdigest()[:12]}_{saved:05d}"
                    if save_png(img, name):
                        saved += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"\n  generation error: {e}")
                    time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n  Stopped early. Saved {saved} images.")
        pbar.close()
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        saved_total += saved
        print(f"[Local Flux-{variant}] Saved {saved} images.")

    return saved_total


# ---------------------------------------------------------------------------
# Source 2: Replicate — small batches across many 2025-era commercial models.
# Breadth per dollar beats depth: a few hundred images per generator is enough.
# ---------------------------------------------------------------------------
REPLICATE_MODELS = [
    # (owner/name, input builder, ~$/image)
    ("black-forest-labs/flux-dev",
     lambda p: {"prompt": p, "aspect_ratio": random.choice(ASPECT_STRINGS),
                "output_format": "png"}, 0.025),
    ("black-forest-labs/flux-1.1-pro",
     lambda p: {"prompt": p, "aspect_ratio": random.choice(ASPECT_STRINGS),
                "output_format": "png"}, 0.04),
    ("stability-ai/stable-diffusion-3.5-large",
     lambda p: {"prompt": p, "aspect_ratio": random.choice(ASPECT_STRINGS),
                "output_format": "png"}, 0.065),
    ("ideogram-ai/ideogram-v2-turbo",
     lambda p: {"prompt": p, "aspect_ratio": random.choice(ASPECT_STRINGS)}, 0.05),
    ("recraft-ai/recraft-v3",
     lambda p: {"prompt": p, "size": "1024x1024"}, 0.04),
]

def collect_replicate(count: int):
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("\n[Replicate] Skipping — REPLICATE_API_TOKEN not set in .env")
        return 0

    est = sum(c for _, _, c in REPLICATE_MODELS) / len(REPLICATE_MODELS) * count
    print(f"\n[Replicate] Generating {count} images across {len(REPLICATE_MODELS)} models (~${est:.2f})...")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    saved = 0
    pbar = tqdm(total=count, desc="  Replicate")
    while saved < count:
        model, build_input, _ = REPLICATE_MODELS[saved % len(REPLICATE_MODELS)]
        prompt = random_prompt()
        try:
            resp = requests.post(
                f"https://api.replicate.com/v1/models/{model}/predictions",
                headers={**headers, "Prefer": "wait=60"},
                json={"input": build_input(prompt)},
                timeout=90,
            )
            resp.raise_for_status()
            pred = resp.json()
            for _ in range(30):
                if pred.get("status") in ("succeeded", "failed", "canceled"):
                    break
                time.sleep(3)
                pred = requests.get(
                    f"https://api.replicate.com/v1/predictions/{pred['id']}",
                    headers=headers, timeout=20,
                ).json()
            if pred.get("status") != "succeeded":
                continue
            output = pred.get("output")
            img_url = output[0] if isinstance(output, list) else output
            img_resp = requests.get(img_url, timeout=60)
            if save_bytes(img_resp.content, f"rep_{model.split('/')[1].replace('-', '_')}"):
                saved += 1
                pbar.update(1)
        except Exception as e:
            print(f"\n  Replicate error ({model}): {e}")
            time.sleep(5)
    pbar.close()
    print(f"[Replicate] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Source 3: OpenAI — gpt-image-1 (current) or dall-e-3 (legacy artifacts)
# ---------------------------------------------------------------------------
def collect_openai(count: int, model: str = "gpt-image-1"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[OpenAI] Skipping — OPENAI_API_KEY not set in .env")
        return 0

    print(f"\n[OpenAI {model}] Generating {count} images...")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sizes = (["1024x1024", "1536x1024", "1024x1536"] if model == "gpt-image-1"
             else ["1024x1024", "1792x1024", "1024x1792"])

    saved = 0
    pbar = tqdm(total=count, desc=f"  {model}")
    while saved < count:
        body = {"model": model, "prompt": random_prompt(), "n": 1,
                "size": random.choice(sizes)}
        if model == "dall-e-3":
            body["quality"] = "standard"
            body["response_format"] = "b64_json"
        try:
            resp = requests.post("https://api.openai.com/v1/images/generations",
                                 headers=headers, json=body, timeout=180)
            resp.raise_for_status()
            item = resp.json()["data"][0]
            if "b64_json" in item:
                data = base64.b64decode(item["b64_json"])
            else:
                data = requests.get(item["url"], timeout=60).content
            prefix = "gptimg" if model == "gpt-image-1" else "dalle3"
            if save_bytes(data, prefix):
                saved += 1
                pbar.update(1)
            time.sleep(1)
        except Exception as e:
            print(f"\n  OpenAI error: {e}")
            time.sleep(5)
    pbar.close()
    print(f"[OpenAI {model}] Saved {saved} images.")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="local",
                        choices=["local", "replicate", "openai", "all"])
    parser.add_argument("--model", default="both", choices=["schnell", "dev", "both"],
                        help="Flux variant for --source local")
    parser.add_argument("--openai-model", default="gpt-image-1",
                        choices=["gpt-image-1", "dall-e-3"])
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    existing = sum(1 for _ in OUTPUT_DIR.iterdir())
    print(f"Output: {OUTPUT_DIR}")
    print(f"Already collected: {existing} images")

    total = 0
    if args.source == "all":
        total += collect_local(int(args.count * 0.70), args.model)
        total += collect_replicate(int(args.count * 0.20))
        total += collect_openai(int(args.count * 0.10), args.openai_model)
    elif args.source == "local":
        total += collect_local(args.count, args.model)
    elif args.source == "replicate":
        total += collect_replicate(args.count)
    elif args.source == "openai":
        total += collect_openai(args.count, args.openai_model)

    final = sum(1 for _ in OUTPUT_DIR.iterdir())
    print(f"\nDone. Generated {total} new images. Total in directory: {final}")


if __name__ == "__main__":
    main()
