# CLAUDE.md — detect-model

This file is the canonical handoff document for any AI agent or developer picking up this project.
Read this before touching any code.

---

## What This Is

A fine-tuned image classification model for detecting AI-generated images. Intended to replace the
open-source HuggingFace models (commfor, haywood, ateeqq) used in the Detect Chrome extension
(separate repo: github.com/raidenippen/detect), which have unacceptably high false-positive rates
on polished real photography (~99% AI confidence on real concert photos).

The trained model will be hosted on the existing Render inference service and called by the Detect
backend. No changes to the extension are required — the API response shape stays the same.

---

## Why We're Building This

- Hive AI detection API works well but costs $0.006/image — unit economics don't work at $10/month Pro
- Open-source HF models (commfor384, haywood, ateeqq) have very high FP rates on professional photography
- Training our own model eliminates per-scan API costs beyond existing Render infrastructure
- We own the model and can retrain as new AI generators emerge

---

## Target Architecture

1. **Base model**: Fine-tuned ViT (Vision Transformer) or CLIP — to be decided after initial experiments
2. **Training data**: ~20,000–50,000 labeled images (AI vs real), weighted toward hard real negatives
3. **Inference**: Hosted on existing Render Standard instance (detect-inference service)
4. **API shape**: Must return `{ isAI, confidence, verdict, flagCount, signals }` — same as current

---

## Current Status

**Phase: Step 1 in progress — real image collection running.**

### What exists
- `scripts/collect_real.py` — collects real images from Open Images, COCO, Wikimedia, Flickr
- `scripts/collect_ai.py` — generates AI images via local Flux (CUDA/MPS), Replicate, DALL-E 3
- `scripts/split_val.py` — splits 10% of collected images into validation directories
- `scripts/validate_dataset.py` — checks diversity, duplicates, statistics, scene coverage
- `requirements.txt` — all Python dependencies
- `.env.example` — copy to `.env` and fill in API keys

### Real image collection status (as of Jun 2026)
| Source | Status | Count |
|--------|--------|-------|
| COCO | ✅ Done | 3,751 |
| Open Images | ✅ Done | 5,748 |
| Wikimedia | 🔄 Running | ~3,500 target |
| **Total** | | **~13,000** |

Flickr dropped — commercial API approval required, not worth the friction.

### AI image collection status
- Not started. Run on the Windows 4090 machine.
- Script auto-detects CUDA — no code changes needed.
- Target: 7,500–10,000 images via local Flux (free).

### Keys needed
- `REPLICATE_API_TOKEN` — https://replicate.com (optional top-up)
- `OPENAI_API_KEY` — optional, DALL-E 3 only

### Setup (on any machine)
```bash
git clone https://github.com/raidenippen/detect-model
cd detect-model
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Next Steps (in order)

### Step 1 — Data collection scripts
Write Python scripts to collect labeled training data:

**Real images (target: 15,000–25,000):**
- Unsplash API (free, CC0) — prioritize hard cases: concerts, portraits with bokeh, professional
  product photography, heavily post-processed images. These are exactly what current models fail on.
- LAION or COCO subsets for volume and diversity
- Script should download, resize to 384×384, save as JPEG quality 90, label as `real`

**AI images (target: 15,000–25,000):**
- Replicate API (Flux, Stable Diffusion XL) — cheapest, ~$0.003/image
- DALL-E 3 via OpenAI API — ~$0.04/image, use sparingly for diversity
- Prompts should be diverse: portraits, landscapes, concerts, product shots, abstract
  — specifically include prompts that mimic the hard real-photo cases
- Script should save images labeled as `ai`

**Output structure:**
```
data/
  real/       # real photos
  ai/         # AI-generated images
  val_real/   # held-out validation set (10% of real)
  val_ai/     # held-out validation set (10% of AI)
```

**APIs needed:**
- Unsplash API key: https://unsplash.com/developers (free)
- Replicate API key: https://replicate.com (paid, ~$0.003/image for Flux)
- OpenAI API key: optional, for DALL-E 3 diversity

### Step 2 — EDA and baseline
- Check class balance, image size distribution
- Run existing HF models (commfor, haywood, ateeqq) against the validation set to confirm FP rates
- This gives a benchmark to beat

### Step 3 — Fine-tuning pipeline
- Fine-tune ViT-small or CLIP ViT-B/32 on the training set
- Use PyTorch + HuggingFace Transformers
- Train on cloud GPU: RunPod or Lambda Labs (~$50–200 one-time)
- Target: FP rate < 5% on hard real negatives, TP rate > 90% on current AI generators

### Step 4 — Evaluation
- Measure per-class accuracy, FP rate, FN rate on held-out validation set
- Specifically test against the known failure case: concert photo, polished portrait, product photography
- Compare against Hive API results on same images as a quality bar

### Step 5 — Integration
- Export model to ONNX or keep as PyTorch
- Drop into existing `space/app.py` in the detect repo as a new pipeline
- Replace commfor/haywood/ateeqq with this model
- Keep NYUAD as a secondary signal (it performed well on the concert FP case)
- Keep C2PA as primary override signal

### Step 6 — Retraining cadence
- Retrain every 3–4 months as new AI generators release
- Add new generator outputs to `data/ai/` and rerun from Step 3
- Keep validation set stable so metrics are comparable across versions

---

## Constraints

- Final model must run on Render Standard (~2GB RAM, no GPU)
- Inference time budget: < 2 seconds per image
- ViT-small is ~22M params and fits comfortably; ViT-base is borderline
- Do not use ViT-large or any model > 300MB

---

## Related Repo

The Chrome extension + backend that will consume this model:
- Repo: https://github.com/raidenippen/detect
- Inference server: `space/app.py` (FastAPI, runs on Render as detect-inference)
- The trained model replaces pipelines loaded at startup in `app.py`

---

## Key Decisions Made

- **Own the model** rather than pay per-scan API (Hive $0.006/image doesn't work at $10/month Pro)
- **Hard real negatives are the priority** — the failure mode is FP on polished photography, not FN on AI
- **NYUAD stays** as a secondary signal — its 3-class architecture performed well on the FP case
- **C2PA stays** as a hard override — cryptographic provenance beats any classifier
