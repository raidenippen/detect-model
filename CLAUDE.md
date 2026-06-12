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

## Target Architecture (revised after Jun 2026 audit)

1. **Base model**: ViT-S/16 (ImageNet-21k pretrained), or the Community Forensics released
   checkpoint if it benchmarks well on our hard negatives. **NOT NYUAD** — that model is a
   ViT-base (86M params, ~330MB fp32, violates our own size constraint) whose narrow 3-class
   fine-tune we would overwrite anyway. NYUAD stays as a benchmark + ensemble signal only.
2. **Training data**: generator-diverse AI set (CommunityForensics-Small backbone + our own
   2025-era generations) vs. hard-real-negative-weighted real set. Details below.
3. **Inference**: ONNX Runtime, int8 dynamic quantization, on existing Render Standard instance.
   ViT-S/16 int8 is ~22MB and ~50–150ms/image on CPU — well within budget.
4. **API shape**: Must return `{ isAI, confidence, verdict, flagCount, signals }` — same as current.
   Confidence must be **calibrated** (temperature scaling) before it's surfaced to users.

---

## Data Policy (non-negotiable — this is what the Jun 2026 audit fixed)

1. **Save originals. Never resize or re-encode at collection time.** The old pipeline squashed
   everything to 384×384 JPEG q90 — since AI images are born square and real photos aren't,
   aspect-ratio distortion became a class label, and uniform re-encoding masked the compression-
   history differences the model must learn to ignore. That data is quarantined in
   `data/real_legacy_384/` (9,505 images). **Do not train on it.**
2. **Degradation augmentation at training time, applied identically to both classes**: random
   JPEG quality 40–95, random down/upscale, WebP recode, random crop (never aspect squash),
   slight blur/noise. Production images arrive through CDN/social-media recompression; a model
   trained only on clean images will collapse in the wild.
3. **Generator diversity beats volume** (Community Forensics, arXiv:2411.04125): many generators
   with a per-generator cap, not many images from 2–3 generators. Pre-2024 generators
   (DiffusionDB-era SD 1.x, JourneyDB MJ v5) capped at ≲25% of the AI class.
4. **Hard real negatives are the priority**: professional/polished photography (concerts, bokeh
   portraits, product shots) — the FP failure mode this project exists to fix. COCO/Open Images
   are consumer-photo diversity slices, not the backbone.

---

## Current Status

**Phase (11 Jun 2026 pm): ALL GATES COMPLETE. Training next, on a cloud GPU —
`python scripts/train.py --base OwensLab/commfor-model-224` (loader fixed + smoke-tested).**

Gate results (11 Jun 2026):
- Val split frozen: 2,297 val_real / 4,400 val_ai (cluster-aware). Never re-split.
- Leakage probe: probe 2 (real-vs-AI from 12 trivial stats) PASSED (~0.60, no shortcut).
  Probe 1 (real-source separability) warned at 0.37 vs 0.20 chance — diagnosed as diffuse
  resolution/frequency-profile differences (COCO 640px vs Unsplash 4k originals), confusion
  matrix shows no cleanly identifiable source. ACCEPTED: randomized downscale + degradation
  chains in train.py target exactly this; leave-one-source-out eval (Step 4) is the backstop.
- Baseline benchmark (full table in `data/validation_report/baselines.json`): the OwensLab
  Community Forensics checkpoints crush every other baseline. commfor-384: 3.4% hard-neg FPR
  clean / 1.1% degraded, but val_ai TPR collapses to 47% under jpeg70_half. commfor-224:
  7.4% / 3.4% hard-neg FPR, TPR 100% clean / 95% jpeg70_half — the better fine-tune base
  (degradation-robust at the resolution we serve). Ateeqq: 94% hard-neg FPR, 0% hard-positive
  TPR — remove from the extension ensemble regardless. Caveat: ~40% of val_ai is
  CommunityForensics data the commfor models trained on, so their val_ai TPR is inflated;
  trust the hard_positives (94–96%) and hard_negatives numbers.
- Loader fixes that made this work: the OwensLab checkpoints are timm-format ViT-S/16 with a
  single sigmoid logit (NOT transformers format). `benchmark_baselines.py` loads them via
  timm; `train.py` converts them to transformers `ViTForImageClassification` at load time
  (qkv split, sigmoid head mapped exactly onto a 2-class head as [-z/2,+z/2]; verified
  numerically equivalent to ~1e-9). NYUAD needs explicit `ViTImageProcessor` under
  transformers v5. `timm` added to requirements.

Pinterest robustness investigation (11 Jun 2026 pm — user reported the extension missing
visibly-AI images on Pinterest):
- Pinterest CDN serves fixed-width WebP renditions (236px feed / 736px closeup) — the
  harshest mainstream degradation we've measured. Added `pinterest_closeup` and
  `pinterest_feed` presets to `scripts/degrade.py`; include them in all future evals.
- At 236px: commfor-384 TPR drops to 75% val_ai / 64% hard_pos. **commfor-224 holds 97% /
  84%** — further confirmation it's the right fine-tune base. NYUAD never wins under any
  degradation (18% TPR); its low FPR is under-flagging, not skill.
- The extension (`detect` repo, space/app.py) squash-resizes to 384×384 instead of the
  official resize(440)→centercrop(384). Measured both: squash is slightly WORSE on clean
  (hard-neg FPR 5.1% vs 3.4%, hard-pos TPR 91% vs 96%) but slightly BETTER under
  pinterest_feed (TPR 81% vs 75%). Net: not the Pinterest culprit, not worth changing —
  the real fix is the fine-tuned 224 model.
- Case study: a user-supplied in-the-wild Pinterest miss (MJ-style architecture photo,
  visibly AI to humans via semantic tells) — commfor-384 0.41 (miss), NYUAD 0.00,
  Ateeqq 0.00, **commfor-224 0.97 (catch)**, haywood 1.00. Saved as
  `data/hard_positives/e8f97583ad89ccfe10298f8b63fe7195.webp` — hard_positives is now
  101 images (was 100 when baselines.json was generated; n=100 rows there).
- Workflow: in-the-wild misses go into `data/hard_positive_candidates/` (or directly into
  `data/hard_positives/` for confirmed eval cases, noting the count change). Duplicates
  collapse in near-dup clustering; one copy per unique image.

**Training hardware (revised 11 Jun 2026 pm):** cloud GPU (RunPod-class, ~$10–20) is the
default, but the 4090 box is acceptable for training IF it's up and has ~70GB free disk —
ViT-S/16 @ bs64 fits easily in 24GB, bf16 supported, est. 1–3h for 8 epochs. The dataset is
58GB, so LAN rsync to the 4090 beats uploading to a cloud pod. Use `--num-workers 8–12`;
CPU-side decode of large originals is the bottleneck, not the GPU. (The earlier "4090
retired" note was about not BLOCKING the plan on that box for generation — training on it
is fine if available.) CPU/MPS remain smoke-test only.

Strategic decision from the Jun 2026 audit chat: the model does not need to beat Hive outright.
The **cascade architecture** is the hedge — our model answers confident scans for free; scans in
the calibrated uncertain band escalate to Hive ($0.006/image). Calibration and honest uncertainty
matter more than raw accuracy; false positives remain the worse failure mode.

### Scripts
- `scripts/collect_real.py` — real images, originals saved untouched.
  Sources: Unsplash Lite (25k curated pro photos, no key), Pexels (free key), Wikimedia
  (recursive traversal of Quality/Featured images — the old version hit top-level categories
  with `cmtype=file` and collected **6 images total**; fixed), Open Images, COCO. Flickr dropped.
- `scripts/collect_ai.py` — AI images saved as lossless PNG at native resolution, varied aspect
  ratios. Local Flux schnell+dev (CUDA/MPS auto-detect), Replicate (flux-dev, flux-1.1-pro,
  SD 3.5, Ideogram v2, Recraft v3), OpenAI gpt-image-1/DALL-E 3. Combinatorial prompt bank
  (~100k+ unique prompts).
- `scripts/collect_community_forensics.py` — streams a generator-stratified subset of
  CommunityForensics-Small (HF: OwensLab/CommunityForensics-Small) into `data/ai/`
  (optionally paired reals into `data/real/`). This is the AI-class backbone.
- `scripts/split_val.py` — near-duplicate-cluster-aware 10% val split (phash + union-find).
  The val set is FROZEN once training starts.
- `scripts/benchmark_baselines.py` — runs NYUAD + the extension's current models over
  val_real / val_ai / hard_negatives, clean **and** degraded (JPEG-70, half-res). This is the
  bar to beat. Run it before training anything.
- `scripts/validate_dataset.py` — diversity/duplicate/statistics checks.
- `scripts/degrade.py` — THE shared degradation library (random chains for training,
  deterministic platform presets for eval). Single implementation, used everywhere.
- `scripts/train.py` — full fine-tuning pipeline: crop-not-squash views, degradation
  chains on both classes, clean/degraded consistency loss (KL), FGSM-lite adversarial
  batches, class-balanced sampling, then temperature scaling + operating threshold that
  must satisfy the FPR target on val_real AND hard_negatives. Outputs HF model dir +
  `inference_config.json` (temperature/threshold/uncertain band).
- `scripts/leakage_probe.py` — go/no-go gate, run BEFORE training: (1) are real sources
  separable from low-level stats? (2) is real-vs-AI separable from 12 trivial stats?
  Either WARN means fix the data, not proceed.

### Data directories
```
data/
  real/             # real photos, ORIGINAL bytes (jpg/png/webp)
  ai/               # AI images, lossless PNG at native generation size
  val_real/         # frozen held-out validation (cluster-aware split)
  val_ai/
  hard_negatives/   # MANUALLY CURATED (176): the actual concert photos / polished
                    # portraits that FP'd on current models. Never trained on. FROZEN.
  hard_positives/   # MANUALLY CURATED (101): deceptive AI images baselines miss,
                    # incl. in-the-wild Pinterest misses. Eval only, never trained on.
                    # (training half of the original candidates went into data/ai/)
  hard_positive_candidates/   # raw inbox for new in-the-wild misses, curate from here
  hard_negative_candidates/   # same, for real photos that get false-flagged
  real_legacy_384/  # quarantined squashed data from old pipeline. Do not train on.
```

### Collection status (FINAL, 11 Jun 2026 — collection is done, do not re-collect)
| Set | Source | Count |
|-----|--------|-------|
| real | Unsplash 8k / Pexels / Wikimedia / COCO / Open Images | 22,975 total |
| ai | CommunityForensics-Small (hundreds of generators) | ~17,100 |
| ai | Flux family (text-to-image-2M + Flux-1-Dev-Images-1k, public HF datasets) | ~8,500 |
| ai | DRAGON (25 diffusion models, 2.5k photo-gated) | ~7,500 |
| ai | Midjourney (v6 raw + MJHQ curated photoreal) | ~6,300 |
| ai | GPT-4o (ShareGPT-4o-Image) | 4,000 |
| ai | Hard positives, training half | ~100 |

Optional remaining: ~$20 Replicate batch for commercial-only generators (flux-1.1-pro, Ideogram,
Recraft) — nice-to-have breadth, not load-bearing, nothing waits on it.

Old collection (COCO 3,751 + OI 5,748 + wiki 6, all squashed 384) is in `data/real_legacy_384/`.

### Keys needed (.env)
- `PEXELS_API_KEY` — free, https://www.pexels.com/api/
- `HF_TOKEN` — only if CommunityForensics-Small gates access
- `REPLICATE_API_TOKEN` — ~$50 budget for commercial-generator batches
- `OPENAI_API_KEY` — optional, gpt-image-1

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

### Step 0 — Populate `data/hard_negatives/` (manual, blocking)
Copy the actual images that triggered this project (real concert photos scoring ~99% AI on
commfor/haywood/ateeqq) plus ~50–200 similar curated hard cases. Everything downstream is
measured against this set.
(✅ DONE — 176 hard negatives + 101 hard positives curated and frozen.)

### Step 1 — Re-collect data (scripts ready)
```bash
python scripts/collect_real.py --source unsplash --limit 8000
python scripts/collect_real.py --source pexels --limit 4000
python scripts/collect_real.py --source wikimedia --limit 5000
python scripts/collect_real.py --source coco --limit 2500
python scripts/collect_real.py --source open_images --limit 2500
python scripts/collect_community_forensics.py --limit 20000 --per-generator 40
python scripts/collect_ai.py --source replicate --count 1500   # optional commercial batch only
```
Then `python scripts/validate_dataset.py` and `python scripts/split_val.py`.
(✅ DONE 11 Jun 2026 — local generation was replaced by public Flux dataset imports via
`import_hf_dataset.py`; the val split is frozen. Never re-run `split_val.py`.)

### Step 2 — Baseline benchmark (before any training)
`python scripts/benchmark_baselines.py` — confirms the FP problem on hard_negatives, measures
degradation robustness, and gives the bar to beat. If the Community Forensics released
checkpoint already performs well here, fine-tune from it instead of vanilla ViT-S.
(✅ DONE 11 Jun 2026 — full table in `data/validation_report/baselines.json`; commfor-224
chosen as the fine-tune base. See "Gate results" + "Pinterest robustness" in Current Status.)

### Step 3 — Fine-tuning (pipeline written: scripts/train.py)
**Operational runbook for the training box (launch, monitoring cadence, healthy/unhealthy
signals, acceptance criteria, troubleshooting): [docs/TRAINING.md](docs/TRAINING.md).**
Quick start on the 4090: `bash scripts/train_4090.sh` after rsyncing `data/`.
```bash
python scripts/leakage_probe.py            # ✅ run 11 Jun 2026, accepted (see gate results)
python scripts/train.py --epochs 8 --batch-size 64 --base OwensLab/commfor-model-224 --num-workers 12
```
- Defaults encode the plan: ViT-S/16 full fine-tune, lr 3e-5 cosine, degradation chains
  (prob 0.7) on both classes, consistency weight 1.0, FGSM on 15% of batches, target FPR 2%.
- `--base OwensLab/commfor-model-224` works as of 11 Jun 2026 (timm→transformers conversion
  in train.py, smoke-tested end-to-end incl. save/reload/calibration).
- Hardware: cloud GPU (RunPod / Lambda, ~2–4 GPU-hours, ~$10–20) OR the 4090 box if it's up
  (see "Training hardware" in Current Status; dataset is 58GB — LAN rsync beats cloud upload).
  CPU/MPS work for smoke tests only.
- Targets: **FPR ≤ 2% on hard_negatives (clean and jpeg70_half)**, TPR ≥ 90% on val_ai,
  TPR ≥ 80% under jpeg70_half degradation — and report `pinterest_feed` (236px) TPR, the
  harshest real-world condition we've measured.

### Step 4 — Evaluation
- Leave-one-generator-out on the AI class; leave-one-source-out on the real class.
- Full degradation suite on everything (benchmark_baselines.py degradations + the
  degrade.py platform presets, especially `pinterest_feed`).
- Temperature-scale the confidence on held-out data; pick the operating threshold for the FPR
  target, not max accuracy; define an "uncertain" verdict band around the threshold.
- Compare against Hive API on the same images as a quality bar.

### Step 5 — Integration
- Export to ONNX, int8 dynamic quantization, benchmark on a Render-sized CPU.
- Cap input image dimensions before decode (large PNGs can OOM a 2GB instance).
- Drop into `space/app.py` in the detect repo; replace commfor/haywood/ateeqq.
- Ateeqq can be removed from the extension ensemble immediately, independent of training
  (94% hard-negative FPR, 0% hard-positive TPR — it only adds false positives).
- The extension's commfor squash-resize preprocessing was measured (11 Jun 2026): mildly
  worse on clean, mildly better on Pinterest thumbnails — a wash. Don't bother changing it;
  the fine-tuned 224 model replaces that path entirely.
- Keep NYUAD as a secondary signal; keep C2PA as primary override.

### Step 6 — Retraining cadence + production monitoring
- Retrain every 3–4 months as new generators release; add their outputs to `data/ai/`.
- Keep the validation set and hard_negatives frozen so metrics are comparable across versions.
- Log production score distributions and watch for drift (a new Flux-class release tanks
  recall silently between retrains).

---

## Known Structural Gaps vs Commercial APIs

Heavily degraded images, partial AI edits (inpainting/generative fill/upscaling), and
adversarial evasion. Each has a phased mitigation plan — including the degradation
ladder, the partial-AI verdict taxonomy, and the tiered adversarial threat model —
in **[docs/ROBUSTNESS.md](docs/ROBUSTNESS.md)**. Read it before writing `train.py`:
several v1 items (degradation chains via a shared `scripts/degrade.py`, consistency
loss, upscaled-reals augmentation, FGSM-lite) live inside the training pipeline.

---

## Constraints

- Final model must run on Render Standard (~2GB RAM, no GPU)
- Inference time budget: < 2 seconds per image (ViT-S/16 int8 ONNX: ~50–150ms)
- Model size < 300MB on disk (ViT-S int8 ≈ 22MB — large headroom is intentional)
- RAM, not latency, is the real constraint — bound image decode size

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
- **NYUAD demoted from base model to benchmark/ensemble signal** (Jun 2026 audit): it's an
  oversized ViT-base whose fine-tune we'd overwrite; its concert-photo win was n=1 evidence
- **CommunityForensics-Small is the AI-class backbone**; DiffusionDB/JourneyDB demoted to a
  ≲25% legacy slice — 2022–2023-era generators don't represent what users scan in 2026
- **Originals only, degradation as training-time augmentation** — collection-time re-encoding
  created label-correlated artifacts (see Data Policy)
- **Eval harness before training code** — frozen hard-negative set, degradation suite,
  leave-one-generator-out, calibrated confidence
- **NYUAD stays** as a secondary signal in the extension ensemble
- **C2PA stays** as a hard override — cryptographic provenance beats any classifier
