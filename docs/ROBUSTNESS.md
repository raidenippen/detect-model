# Robustness Plans — the three places we structurally lose

This doc covers the three known structural weaknesses vs. commercial APIs
(Hive/SightEngine) and what we do about each: **heavily degraded images**,
**partial AI edits**, and **adversarial evasion**. Retraining cadence/staleness
is covered in CLAUDE.md Step 6 and is out of scope here.

Phasing used below:
- **v1** — ships with the first trained model. Cheap, no new architecture.
- **v1.5** — first retrain cycle. Needs new data generation, no new serving code.
- **v2** — needs product/serving changes or a second model. Only if v1 metrics
  justify it.

Honest framing up front: degraded images are a *narrowable* gap, partial AI is
a *policy + data* problem we can handle at the common-case level, adversarial
robustness is a *cost-raising* exercise — a determined attacker wins against
us, Hive, and everyone else. The goal is to fail less often and more honestly,
not to be unbeatable.

---

## 1. Heavily degraded images

**Failure mode:** CDN/social-media recompression, thumbnails, screenshots, and
crops destroy the high-frequency forensic signal detectors rely on. A model
that scores 95% on clean val data can drop 20–40 points on twice-recompressed
images. This is the #1 production gap because it affects *most* images a
Chrome-extension user scans.

### v1 — training-time degradation (the core fix)

Build `scripts/degrade.py` as a shared library used by BOTH the training
loader and the eval harness — never two implementations:

- **Degradation chains, not single ops.** Sample 0–3 ops per image: JPEG
  (q 40–95, including double-JPEG with mismatched quality and 4px grid
  misalignment), WebP recode, down/upscale (varied kernels: bilinear, bicubic,
  Lanczos), Gaussian blur/noise, chroma subsampling, mild sharpening.
- **Named platform presets** that mirror real pipelines, e.g.
  `twitter` (max 2048px, JPEG ~q85), `instagram` (1080px, ~q80),
  `whatsapp` (~q70 + resize), `discord_thumb`, `screenshot` (scale to a
  viewport fraction at 1x/2x DPR, then PNG→JPEG). Presets are used verbatim in
  the eval ladder so "TPR under the Twitter preset" is a tracked number.
- **Applied identically to both classes** (non-negotiable, see CLAUDE.md Data
  Policy) — otherwise compression history becomes the label again.
- **Consistency regularization** (cheap, high value): each batch, run a clean
  view and a degraded view of the same image; add a KL term pulling their
  logits together. This teaches degradation-*invariant* features instead of
  just exposing the model to noise.
- **Screenshot variant set:** Playwright script renders a few hundred train/val
  images in a browser at assorted zoom/DPR and captures screenshots. Cheap to
  build, covers a pipeline JPEG augmentation does not reproduce.

### v1 — inference-time

- **Scan the best available copy.** The extension should prefer the
  least-degraded variant the page offers: `og:image`, the largest `srcset`
  entry, the original `src` over a lazy-loaded thumbnail. This is a pure
  product win — no ML — and probably worth more than any training trick.
  (Change lands in the detect repo, not here.)
- **Quality gating + honest abstention.** Estimate input quality before
  classifying: resolution, blockiness/JPEG quantization estimate, blur metric.
  Below a floor, widen the uncertain band and say so in `signals`
  (e.g., `"low_quality_input"`), reducing displayed confidence. Converts
  silent failure into honest UX. Floors are chosen from the eval ladder, not
  guessed.
- **Light test-time augmentation:** average logits over 3–5 native-resolution
  crops + one global resized view. ViT-S int8 at ~100ms/pass keeps this inside
  the 2s budget. Native-res crops matter: the forensic signal lives in
  original pixels, and always-downsampling throws it away.

### Eval & targets

`benchmark_baselines.py` already runs a small ladder (jpeg70, half-res). Extend
it with the platform presets and report an accuracy-vs-severity curve per
model. Suggested v1 targets:

| Condition | Target |
|---|---|
| Clean | TPR ≥ 90%, FPR ≤ 2% on hard negatives |
| `twitter` / `instagram` preset | TPR ≥ 80%, FPR ≤ 3% |
| `screenshot` of a preset | TPR ≥ 65%, FPR ≤ 4% |
| Below quality floor | model abstains (uncertain verdict), never high-confidence |

If the screenshot row can't be met, that's what the abstention path is for —
do not ship high-confidence verdicts on inputs the eval says we can't read.

---

## 2. Partial AI (inpainting, generative fill, AI upscaling)

**Failure mode:** an image that is 90% real photo with a generative-fill sky,
an inpainted object, or AI upscaling. A global binary classifier produces
meaningless mid-range scores; worse, the training labels don't even define
what the right answer is.

### Policy first (blocks everything else)

Define the taxonomy and the verdict each class maps to. Proposed:

| Class | Example | v1 verdict |
|---|---|---|
| (a) Fully generated | Flux/MJ/gpt-image output | **AI** |
| (b) AI-edited real | generative fill, object insert/remove, sky swap | **uncertain** in v1 → **"AI-edited"** in v1.5 |
| (c) AI-enhanced real | upscalers, denoise, phone computational photography | **real** |
| (d) Real | straight photo | **real** |

Class (c) must map to *real*: every modern phone photo is AI-processed, and
flagging upscales would recreate the false-positive problem we exist to fix.
Training labels follow this table — ambiguous (b) images are *excluded* from
v1 binary training, not guessed at.

### v1 — cheap moves that ship with the first model

- **Upscaled reals as real-class augmentation.** Run Real-ESRGAN (and one or
  two other upscalers) on a few thousand of our real photos and add them to
  the real class. Without this, AI upscales of real photos carry
  decoder fingerprints and become an FP storm. This is one script and a
  weekend of 4090 time. (Counterpart of policy class (c).)
- **C2PA does the heavy lifting for (b).** Adobe Firefly/Photoshop generative
  fill — the most common real-world partial-AI source — writes C2PA manifests
  with edit assertions. The extension already treats C2PA as an override;
  extend the reader to surface "edited with AI tools" assertions as an
  `AI-edited` signal. Zero ML, covers the most common case. (detect repo.)
- **Patch-grid scoring at inference** (also serves degradation TTA): run the
  classifier on a 3×3 overlapping crop grid at native resolution plus the
  global view. Fully generated → most patches hot; inpainted → localized hot
  patches; real → uniformly cold. Heuristic verdict: `max_patch ≫ global`
  → "AI-edited (region)". 10 ViT-S int8 passes ≈ ~1s — inside budget. Free
  localization payload for `signals`.

### v1.5 — train for it

- **Generate our own inpainting dataset:** take our real photos, auto-mask
  regions (segmentation: sky, faces, objects), inpaint with SDXL/Flux-fill on
  the 4090, keep the mask as ground truth. Few thousand images, one script
  (`collect_partial_ai.py`), same no-re-encode policy.
- Supplement with public sets for eval diversity: AutoSplice, MagicBrush,
  TGIF-style text-guided forgery sets — use as *eval*, train on our own
  generation to avoid license/leakage questions.
- Either add a third class ("AI-edited") to the classifier head, or keep the
  binary head and train a small patch-level head — decide from how well the
  v1 patch-grid heuristic performs on the generated eval set.

### v2 — only if demand proves out

Proper manipulation-localization models (TruFor-class, noiseprint-based) are
the real solution but don't fit 2GB/2s today. Revisit if "AI-edited" becomes a
headline product feature.

### Eval & targets

New eval dir: `data/val_partial/` (our generated inpaints + public-set
samples, frozen like the rest). v1 target is modest and honest: fully-generated
TPR unaffected (≥90%), inpainted images land in *uncertain or AI-edited* ≥60%
of the time, and — most important — **upscaled real photos FP ≤ 2%**.

---

## 3. Adversarial robustness

**Threat model first.** Three tiers:

1. **Casual** — filters, slight crop, screenshot, re-save. Overlaps ~entirely
   with the degradation plan; tier 1 is *defeated* by Section 1 work.
2. **Tool-assisted** — "AI detector bypass" / "humanizer" services adding
   adversarial noise or style perturbation, tuned against public open-source
   detectors.
3. **Targeted** — an attacker probing *our* API specifically with
   query-feedback attacks.

Realistic goal: fully handle tier 1, materially raise cost for tier 2, accept
tier 3 (so does Hive). Never claim certainty — C2PA positive provenance is the
only "certain" signal in the product.

### v1

- **Adversarial training, lite.** FGSM/small-ε PGD on a fraction (~10–25%) of
  training batches. Cheap for ViT-S, and small-Lp noise is exactly what
  commodity bypass tools sell. Keep ε small to protect clean accuracy; track
  the tradeoff in eval.
- **Input randomization at inference** — already getting it for free: the
  multi-crop TTA from Section 1 (random crop offsets + resize jitter, averaged
  logits) breaks the exact-gradient assumption of transferred perturbations.
- **Heterogeneous ensemble.** Keep NYUAD as second opinion (already planned)
  and add a near-free frequency-domain feature (FFT/DCT high-band energy
  statistic) as a third signal into `flagCount`/`signals`. Adversarial
  perturbations transfer poorly across architecturally different models;
  requiring agreement raises evasion cost. (The API shape already supports
  multi-signal — no contract change.)
- **Don't leak the gradient through the product.** Score-feedback attacks need
  many precise queries. Externally: coarse verdict buckets + banded
  confidence, not raw float scores; per-account scan caps already exist
  (25/day Pro) — keep them as a security control, not just a pricing one.
  Internally: log full-precision scores.

### v1.5 — detection of evasion, not just resistance

- **Probing/drift monitoring:** log score distributions per deployment. A
  cluster of near-threshold scores from one account or one image family =
  probing or a bypass tool in the wild. Alert, capture (within privacy
  policy), and feed those images into the hard-case pool for the next retrain.
- **Bypass-tool red team:** once per retrain cycle, run the current popular
  bypass tools/services against our val_ai set, measure TPR drop, add the
  outputs to training. This is the same flywheel as new-generator coverage —
  treat "Flux 2 released" and "new bypass tool released" as the same event
  class.

### Eval & targets

Add an adversarial row to the eval harness: FGSM/PGD at 2–3 ε values
(white-box on our own model = worst case), plus outputs of at least one public
bypass tool. Targets: white-box small-ε TPR ≥ 60% with adversarial training
(vs ~0% without — that's the honest baseline); bypass-tool TPR within 15
points of clean TPR. UX rule regardless of numbers: verdict language stays
probabilistic ("likely AI", "AI signals detected"), absolutes are reserved for
C2PA provenance.

---

## Cross-cutting build list (what this adds to the roadmap)

| Item | Phase | Where |
|---|---|---|
| `scripts/degrade.py` — shared chains + platform presets | v1 | this repo |
| Consistency loss + degradation aug in `train.py` | v1 | this repo |
| Quality gating + abstention floor | v1 | detect repo (`space/app.py`) |
| Patch-grid TTA + localization heuristic | v1 | detect repo (`space/app.py`) |
| Best-available-copy fetching (og:image / srcset) | v1 | detect repo (extension) |
| Upscaled-reals augmentation script | v1 | this repo |
| C2PA "AI-edited" assertion surfacing | v1 | detect repo |
| FGSM/PGD-lite in training | v1 | this repo |
| Screenshot variant set (Playwright) | v1 | this repo |
| `collect_partial_ai.py` + `data/val_partial/` | v1.5 | this repo |
| "AI-edited" class or patch head | v1.5 | this repo |
| Score-drift / probing monitoring | v1.5 | detect repo (backend) |
| Bypass-tool red team per retrain | v1.5 | process |
| Manipulation localization model | v2 | maybe never |
