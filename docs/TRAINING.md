# TRAINING.md — fine-tune runbook (written for a Claude Code session on the training box)

This is the operational guide for running and monitoring the v1 fine-tune. Read
`CLAUDE.md` first for project context. As of 11 Jun 2026 all pre-training gates are
complete; nothing here requires re-running collection, splits, probes, or baselines.

**The one-line mission:** fine-tune `OwensLab/commfor-model-224` so hard-negative FPR
drops from 7.4%/3.4% (clean/degraded) to ≤2% on both, without losing its 95%+ degraded
TPR. False positives on real photos are the worse failure mode — always.

---

## 0. Quick reference

| Thing | Value |
|---|---|
| Launch | `bash scripts/train_4090.sh` (does setup + sanity checks + nohup launch) |
| Manual launch | `venv/bin/python scripts/train.py --epochs 8 --batch-size 64 --base OwensLab/commfor-model-224 --num-workers 12` |
| Log | `logs/train_<timestamp>.log` (stdout) + `models/<timestamp>/log.jsonl` (per-epoch JSON) |
| Output | `models/<timestamp>/best/` + `inference_config.json` + `metrics.json` |
| Expected duration | ~1–3 h on a 4090 (8 epochs, ~60k images/epoch; CPU decode is the bottleneck) |
| Dataset | 58GB in `data/` — rsync from the Mac, see `scripts/train_4090.sh` header |
| Targets | hard-neg FPR ≤ 2% clean AND jpeg70_half; val_ai TPR ≥ 90% clean / ≥ 80% jpeg70_half; report pinterest_feed TPR |
| Bar to beat | `data/validation_report/baselines.json` (committed in repo) — the commfor-224 rows |

## 1. Pre-flight (the launch script does all of this; verify if launching manually)

1. `nvidia-smi` shows the 4090 with ≥20GB free; `torch.cuda.is_available()` is True.
2. ≥70GB free disk (58GB data + checkpoints + headroom).
3. Dataset counts (a partial rsync is the most likely silent failure):
   `real ≥ 20,000 · ai ≥ 35,000 · val_real ≥ 2,200 · val_ai ≥ 4,300 ·
   hard_negatives ≥ 170 · hard_positives ≥ 100`
4. Do NOT re-run `split_val.py` or `leakage_probe.py` — the val split is frozen and the
   probe was run + accepted 11 Jun 2026 (see CLAUDE.md gate results).
5. `data/real_legacy_384/` must not be present on the box (excluded from rsync). If it
   is, that's fine — train.py never reads it — but never add it to `data/real/`.

## 2. What the training run does (so you can judge the log)

`scripts/train.py`:
- Converts the timm-format commfor checkpoint to transformers ViT at load
  (verified numerically exact). First log lines show the data counts and device.
- Each batch: one clean view + one degraded view (shared random crop, never squashed),
  cross-entropy on both + KL consistency loss (degraded should predict like clean).
  ~15% of batches add an FGSM adversarial term. Class-balanced sampling.
- Per epoch: evaluates val AUC/FPR/TPR and `hard_neg_fpr@0.5`, appends a JSON line to
  `models/<ts>/log.jsonl`, saves to `best/` when val AUC improves.
- After the last epoch: reloads `best/`, fits temperature scaling, picks the operating
  threshold to satisfy `--target-fpr 0.02` on val_real AND hard_negatives, writes
  `inference_config.json` (temperature, threshold_ai, uncertain_band) + `metrics.json`.

## 3. Monitoring — what to check, when, and what's normal

A sensible cadence for an agent: check ~10 min after launch (catch early crashes), then
every 20–30 min, then at expected-completion time.

```bash
tail -20 logs/train_*.log                  # latest progress
grep '^epoch' logs/train_*.log             # one line per finished epoch
cat models/*/log.jsonl | python -m json.tool --json-lines 2>/dev/null | tail -40
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5  # spot-check
```

Healthy run:
- Step loss starts ~0.3–0.7 (the base is already good) and falls; no NaN/inf.
- Epoch time roughly constant (first epoch can be slower — HF download + warm cache).
- `val auc` ≥ 0.99 early and stays there (the base starts near-perfect on val;
  the interesting movement is hard_neg_fpr, not AUC).
- `hard_neg_fpr@0.5` trends down across epochs toward ≤ 0.02–0.04. Non-monotonic is
  normal; a sustained climb above the epoch-0 value is not.
- GPU util 50–90% with bursts; if it sits <30%, the dataloader is starving the GPU
  (see troubleshooting).

When to intervene vs. leave it alone:
- **Leave alone:** noisy loss, one bad epoch, val TPR dipping a point or two,
  GPU util fluctuating. The best-checkpoint logic and calibration handle this.
- **Kill and fix:** NaN loss, CUDA OOM loops, epoch time exploding, val AUC
  collapsing (<0.95) and staying down — that smells like an LR or data problem, not
  something more epochs fix. The run is cheap; diagnose, fix, relaunch.
- Training crashed mid-run? There is no resume — relaunch from scratch (it's ≤3h).
  `models/<ts>/best/` from a crashed run is still a valid model if ≥1 epoch finished.

## 4. Acceptance — judging the finished run

`metrics.json` has the headline numbers, but the binding criteria are:

1. `hard_neg_fpr@threshold` ≤ 0.02 (it's computed at the calibrated operating
   threshold, which is the number production will use).
2. `achieved_tpr` ≥ 0.90 at that threshold.
3. The uncertain band should be narrow. A band like [0.05, 0.95] means the model
   couldn't satisfy FPR and TPR targets simultaneously — that's a soft fail: usable
   in the cascade (uncertain → Hive), but report it prominently.
4. Then run the degradation eval — the per-epoch numbers are CLEAN only:

```bash
venv/bin/python scripts/benchmark_baselines.py --models models/<ts>/best
```

   This appends the trained model to `data/validation_report/baselines.json` next to
   the baselines (clean/jpeg70/half/jpeg70_half × val_real/val_ai/hard_neg/hard_pos).
   Compare directly against the `OwensLab/commfor-model-224` rows: we must be strictly
   better on hard_negatives at equal-or-better TPR, and we want hard_positives TPR up
   from 94% (those are in-the-wild misses, incl. Pinterest).
5. Pinterest check (the harshest condition; commfor-224 baseline: 97% val_ai / 84%
   hard_pos at 236px): score val_ai + hard_positives through
   `degrade.apply_preset(img, "pinterest_feed")` with the model and report TPR.
   The session that adds this as a `--degradations` option in benchmark_baselines.py
   should do so — it's a 5-line change, presets already exist in `scripts/degrade.py`.

Caveat to repeat in any report: ~40% of val_ai is CommunityForensics data — val_ai TPR
is partly in-distribution for the base AND our fine-tune. hard_positives and
hard_negatives (never trained on, frozen) are the honest numbers.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unrecognized model` loading base | running old code | `git pull` — conversion landed 11 Jun 2026 |
| CUDA OOM at bs64 | other procs on GPU | `--batch-size 32` (lr stays fine for ViT-S) |
| GPU util < 30%, epochs slow | CPU decode bottleneck | raise `--num-workers` toward #cores; check disk isn't NFS/spinning rust |
| DataLoader worker crashes | bad/truncated image | train.py skips unreadable files; repeated same-file crashes → delete that file, note it |
| NaN loss | LR too high for this base | relaunch `--lr 1e-5`; if stable but slow, try 2e-5 |
| val AUC ~0.5 entire first epoch | label/head wiring broke | stop; verify base-loader equivalence (CLAUDE.md gate results have the method) |
| Box dies / power loss | — | relaunch; no resume support, runs are ≤3h |

## 6. After acceptance — handoff back

1. Copy `models/<ts>/` (best/ + both JSONs) back to the Mac, or push the model dir to
   HF hub (private) if more convenient.
2. Update CLAUDE.md "Current Status" with the metrics table (clean + degraded +
   pinterest_feed) next to the baseline rows, and mark Step 3 ✅ DONE.
3. Commit `baselines.json` if the eval appended the trained model's rows.
4. Next phase is Step 4/5 in CLAUDE.md: leave-one-generator-out eval, Hive comparison,
   ONNX int8 export, Render benchmark, integration into the detect repo's
   `space/app.py` (and remove Ateeqq from the ensemble — decided, see CLAUDE.md).
