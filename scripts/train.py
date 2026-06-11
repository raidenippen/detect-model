"""
Fine-tune a ViT binary AI-image detector. Implements the v1 training items
from docs/ROBUSTNESS.md and the CLAUDE.md data policy:

  - originals in, random CROP (never aspect squash) to 224
  - degradation chains (scripts/degrade.py) applied identically to both classes
  - consistency loss: KL between clean-view and degraded-view logits, so the
    model learns degradation-INVARIANT features
  - FGSM-lite adversarial training on a fraction of batches
  - class-balanced sampling
  - post-training: temperature scaling (calibrated confidence) + operating
    threshold chosen for a target FPR on val_real + uncertain band
  - evaluates FPR on data/hard_negatives every epoch if present

Outputs to --out (default models/<timestamp>):
  best/                  HF save_pretrained model + image processor
  inference_config.json  { temperature, threshold_ai, uncertain_band, ... }
  metrics.json           final val/hard-negative metrics
  log.jsonl              per-epoch training log

Usage (4090 box or cloud GPU; CPU/MPS work but are slow):
  python train.py --epochs 8 --batch-size 64
  python train.py --base OwensLab/commfor-model-224       # Community Forensics
  python train.py --base WinKawaks/vit-small-patch16-224 --target-fpr 0.02

Run scripts/leakage_probe.py BEFORE burning GPU time — if real sources are
separable from low-level stats, fix the data first.
"""

import json
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

import degrade

ROOT = Path(__file__).parent.parent
EXTS = (".jpg", ".jpeg", ".png", ".webp")

# The Community Forensics checkpoints (the baseline-benchmark winners) are
# timm-format ViT-S/16 weights with a single sigmoid logit — not transformers
# format, so AutoModel can't load them. Convert to ViTForImageClassification at
# load time so the rest of the pipeline (save_pretrained, best-checkpoint
# reload, ONNX export) stays transformers-native. The 1-logit head z = w·x+b
# maps exactly onto a 2-class head [-z/2, +z/2]: softmax P(ai) == sigmoid(z),
# so fine-tuning starts from the checkpoint's exact decision function.
COMMFOR_BASES = {"OwensLab/commfor-model-224": 224, "OwensLab/commfor-model-384": 384}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_commfor_base(base):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    size = COMMFOR_BASES[base]
    src = load_file(hf_hub_download(base, "model.safetensors"))
    src = {k.removeprefix("vit."): v for k, v in src.items()}

    cfg = ViTConfig(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                    intermediate_size=1536, image_size=size, patch_size=16,
                    layer_norm_eps=1e-6, num_labels=2,
                    id2label={0: "real", 1: "ai"}, label2id={"real": 0, "ai": 1})
    model = ViTForImageClassification(cfg)

    dst = {
        "vit.embeddings.cls_token": src["cls_token"],
        "vit.embeddings.position_embeddings": src["pos_embed"],
        "vit.embeddings.patch_embeddings.projection.weight": src["patch_embed.proj.weight"],
        "vit.embeddings.patch_embeddings.projection.bias": src["patch_embed.proj.bias"],
        "vit.layernorm.weight": src["norm.weight"],
        "vit.layernorm.bias": src["norm.bias"],
        "classifier.weight": torch.cat([-src["head.weight"] / 2, src["head.weight"] / 2]),
        "classifier.bias": torch.cat([-src["head.bias"] / 2, src["head.bias"] / 2]),
    }
    for i in range(cfg.num_hidden_layers):
        t, h = f"blocks.{i}", f"vit.layers.{i}"
        qkv_w = src[f"{t}.attn.qkv.weight"].chunk(3)
        qkv_b = src[f"{t}.attn.qkv.bias"].chunk(3)
        for j, proj in enumerate(("q_proj", "k_proj", "v_proj")):
            dst[f"{h}.attention.{proj}.weight"] = qkv_w[j]
            dst[f"{h}.attention.{proj}.bias"] = qkv_b[j]
        dst[f"{h}.attention.o_proj.weight"] = src[f"{t}.attn.proj.weight"]
        dst[f"{h}.attention.o_proj.bias"] = src[f"{t}.attn.proj.bias"]
        dst[f"{h}.layernorm_before.weight"] = src[f"{t}.norm1.weight"]
        dst[f"{h}.layernorm_before.bias"] = src[f"{t}.norm1.bias"]
        dst[f"{h}.layernorm_after.weight"] = src[f"{t}.norm2.weight"]
        dst[f"{h}.layernorm_after.bias"] = src[f"{t}.norm2.bias"]
        dst[f"{h}.mlp.fc1.weight"] = src[f"{t}.mlp.fc1.weight"]
        dst[f"{h}.mlp.fc1.bias"] = src[f"{t}.mlp.fc1.bias"]
        dst[f"{h}.mlp.fc2.weight"] = src[f"{t}.mlp.fc2.weight"]
        dst[f"{h}.mlp.fc2.bias"] = src[f"{t}.mlp.fc2.bias"]
    model.load_state_dict(dst)

    processor = ViTImageProcessor(image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
                                  size={"height": size, "width": size})
    return model, processor


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_images(*dirs):
    files = []
    for d in dirs:
        if d.exists():
            files.extend(p for p in d.iterdir() if p.suffix.lower() in EXTS)
    return sorted(files)


class TrainSet(Dataset):
    """Returns (clean_view, degraded_view, label). Both views share the same
    random crop + flip so the consistency loss compares like with like; only
    the degradation differs."""

    def __init__(self, real_files, ai_files, image_size, mean, std, degrade_prob):
        self.items = [(p, 0) for p in real_files] + [(p, 1) for p in ai_files]
        self.size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.degrade_prob = degrade_prob

    def __len__(self):
        return len(self.items)

    def _to_tensor(self, img):
        t = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return (t - self.mean) / self.std

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return self[random.randrange(len(self))]

        # Randomized shorter-side downscale (never upscale). Randomized so a
        # source's native resolution can't become a deterministic signature.
        short = min(img.size)
        target_short = random.randint(384, 768)
        if short > target_short:
            scale = target_short / short
            img = img.resize((max(1, round(img.width * scale)),
                              max(1, round(img.height * scale))), Image.BICUBIC)

        # Shared random crop -> size x size (CROP, never squash)
        area = img.width * img.height
        for _ in range(10):
            t_area = area * random.uniform(0.5, 1.0)
            ar = math.exp(random.uniform(math.log(0.8), math.log(1.25)))
            cw, ch = round(math.sqrt(t_area * ar)), round(math.sqrt(t_area / ar))
            if cw <= img.width and ch <= img.height:
                x = random.randint(0, img.width - cw)
                y = random.randint(0, img.height - ch)
                img = img.crop((x, y, x + cw, y + ch))
                break
        base = img.resize((self.size, self.size), Image.BICUBIC)
        if random.random() < 0.5:
            base = base.transpose(Image.FLIP_LEFT_RIGHT)

        deg = degrade.random_chain(base, random) if random.random() < self.degrade_prob else base
        if deg.size != base.size:  # some ops (double-JPEG grid shift) change size
            deg = deg.resize(base.size, Image.BICUBIC)
        return self._to_tensor(base), self._to_tensor(deg), label


class EvalSet(Dataset):
    """Deterministic: shorter side -> 256, center crop -> size."""

    def __init__(self, files, labels, image_size, mean, std):
        self.files = files
        self.labels = labels
        self.size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.size, self.size))
        scale = 256 / min(img.size)
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))), Image.BICUBIC)
        x = (img.width - self.size) // 2
        y = (img.height - self.size) // 2
        img = img.crop((x, y, x + self.size, y + self.size))
        t = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        t = (t - self.mean) / self.std
        return t, self.labels[idx]


def worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % 2**32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_score(scores, labels):
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    # tie-averaged ranks
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    start = np.searchsorted(np.sort(allv), uniq, side="left") + 1
    ranks = (start + (counts - 1) / 2.0)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        out = model(pixel_values=x.to(device)).logits
        logits_all.append(out.float().cpu())
        labels_all.append(y)
    return torch.cat(logits_all), torch.cat(labels_all)


def summarize(logits, labels, temperature=1.0):
    scores = F.softmax(logits / temperature, dim=1)[:, 1].numpy()
    labels = labels.numpy()
    real, ai = scores[labels == 0], scores[labels == 1]
    out = {"auc": round(auc_score(scores, labels), 4),
           "acc@0.5": round(float(((scores >= 0.5) == labels).mean()), 4)}
    if len(real):
        out["fpr@0.5"] = round(float((real >= 0.5).mean()), 4)
    if len(ai):
        out["tpr@0.5"] = round(float((ai >= 0.5).mean()), 4)
    return out, scores, labels


# ---------------------------------------------------------------------------
# Calibration + operating point
# ---------------------------------------------------------------------------

def fit_temperature(logits, labels):
    # Bounds keep a degenerate (perfectly separated) val set from collapsing
    # T to ~0 and saturating every score to 0/1.
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(0.25, 10.0), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(0.25, 10.0))


def _thr_for_fpr(real_scores, target_fpr):
    # Count-based, not quantile interpolation: the threshold must sit strictly
    # ABOVE the k-th highest real score, otherwise ties at the score mass blow
    # straight through the FPR target.
    k = int(np.floor(target_fpr * len(real_scores)))  # number of FPs we may admit
    desc = np.sort(real_scores)[::-1]
    return min(1.0, float(desc[k]) + 1e-6) if k < len(desc) else 0.0


def pick_operating_point(scores, labels, target_fpr, target_tpr=0.90,
                         hard_neg_scores=None):
    real, ai = scores[labels == 0], scores[labels == 1]
    if len(real):
        thr_fpr = _thr_for_fpr(real, target_fpr)
        # The product requirement is FPR <= target on the HARD cases, not just
        # the average val real — the threshold must satisfy both sets.
        if hard_neg_scores is not None and len(hard_neg_scores):
            thr_fpr = max(thr_fpr, _thr_for_fpr(np.asarray(hard_neg_scores), target_fpr))
    else:
        thr_fpr = 0.5
    if len(ai):
        asc = np.sort(ai)
        j = int(np.floor((1.0 - target_tpr) * len(ai)))
        thr_tpr = float(asc[min(j, len(asc) - 1)])  # >= this keeps TPR >= target
    else:
        thr_tpr = 0.5
    lo, hi = sorted([thr_tpr, thr_fpr])
    return {
        "threshold_ai": round(thr_fpr, 6),          # score >= this -> verdict AI
        "uncertain_band": [round(lo, 6), round(hi, 6)],  # between -> uncertain
        "achieved_fpr": round(float((real >= thr_fpr).mean()), 4) if len(real) else None,
        "achieved_tpr": round(float((ai >= thr_fpr).mean()), 4) if len(ai) else None,
        "target_fpr": target_fpr,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--base", default="WinKawaks/vit-small-patch16-224",
                   help="Any HF image-classification base (e.g. a Community Forensics checkpoint)")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--degrade-prob", type=float, default=0.7)
    p.add_argument("--consistency-weight", type=float, default=1.0)
    p.add_argument("--fgsm-frac", type=float, default=0.15,
                   help="Fraction of batches that get an FGSM adversarial term (0 disables)")
    p.add_argument("--fgsm-eps", type=float, default=2 / 255)
    p.add_argument("--target-fpr", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = device == "cuda"
    data = Path(args.data_root)
    out_dir = Path(args.out) if args.out else ROOT / "models" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train_real = list_images(data / "real", data / "real_upscaled")
    train_ai   = list_images(data / "ai")
    val_real   = list_images(data / "val_real")
    val_ai     = list_images(data / "val_ai")
    hard_neg   = list_images(data / "hard_negatives")

    print(f"train: {len(train_real)} real / {len(train_ai)} ai   "
          f"val: {len(val_real)} real / {len(val_ai)} ai   hard_neg: {len(hard_neg)}")
    if min(len(train_real), len(train_ai), len(val_real), len(val_ai)) == 0:
        raise SystemExit("Empty data split — run collection and split_val.py first.")
    if not hard_neg:
        print("WARNING: data/hard_negatives is empty — the metric this project "
              "exists to improve will not be measured.")

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    if args.base in COMMFOR_BASES:
        if args.image_size != COMMFOR_BASES[args.base]:
            raise SystemExit(f"{args.base} has fixed position embeddings for "
                             f"--image-size {COMMFOR_BASES[args.base]}")
        model, processor = load_commfor_base(args.base)
    else:
        processor = AutoImageProcessor.from_pretrained(args.base)
        model = AutoModelForImageClassification.from_pretrained(
            args.base, num_labels=2, id2label={0: "real", 1: "ai"},
            label2id={"real": 0, "ai": 1}, ignore_mismatched_sizes=True,
        )
    model = model.to(device)
    mean, std = processor.image_mean, processor.image_std

    train_set = TrainSet(train_real, train_ai, args.image_size, mean, std, args.degrade_prob)
    # class-balanced sampling
    weights = ([1.0 / len(train_real)] * len(train_real) + [1.0 / len(train_ai)] * len(train_ai))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"),
                              worker_init_fn=worker_init, drop_last=True)

    val_files = val_real + val_ai
    val_labels = [0] * len(val_real) + [1] * len(val_ai)
    val_loader = DataLoader(EvalSet(val_files, val_labels, args.image_size, mean, std),
                            batch_size=args.batch_size, num_workers=args.num_workers)
    hard_loader = (DataLoader(EvalSet(hard_neg, [0] * len(hard_neg), args.image_size, mean, std),
                              batch_size=args.batch_size, num_workers=args.num_workers)
                   if hard_neg else None)

    # ---- optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)
    eps_t = args.fgsm_eps / std_t
    clamp_lo, clamp_hi = (0 - mean_t) / std_t, (1 - mean_t) / std_t

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log_f = open(out_dir / "log.jsonl", "a")
    best_auc, best_dir = -1.0, out_dir / "best"

    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), []
        for step, (xc, xd, y) in enumerate(train_loader):
            xc, xd, y = xc.to(device), xd.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits_c = model(pixel_values=xc).logits
                logits_d = model(pixel_values=xd).logits
                loss_ce = 0.5 * (F.cross_entropy(logits_c, y) + F.cross_entropy(logits_d, y))
                # degraded view should predict like the clean view (stop-grad on clean)
                loss_kl = F.kl_div(F.log_softmax(logits_d, dim=1),
                                   F.softmax(logits_c.detach(), dim=1),
                                   reduction="batchmean")
                loss = loss_ce + args.consistency_weight * loss_kl

            loss_adv = None
            if args.fgsm_frac > 0 and random.random() < args.fgsm_frac:
                xd_adv = xd.detach().clone().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    adv_ce = F.cross_entropy(model(pixel_values=xd_adv).logits, y)
                grad = torch.autograd.grad(adv_ce, xd_adv)[0]
                x_adv = (xd_adv + eps_t * grad.sign()).clamp(clamp_lo, clamp_hi).detach()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    loss_adv = F.cross_entropy(model(pixel_values=x_adv).logits, y)
                loss = loss + 0.5 * loss_adv

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(float(loss.detach()))
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {np.mean(running[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e}")

        # ---- eval ----
        logits, labels = collect_logits(model, val_loader, device)
        val_metrics, _, _ = summarize(logits, labels)
        entry = {"epoch": epoch, "train_loss": round(float(np.mean(running)), 4),
                 "val": val_metrics, "secs": round(time.time() - t0)}
        if hard_loader is not None:
            h_logits, h_labels = collect_logits(model, hard_loader, device)
            h_scores = F.softmax(h_logits, dim=1)[:, 1].numpy()
            entry["hard_neg_fpr@0.5"] = round(float((h_scores >= 0.5).mean()), 4)
            entry["hard_neg_mean_score"] = round(float(h_scores.mean()), 4)
        print(f"epoch {epoch}: {entry}")
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  new best (auc {best_auc}) -> {best_dir}")

    log_f.close()

    # ---- calibration + operating point on the best checkpoint ----
    print("\nCalibrating best checkpoint...")
    model = AutoModelForImageClassification.from_pretrained(best_dir).to(device)
    logits, labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    metrics, scores, labels_np = summarize(logits, labels, temperature)

    h_scores = None
    if hard_neg:
        h_logits, _ = collect_logits(model, hard_loader, device)
        h_scores = F.softmax(h_logits / temperature, dim=1)[:, 1].numpy()
    op = pick_operating_point(scores, labels_np, args.target_fpr,
                              hard_neg_scores=h_scores)

    final = {"base": args.base, "val": metrics, "temperature": round(temperature, 4), **op}
    if h_scores is not None:
        final["hard_neg_fpr@threshold"] = round(float((h_scores >= op["threshold_ai"]).mean()), 4)
        final["hard_neg_fpr@0.5"] = round(float((h_scores >= 0.5).mean()), 4)

    (out_dir / "inference_config.json").write_text(json.dumps(
        {"temperature": final["temperature"], "threshold_ai": op["threshold_ai"],
         "uncertain_band": op["uncertain_band"], "image_size": args.image_size,
         "labels": {"0": "real", "1": "ai"}}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))

    print(json.dumps(final, indent=2))
    print(f"\nDone. Model: {best_dir}")
    print("Next: run the eval ladder (benchmark_baselines.py --models <best_dir>) "
          "and compare against baselines.json before integrating.")


if __name__ == "__main__":
    main()
