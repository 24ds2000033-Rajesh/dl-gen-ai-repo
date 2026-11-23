# =====================================================================
# SurfaceGuard AI - Industrial Surface Anomaly Segmentation (optimized)
# 5-fold CV | UNet + scSE + classification head | EMA | per-image Dice loss
# OOF-tuned (pixel-threshold, min-area, class-gate) | 4-view TTA | fold ensemble
# Single script: run top to bottom in a Kaggle notebook (GPU on).
# =====================================================================
import os, sys, gc, time, copy, random, subprocess
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)
    import segmentation_models_pytorch as smp

import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
import albumentations as A
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

# ----------------------------- CONFIG --------------------------------
BASE = Path("/kaggle/input/competitions/industrial-surface-defect-detection/public")
WORK = Path("/kaggle/working")
ENCODER = "tu-efficientnet_b3"        # try "tu-efficientnet_b4" if time allows
H, W = 128, 800
BS, EPOCHS, LR, WD = 8, 8, 4e-4, 1e-4
N_FOLDS = 5
RUN_FOLDS = [0, 1, 2, 3, 4]           # use [0] for a quick sanity run
EMA_DECAY = 0.998
SEED = 42
NUM_WORKERS = min(4, os.cpu_count() or 2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
USE_WANDB = True
ENTITY = "24ds2000033-iitm-bs"
PROJECT = "24DS2000033-t32026"

# tuning grids (done on out-of-fold predictions, using cheap count arithmetic)
THRS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
MIN_PIX = [0, 50, 100, 200, 400, 800, 1600]
GATES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FLIPS = [(), (-1,), (-2,), (-1, -2)]

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(SEED)

# ------------------------------ W&B ----------------------------------
run = None
if USE_WANDB:
    try:
        import wandb
        key = None
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception as e:
            print("No WANDB key:", e)
        run = wandb.init(entity=ENTITY, project=PROJECT, mode="online" if key else "offline",
                         name=f"unet-scse-{ENCODER}-cv{N_FOLDS}-ep{EPOCHS}",
                         config=dict(encoder=ENCODER, epochs=EPOCHS, bs=BS, lr=LR, folds=N_FOLDS,
                                     loss="BCE+perimage Dice+cls BCE", ema=EMA_DECAY),
                         settings=wandb.Settings(api_key=key, x_disable_stats=True))
    except Exception as e:
        print("W&B disabled:", e); run = None
def wlog(d):
    if run is not None:
        try: run.log(d)
        except Exception: pass

# ------------------------------ DATA ---------------------------------
df = pd.read_csv(BASE / "train.csv")
ss = pd.read_csv(BASE / "sample_submission.csv")
TRAIN_DIR, TEST_DIR = BASE / "train", BASE / "test"
CLASSES = sorted(df["defect_type"].unique().tolist())
C = len(CLASSES)
df["present"] = df["mask_rle"].fillna("").astype(str).str.strip() != ""

def rle_decode(rle):
    flat = np.zeros(H * W, np.uint8)
    if rle is not None and not pd.isna(rle) and str(rle).strip():
        t = np.array(str(rle).split(), dtype=int)
        for s, l in zip(t[0::2] - 1, t[1::2]):
            flat[s:s + l] = 1
    return flat.reshape((H, W), order="F")

def rle_encode(mask):
    px = np.concatenate([[0], mask.astype(np.uint8).flatten(order="F"), [0]])
    runs = np.where(px[1:] != px[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs)) or " "

# image x class presence matrix + RLE lookup
pres = (df.pivot_table(index="image_id", columns="defect_type", values="present", aggfunc="max")
          .reindex(columns=CLASSES).fillna(False).astype(int))
ALL_IDS = pres.index.to_numpy()
RLES = {(r.image_id, r.defect_type): r.mask_rle for r in df[df.present].itertuples()}
print("images:", len(ALL_IDS), "| positives per class:", dict(zip(CLASSES, pres.sum().tolist())))

AUG = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.4),
])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SegDS(Dataset):
    def __init__(self, ids, img_dir, labeled=True, train=False):
        self.ids, self.dir, self.labeled, self.train = list(ids), img_dir, labeled, train
    def __len__(self): return len(self.ids)
    def read(self, i):
        im = cv2.imread(f"{self.dir}/{i}.jpg")
        if im is None: raise FileNotFoundError(i)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    def __getitem__(self, k):
        i = self.ids[k]; img = self.read(i)
        masks = (np.stack([rle_decode(RLES.get((i, c))) for c in CLASSES], -1)
                 if self.labeled else np.zeros((H, W, C), np.uint8))
        if self.train:
            o = AUG(image=img, mask=masks); img, masks = o["image"], o["mask"]
        img = (torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1) - MEAN) / STD
        masks = torch.from_numpy(np.ascontiguousarray(masks)).permute(2, 0, 1).float()
        return img, masks

def make_loader(ds, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(ds, batch_size=BS, shuffle=shuffle, sampler=sampler, drop_last=drop_last,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)

# ------------------------- MODEL / LOSS / EMA ------------------------
def build_model():
    return smp.Unet(ENCODER, encoder_weights="imagenet", classes=C,
                    decoder_attention_type="scse",
                    aux_params=dict(classes=C, dropout=0.2)).to(DEV)

def loss_fn(logits, cls_logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter, tot = (p * y).sum((2, 3)), p.sum((2, 3)) + y.sum((2, 3))
    dice = 1 - (2 * inter + 1e-3) / (tot + 1e-3)                    # per image AND class (matches metric)
    pos = (y.sum((2, 3)) > 0).float()
    dice = (dice * pos).sum() / pos.sum().clamp(min=1)
    fp = (p.mean((2, 3)) * (1 - pos)).sum() / (1 - pos).sum().clamp(min=1)   # stray pixels on empty pairs
    cls = F.binary_cross_entropy_with_logits(cls_logits, pos)
    return bce + dice + 0.5 * fp + 0.5 * cls

class EMA:
    def __init__(self, model, decay):
        self.m, self.decay, self.n = copy.deepcopy(model).eval(), decay, 0
        for p in self.m.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        self.n += 1; d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(d).add_(p.detach(), alpha=1 - d)
            else: e.copy_(p)

# ------------------------- INFERENCE / METRIC ------------------------
@torch.no_grad()
def infer(models, x, n_views=4):
    P, Cl, n = 0, 0, 0
    for m in models:
        m.eval()
        for dims in FLIPS[:n_views]:
            xf = x.flip(dims) if dims else x
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                lo, cl = m(xf)
            p = torch.sigmoid(lo.float())
            P = P + (p.flip(dims) if dims else p); Cl = Cl + torch.sigmoid(cl.float()); n += 1
    return P / n, Cl / n

def score_counts(ps, gs, inter):
    """Local proxy of competition score from per-(image,class) pixel counts."""
    pos, neg = gs > 0, gs == 0
    dice = (2 * inter[pos] / (ps[pos] + gs[pos])).mean() if pos.any() else 1.0
    fp = (1 - ps[neg] / (H * W)).mean() if neg.any() else 1.0
    emp = (ps[neg] == 0).mean() if neg.any() else 1.0
    return 0.7 * dice + 0.2 * fp + 0.1 * emp, (float(dice), float(fp), float(emp))

@torch.no_grad()
def quick_val(model, dl, thr=0.5, gate=0.5):
    PS, GS, IN = [], [], []
    for x, y in dl:
        x, y = x.to(DEV), (y > 0.5).to(DEV)
        p, c = infer([model], x, n_views=1)
        m = (p > thr) & (c > gate)[:, :, None, None]
        PS.append(m.sum((2, 3)).cpu()); GS.append(y.sum((2, 3)).cpu()); IN.append((m & y).sum((2, 3)).cpu())
    ps, gs, it = [torch.cat(a).numpy().ravel().astype(np.float64) for a in (PS, GS, IN)]
    return score_counts(ps, gs, it)

@torch.no_grad()
def collect_stats(models, dl):
    """Per-threshold pixel counts on a validation set (TTA) -> tiny arrays instead of full prob maps."""
    PS = [[] for _ in THRS]; IN = [[] for _ in THRS]; GS, CL = [], []
    for x, y in tqdm(dl, desc="OOF stats", leave=False):
        x, y = x.to(DEV), (y > 0.5).to(DEV)
        p, c = infer(models, x, 4)
        GS.append(y.sum((2, 3)).cpu()); CL.append(c.cpu())
        for i, t in enumerate(THRS):
            m = p > t
            PS[i].append(m.sum((2, 3)).cpu()); IN[i].append((m & y).sum((2, 3)).cpu())
    f = lambda L: torch.cat(L).numpy()
    return (np.stack([f(a) for a in PS]), np.stack([f(a) for a in IN]), f(GS), f(CL))

# ----------------------------- TRAINING ------------------------------
def train_fold(k, tr_ids, va_ids):
    ck, st = WORK / f"fold{k}.pth", WORK / f"oof_fold{k}.npz"
    if ck.exists() and st.exists():
        print(f"[fold {k}] found checkpoint, skipping training"); return
    tr_ds, va_ds = SegDS(tr_ids, TRAIN_DIR, train=True), SegDS(va_ids, TRAIN_DIR)
    pr = pres.loc[tr_ids].to_numpy()
    freq = np.maximum(pr.mean(0), 1e-3)
    inv = np.clip(np.sqrt(freq.max() / freq), 1, 4)               # oversample rare classes (generic, not hard-coded)
    w = np.where(pr.any(1), (pr * inv).max(1), 1.0)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    tdl, vdl = make_loader(tr_ds, sampler=sampler, drop_last=True), make_loader(va_ds)

    model = build_model(); ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(tdl), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    best = -1

    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0
        for x, y in tqdm(tdl, desc=f"fold{k} ep{ep}/{EPOCHS}", leave=False):
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                lo, cl = model(x)
            loss = loss_fn(lo.float(), cl.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sch.step(); ema.update(model)
            tot += loss.item()
        s, (d, fp, emp) = quick_val(ema.m, vdl)
        print(f"[fold {k}] ep{ep} loss={tot/len(tdl):.4f} val={s:.4f} dice={d:.4f} fp={fp:.4f} empty={emp:.4f}")
        wlog({f"f{k}/loss": tot / len(tdl), f"f{k}/val": s, f"f{k}/dice": d, f"f{k}/fp": fp, f"f{k}/empty": emp})
        if s > best:
            best = s; torch.save(ema.m.state_dict(), ck)
    # out-of-fold statistics from best EMA weights
    ema.m.load_state_dict(torch.load(ck, map_location=DEV))
    ps, it, gs, cl = collect_stats([ema.m], vdl)
    np.savez_compressed(st, ps=ps, inter=it, gs=gs, cls=cl)
    print(f"[fold {k}] best quick val = {best:.4f}")
    del model, ema, opt, tdl, vdl; gc.collect(); torch.cuda.empty_cache()

combo = pres.astype(str).agg("".join, axis=1)
combo = combo.where(combo.map(combo.value_counts()) >= N_FOLDS, combo.value_counts().index[0])
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(ALL_IDS, combo))
for k in RUN_FOLDS:
    tr_i, va_i = folds[k]
    train_fold(k, ALL_IDS[tr_i], ALL_IDS[va_i])

# ------------------- TUNE POST-PROCESSING ON OOF ---------------------
S = [np.load(WORK / f"oof_fold{k}.npz") for k in RUN_FOLDS]
PS = np.concatenate([s["ps"] for s in S], 1).astype(np.float64)       # (T, N, C)
IN = np.concatenate([s["inter"] for s in S], 1).astype(np.float64)
GS = np.concatenate([s["gs"] for s in S], 0).astype(np.float64)       # (N, C)
CL = np.concatenate([s["cls"] for s in S], 0)

THR, MINPIX, GATE = [], [], []
for c in range(C):
    best = (-1, None)
    for ti, t in enumerate(THRS):
        for mp in MIN_PIX:
            for g in GATES:
                keep = (PS[ti, :, c] >= mp) & (CL[:, c] >= g)
                sc, _ = score_counts(PS[ti, :, c] * keep, GS[:, c], IN[ti, :, c] * keep)
                if sc > best[0]: best = (sc, (t, mp, g))
    THR.append(best[1][0]); MINPIX.append(best[1][1]); GATE.append(best[1][2])
    print(f"{CLASSES[c]}: score={best[0]:.4f} thr={best[1][0]} min_pix={best[1][1]} gate={best[1][2]}")

# pooled OOF score with the tuned settings
ps_all, in_all = np.zeros_like(GS), np.zeros_like(GS)
for c in range(C):
    ti = THRS.index(THR[c]); keep = (PS[ti, :, c] >= MINPIX[c]) & (CL[:, c] >= GATE[c])
    ps_all[:, c], in_all[:, c] = PS[ti, :, c] * keep, IN[ti, :, c] * keep
tuned, parts = score_counts(ps_all.ravel(), GS.ravel(), in_all.ravel())
print(f"\nTUNED OOF SCORE: {tuned:.4f}  (dice, fp, empty) = {tuple(round(p, 4) for p in parts)}")
if run is not None:
    run.summary.update(dict(oof_score=tuned, thr=dict(zip(CLASSES, THR)),
                            min_pix=dict(zip(CLASSES, MINPIX)), gate=dict(zip(CLASSES, GATE))))

# --------------------- TEST INFERENCE -> SUBMISSION ------------------
models = []
for k in RUN_FOLDS:
    m = build_model(); m.load_state_dict(torch.load(WORK / f"fold{k}.pth", map_location=DEV)); models.append(m.eval())
test_ids = ss["image_id"].unique().tolist()
tdl = make_loader(SegDS(test_ids, TEST_DIR, labeled=False))
thr_t = torch.tensor(THR, device=DEV).view(1, C, 1, 1)
gate_t, minpix_t = torch.tensor(GATE, device=DEV).view(1, C), torch.tensor(MINPIX, device=DEV).view(1, C)

rle_map, idx = {}, 0
for x, _ in tqdm(tdl, desc="Test"):
    p, c = infer(models, x.to(DEV), 4)
    m = p > thr_t
    keep = (m.sum((2, 3)) >= minpix_t) & (c >= gate_t)
    m = (m & keep[:, :, None, None]).cpu().numpy().astype(np.uint8)
    for b in range(m.shape[0]):
        for ci, d in enumerate(CLASSES):
            rle_map[(test_ids[idx + b], d)] = rle_encode(m[b, ci])
    idx += m.shape[0]

sub = ss.copy()
sub["mask_rle"] = [rle_map[(i, d)] for i, d in zip(sub["image_id"], sub["defect_type"])]
sub.to_csv("submission.csv", index=False)

# ------------------------------ CHECKS -------------------------------
chk = pd.read_csv("submission.csv", keep_default_na=False)
print(chk.shape == ss.shape, chk.columns.tolist())
print("share of rows with predicted mask:", (chk["mask_rle"].str.strip() != "").mean())
t = np.zeros((H, W), np.uint8); t[10:20, 100:130] = 1
assert (rle_decode(rle_encode(t)) == t).all()
print(chk.head())
if run is not None: run.finish()