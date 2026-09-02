#!/usr/bin/env python3
"""Leakage-aware spatial block CV for Sentinel-2 soil organic carbon.

The default orchestrator validates the split, then launches eight independent
single-GPU subprocesses. It never uses DDP. Five control models cover five
spatial folds; three additional jobs compare backbones on fold 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights as ImageNetWeights
from torchvision.models import resnet50 as torchvision_resnet50
from torchvision.transforms import functional as TF
from torchgeo.models import ResNet50_Weights, resnet50 as torchgeo_resnet50


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "soil_organic_carbon"
RUNS_DIR = ROOT / "spatial_cv_runs"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_CSV = ROOT / "spatial_cv_results.csv"
SUMMARY_LOG = ROOT.parent / "spatial-cv-log.md"
BLOCK_KM = 50.0
KM_PER_DEG_LAT = 111.32
N_FOLDS = 5
MIN_FOLD_SAMPLES = 300
WORKERS_PER_JOB = 2
BASE_SEED = 20260901
BANDS_12 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
NO_DATA = 65535


@dataclass(frozen=True)
class Job:
    run_id: str
    gpu: int
    fold: int
    backbone: str
    seed: int
    role: str


JOBS = [
    Job("cv_fold_0", 0, 0, "seco_eco", BASE_SEED + 0, "control_cv"),
    Job("cv_fold_1", 1, 1, "seco_eco", BASE_SEED + 1, "control_cv"),
    Job("cv_fold_2", 2, 2, "seco_eco", BASE_SEED + 2, "control_cv"),
    Job("cv_fold_3", 3, 3, "seco_eco", BASE_SEED + 3, "control_cv"),
    Job("cv_fold_4", 4, 4, "seco_eco", BASE_SEED + 4, "control_cv"),
    # Deliberately different from GPU 0 for run-to-run variance.
    Job("compare_seco_seed", 5, 0, "seco_eco", BASE_SEED + 5000, "backbone_comparison"),
    Job("compare_ssl4eo_moco", 6, 0, "ssl4eo_moco", BASE_SEED + 0, "backbone_comparison"),
    Job("compare_imagenet", 7, 0, "imagenet", BASE_SEED + 0, "backbone_comparison"),
]


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["orchestrate", "preflight", "worker", "consolidate"], default="orchestrate")
    p.add_argument("--run-id")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=WORKERS_PER_JOB)
    p.add_argument("--head-epochs", type=int, default=3)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    return p.parse_args()


def append_log(text: str) -> None:
    with SUMMARY_LOG.open("a") as f:
        f.write(text.rstrip() + "\n")


def load_source() -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(DATA_DIR / "soil_organic_carbon.h5", "r") as f:
        geolocation = f["geolocation"][:]
    targets = np.load(DATA_DIR / "soc_gkg.npy", mmap_mode="r")
    split = json.loads((DATA_DIR / "soil_organic_carbon_split_data.json").read_text())
    return geolocation, targets, split


def spatial_block_ids(lonlat: np.ndarray) -> np.ndarray:
    """Approximate 50 km square cells on a sphere, adjusting longitude by latitude."""
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    lat_step = BLOCK_KM / KM_PER_DEG_LAT
    lat_bin = np.floor((lat + 90.0) / lat_step).astype(np.int64)
    mid_lat = -90.0 + (lat_bin + 0.5) * lat_step
    lon_step = BLOCK_KM / (KM_PER_DEG_LAT * np.maximum(np.cos(np.deg2rad(mid_lat)), 0.1))
    lon_bin = np.floor((lon + 180.0) / lon_step).astype(np.int64)
    return np.asarray([f"{a}:{b}" for a, b in zip(lat_bin, lon_bin)])


def make_folds() -> dict:
    geolocation, targets, split = load_source()
    holdout = np.asarray(split["geographic_test_indices"], dtype=np.int64)
    pool = np.asarray(sorted(set(split["train_100%_indices"]) | set(split["val_indices"]) |
                             set(split["random_test_indices"])), dtype=np.int64)
    assert len(set(pool) & set(holdout)) == 0
    blocks = spatial_block_ids(geolocation)
    group_cv = GroupKFold(n_splits=N_FOLDS)
    folds = []
    for fold, (tr_pos, va_pos) in enumerate(group_cv.split(pool, targets[pool], groups=blocks[pool])):
        train_idx, val_idx = pool[tr_pos], pool[va_pos]
        train_blocks, val_blocks = set(blocks[train_idx]), set(blocks[val_idx])
        assert train_blocks.isdisjoint(val_blocks), f"block leakage in fold {fold}"
        bbox = geolocation[val_idx]
        folds.append({
            "fold": fold,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_blocks": len(train_blocks),
            "val_blocks": len(val_blocks),
            "val_bbox": {"lon_min": float(bbox[:, 0].min()), "lat_min": float(bbox[:, 1].min()),
                         "lon_max": float(bbox[:, 0].max()), "lat_max": float(bbox[:, 1].max())},
        })
    hb = geolocation[holdout]
    return {
        "method": "latitude bands of 50/111.32 degrees; longitude cell width corrected by cosine of band midpoint",
        "block_km": BLOCK_KM,
        "choice": "50 km was user-specified as the target scale for suppressing local spatial autocorrelation",
        "r2_convention": "R2 = 1 - SSE / sum((y - mean(y_evaluation_split))^2); fold/split mean denominator for every row",
        "pool_indices": pool.tolist(), "holdout_indices": holdout.tolist(), "folds": folds,
        "holdout_bbox": {"lon_min": float(hb[:, 0].min()), "lat_min": float(hb[:, 1].min()),
                         "lon_max": float(hb[:, 0].max()), "lat_max": float(hb[:, 1].max())},
        "holdout_blocks": int(len(set(blocks[holdout]))),
    }


def preflight(write: bool = True) -> dict:
    design = make_folds()
    lines = [
        "## Spatial split preflight", "",
        f"- Block target: **{design['block_km']:.0f} km × {design['block_km']:.0f} km** (approximately).",
        f"- Choice: {design['choice']}.",
        f"- Construction: {design['method']}.",
        f"- R² convention: {design['r2_convention']}.",
        f"- Dataloader workers: **{WORKERS_PER_JOB} per job**, 16 total across eight concurrent jobs.", "",
        "| Fold | Train samples | Validation samples | Train blocks | Validation blocks | Validation bbox (lon_min, lat_min, lon_max, lat_max) |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    too_small = []
    for f in design["folds"]:
        b = f["val_bbox"]
        lines.append(f"| {f['fold']} | {len(f['train_indices'])} | {len(f['val_indices'])} | {f['train_blocks']} | {f['val_blocks']} | "
                     f"({b['lon_min']:.3f}, {b['lat_min']:.3f}, {b['lon_max']:.3f}, {b['lat_max']:.3f}) |")
        if len(f["val_indices"]) < MIN_FOLD_SAMPLES:
            too_small.append((f["fold"], len(f["val_indices"])))
    hb = design["holdout_bbox"]
    lines += ["", f"Geographic holdout: **{len(design['holdout_indices'])} samples in {design['holdout_blocks']} blocks**; "
                     f"bbox ({hb['lon_min']:.3f}, {hb['lat_min']:.3f}, {hb['lon_max']:.3f}, {hb['lat_max']:.3f}).",
              "", "Whole-block disjointness assertions passed for all folds."]
    report = "\n".join(lines)
    print(report, flush=True)
    if write:
        if SUMMARY_LOG.exists():
            raise FileExistsError(f"Refusing to overwrite existing {SUMMARY_LOG}")
        SUMMARY_LOG.write_text("# Spatial block cross-validation run\n\n" + report + "\n")
        RUNS_DIR.mkdir(parents=True, exist_ok=False)
        (RUNS_DIR / "split_design.json").write_text(json.dumps(design, indent=2))
    if too_small:
        raise RuntimeError(f"Fold(s) below hard minimum {MIN_FOLD_SAMPLES}: {too_small}")
    return design


class SOCDataset(Dataset):
    """Same mmap loading, reflectance scaling, resize, and augmentation as train_soc.py."""
    def __init__(self, indices: list[int], train: bool):
        self.images = np.load(DATA_DIR / "sentinel2.npy", mmap_mode="r")
        self.targets = np.load(DATA_DIR / "soc_gkg.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        x = torch.from_numpy(np.array(self.images[idx], dtype=np.float32, copy=True))
        x[x == NO_DATA] = 0
        x = x.clamp_(0, 10000).div_(10000.0)
        x = TF.resize(x, [224, 224], antialias=True)
        if self.train:
            if random.random() < 0.5: x = TF.hflip(x)
            if random.random() < 0.5: x = TF.vflip(x)
            x = torch.rot90(x, random.randrange(4), dims=(-2, -1))
        return x, torch.tensor(float(self.targets[idx]), dtype=torch.float32), idx


class InputAdapter(nn.Module):
    def __init__(self, model: nn.Module, kind: str):
        super().__init__(); self.model = model; self.kind = kind
        self.register_buffer("moco_mean", torch.tensor([1612.9,1397.6,1322.3,1373.1,1561.,2108.4,2390.7,2318.7,2581.,837.7,22.,2195.2,1537.4]).view(1,13,1,1)/10000)
        self.register_buffer("moco_std", torch.tensor([791.,854.3,878.7,1144.9,1127.5,1164.2,1276.,1249.5,1345.9,577.5,47.5,1340.,1142.9]).view(1,13,1,1)/10000)
        self.register_buffer("img_mean", torch.tensor([0.449]).view(1,1,1,1))
        self.register_buffer("img_std", torch.tensor([0.226]).view(1,1,1,1))

    @property
    def fc(self): return self.model.fc

    def forward(self, x):
        if self.kind == "ssl4eo_moco":
            # Insert absent L2A B10 between B9 and B11; its normalized value is explicit.
            x = torch.cat([x[:, :10], torch.zeros_like(x[:, :1]), x[:, 10:]], dim=1)
            x = (x - self.moco_mean) / self.moco_std
        elif self.kind == "imagenet":
            x = (x - self.img_mean) / self.img_std
        return self.model(x)


def build_model(kind: str) -> nn.Module:
    if kind == "seco_eco":
        w = ResNet50_Weights.SENTINEL2_ALL_SECO_ECO
        return InputAdapter(torchgeo_resnet50(weights=w, num_classes=1), kind)
    if kind == "ssl4eo_moco":
        w = ResNet50_Weights.SENTINEL2_ALL_MOCO
        return InputAdapter(torchgeo_resnet50(weights=w, num_classes=1), kind)
    if kind == "imagenet":
        m = torchvision_resnet50(weights=ImageNetWeights.IMAGENET1K_V2)
        old = m.conv1
        new = nn.Conv2d(12, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            mean_kernel = old.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(mean_kernel.repeat(1, 12, 1, 1) * (3.0 / 12.0))
        m.conv1 = new; m.fc = nn.Linear(m.fc.in_features, 1)
        return InputAdapter(m, kind)
    raise ValueError(kind)


def set_frozen(model: InputAdapter, frozen: bool) -> None:
    for p in model.parameters(): p.requires_grad = not frozen
    for p in model.fc.parameters(): p.requires_grad = True


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    sse = float(np.sum(err ** 2)); denom = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - sse / denom) if denom > 0 else float("nan"), "bias": float(np.mean(err))}


@torch.no_grad()
def predict(model, loader, mean, std, device):
    model.eval(); ys, ps, ids = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): z = model(x).flatten()
        ys.append(y.numpy()); ps.append((z.float().cpu().numpy() * std + mean)); ids.append(idx.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ids)


def worker(job: Job, cfg: argparse.Namespace) -> None:
    out = RUNS_DIR / job.run_id; out.mkdir(exist_ok=False)
    log_path = out / "train.log"
    def say(s):
        print(s, flush=True)
        with log_path.open("a") as f: f.write(s + "\n")
    random.seed(job.seed); np.random.seed(job.seed); torch.manual_seed(job.seed)
    torch.cuda.set_device(0); device = torch.device("cuda:0"); torch.backends.cudnn.benchmark = True
    design = json.loads((RUNS_DIR / "split_design.json").read_text())
    fold = design["folds"][job.fold]
    train_idx, val_idx, hold_idx = fold["train_indices"], fold["val_indices"], design["holdout_indices"]
    target = np.load(DATA_DIR / "soc_gkg.npy", mmap_mode="r")
    target_mean = float(np.mean(target[train_idx], dtype=np.float64)); target_std = float(np.std(target[train_idx], dtype=np.float64))
    train = DataLoader(SOCDataset(train_idx, True), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
                       pin_memory=True, persistent_workers=cfg.workers > 0, drop_last=True)
    val = DataLoader(SOCDataset(val_idx, False), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers,
                     pin_memory=True, persistent_workers=cfg.workers > 0)
    hold = DataLoader(SOCDataset(hold_idx, False), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers,
                      pin_memory=True, persistent_workers=cfg.workers > 0)
    try:
        model = build_model(job.backbone).to(device)
    except Exception as e:
        result = {"status": "skipped_unavailable", "job": asdict(job), "error": repr(e)}
        (out / "result.json").write_text(json.dumps(result, indent=2)); say(json.dumps(result)); return
    best_mae, best_epoch, stale = float("inf"), 0, 0
    total_epochs = cfg.head_epochs + cfg.finetune_epochs
    set_frozen(model, True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    say(f"START {asdict(job)} train={len(train_idx)} val={len(val_idx)} holdout={len(hold_idx)} workers={cfg.workers}")
    history = []
    for epoch in range(1, total_epochs + 1):
        if epoch == cfg.head_epochs + 1:
            set_frozen(model, False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.finetune_epochs, eta_min=4e-6)
        started=time.time(); model.train(); sum_loss=0.; seen=0
        for x,y,_ in train:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); yz=(y-target_mean)/target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=F.huber_loss(model(x).flatten(),yz,delta=1.)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            sum_loss += float(loss.detach())*len(y); seen += len(y)
        if epoch > cfg.head_epochs: scheduler.step()
        vy,vp,vi=predict(model,val,target_mean,target_std,device); vm=metrics(vy,vp)
        row={"epoch":epoch,"phase":"head" if epoch<=cfg.head_epochs else "finetune","train_loss":sum_loss/seen,
             **{f"val_{k}":v for k,v in vm.items() if k!="n"},"seconds":time.time()-started}
        history.append(row); say("EPOCH "+json.dumps(row))
        if vm["mae"] < best_mae:
            best_mae,best_epoch,stale=vm["mae"],epoch,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"job":asdict(job),"target_mean":target_mean,
                        "target_std":target_std,"val":vm},out/"best_model.pt")
        elif epoch > cfg.head_epochs:
            stale += 1
        if epoch > cfg.head_epochs and stale >= cfg.patience: break
    ckpt=torch.load(out/"best_model.pt",map_location=device,weights_only=False); model.load_state_dict(ckpt["model"])
    vy,vp,vi=predict(model,val,target_mean,target_std,device); hy,hp,hi=predict(model,hold,target_mean,target_std,device)
    np.savez_compressed(out/"predictions.npz",val_y=vy,val_pred=vp,val_idx=vi,holdout_y=hy,holdout_pred=hp,
                        holdout_idx=hi,val_null=np.full_like(vy,target_mean),holdout_null=np.full_like(hy,target_mean))
    result={"status":"complete","job":asdict(job),"best_epoch":best_epoch,"target_mean":target_mean,
            "target_std":target_std,"validation":metrics(vy,vp),"geographic_holdout":metrics(hy,hp),
            "validation_null":metrics(vy,np.full_like(vy,target_mean)),
            "geographic_holdout_null":metrics(hy,np.full_like(hy,target_mean)),"history":history}
    (out/"result.json").write_text(json.dumps(result,indent=2)); say("FINAL "+json.dumps(result))


def result_rows(run_id: str, role: str, backbone: str, seed, fold, split_name: str, model_m: dict, null_m: dict):
    base={"run_id":run_id,"role":role,"backbone":backbone,"seed":seed,"fold":fold,"split":split_name}
    return [{**base,"row_type":"model",**model_m},{**base,"row_type":"null_train_mean",**null_m}]


def consolidate() -> None:
    completed=[]; skipped=[]
    for job in JOBS:
        p=RUNS_DIR/job.run_id/"result.json"
        if not p.exists(): skipped.append({"job":asdict(job),"error":"missing result.json"}); continue
        r=json.loads(p.read_text())
        (completed if r.get("status")=="complete" else skipped).append(r)
    controls=[r for r in completed if r["job"]["role"]=="control_cv"]
    if len(controls)!=5: raise RuntimeError(f"Need all 5 control folds, got {len(controls)}; skipped={skipped}")
    rows=[]
    for r in completed:
        j=r["job"]
        rows += result_rows(j["run_id"],j["role"],j["backbone"],j["seed"],j["fold"],"spatial_validation",r["validation"],r["validation_null"])
        rows += result_rows(j["run_id"],j["role"],j["backbone"],j["seed"],j["fold"],"geographic_holdout",r["geographic_holdout"],r["geographic_holdout_null"])
    oof=[]; hold=[]
    for r in controls:
        z=np.load(RUNS_DIR/r["job"]["run_id"]/"predictions.npz"); oof.append(z); hold.append(z)
    oof_y=np.concatenate([z["val_y"] for z in oof]); oof_p=np.concatenate([z["val_pred"] for z in oof]); oof_n=np.concatenate([z["val_null"] for z in oof])
    rows += result_rows("control_pooled_oof","aggregate","seco_eco","",-1,"spatial_validation",metrics(oof_y,oof_p),metrics(oof_y,oof_n))
    hold_y=hold[0]["holdout_y"]; ensemble=np.mean(np.stack([z["holdout_pred"] for z in hold]),axis=0)
    null_ensemble=np.mean(np.stack([z["holdout_null"] for z in hold]),axis=0)
    rows += result_rows("control_5model_ensemble","ensemble_secondary","seco_eco","",-1,"geographic_holdout",metrics(hold_y,ensemble),metrics(hold_y,null_ensemble))
    fields=["run_id","role","backbone","seed","fold","split","row_type","n","mae","rmse","r2","bias"]
    with RESULTS_CSV.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Best run is selected exclusively by spatial-validation R2.
    best=max(completed,key=lambda r:r["validation"]["r2"]); bz=np.load(RUNS_DIR/best["job"]["run_id"]/"predictions.npz")
    scatter=ARTIFACTS_DIR/"spatial_cv_best_scatter.png"; bars=ARTIFACTS_DIR/"spatial_cv_fold_r2.png"
    for p in (scatter,bars):
        if p.exists(): raise FileExistsError(f"Refusing to overwrite {p}")
    fig,ax=plt.subplots(1,2,figsize=(11,5))
    for a,x,y,title in [(ax[0],bz["val_y"],bz["val_pred"],"Best run: spatial validation"),(ax[1],bz["holdout_y"],bz["holdout_pred"],"Same run: geographic holdout")]:
        a.scatter(x,y,s=10,alpha=.45); lo=float(min(x.min(),y.min()));hi=float(max(x.max(),y.max()));a.plot([lo,hi],[lo,hi],"k--");a.set(xlabel="Actual SOC (g/kg)",ylabel="Predicted SOC (g/kg)",title=title);a.grid(alpha=.2)
    fig.suptitle(best["job"]["run_id"]);fig.tight_layout();fig.savefig(scatter,dpi=170);plt.close(fig)
    fold_r2=[r["validation"]["r2"] for r in sorted(controls,key=lambda x:x["job"]["fold"])]
    fig,ax=plt.subplots(figsize=(7,4.5));ax.bar(range(5),fold_r2);ax.axhline(0,color="k",lw=1);ax.set(xlabel="Spatial fold",ylabel="R² (fold-mean denominator)",title="SeCo-Eco spatial block CV");ax.set_xticks(range(5));fig.tight_layout();fig.savefig(bars,dpi=170);plt.close(fig)
    # Headline is distribution of individual deployable control models, never the ensemble.
    keys=["mae","rmse","r2","bias"]
    ind={k:[r["geographic_holdout"][k] for r in controls] for k in keys}
    headline={k:{"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1))} for k,v in ind.items()}
    ensemble_m=metrics(hold_y,ensemble); pooled_m=metrics(oof_y,oof_p)
    gpu0=next(r for r in completed if r["job"]["run_id"]=="cv_fold_0")
    gpu5=next((r for r in completed if r["job"]["run_id"]=="compare_seco_seed"),None)
    gap={s:{k:gpu5[s][k]-gpu0[s][k] for k in keys} for s in ("validation","geographic_holdout")} if gpu5 else None
    lines=["", "## Final consolidated results", "",
           "R² uses the mean of the evaluation split in its denominator for every model and null row. The null predictor outputs that run's training-set mean SOC.", "",
           "### Headline: individual deployable control models on geographic holdout", "",
           "Mean ± sample SD across the five separately trained control-fold models:", ""]
    for k in keys: lines.append(f"- {k.upper()}: **{headline[k]['mean']:.4f} ± {headline[k]['std']:.4f}**")
    lines += ["", "### Secondary: what a five-model ensemble buys", "", json.dumps(ensemble_m,indent=2), "",
              "### Pooled out-of-fold spatial CV", "", json.dumps(pooled_m,indent=2), "",
              "### GPU 0 versus GPU 5 seed-only gap (GPU5 - GPU0)", "", json.dumps(gap,indent=2), "",
              f"Best run selected by spatial-validation R²: `{best['job']['run_id']}`.", "",
              f"Skipped jobs: `{json.dumps(skipped)}`", ""]
    append_log("\n".join(lines))
    print("\n".join(lines)); print(f"RESULTS {RESULTS_CSV}")


def prefetch_weights() -> dict[str,str]:
    status={}
    for kind in ("seco_eco","ssl4eo_moco","imagenet"):
        try:
            m=build_model(kind); del m; status[kind]="available"
        except Exception as e: status[kind]=f"unavailable: {e!r}"
    append_log("\n## Backbone availability\n\n"+"\n".join(f"- {k}: {v}" for k,v in status.items())+"\n")
    print(status,flush=True); return status


def orchestrate(cfg):
    for p in (RESULTS_CSV, ARTIFACTS_DIR/"spatial_cv_best_scatter.png", ARTIFACTS_DIR/"spatial_cv_fold_r2.png"):
        if p.exists(): raise FileExistsError(f"Refusing to overwrite existing output {p}")
    preflight(write=True); availability=prefetch_weights()
    append_log("\n## GPU job launch\n")
    procs=[]
    for job in JOBS:
        if availability[job.backbone].startswith("unavailable"):
            out=RUNS_DIR/job.run_id;out.mkdir()
            (out/"result.json").write_text(json.dumps({"status":"skipped_unavailable","job":asdict(job),"error":availability[job.backbone]},indent=2));continue
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(job.gpu);env["OMP_NUM_THREADS"]="4"
        cmd=[sys.executable,str(Path(__file__).resolve()),"--mode","worker","--run-id",job.run_id,"--batch-size",str(cfg.batch_size),"--workers",str(cfg.workers),"--head-epochs",str(cfg.head_epochs),"--finetune-epochs",str(cfg.finetune_epochs),"--patience",str(cfg.patience)]
        log=(RUNS_DIR/f"{job.run_id}.launcher.log").open("w")
        append_log(f"- GPU {job.gpu}: `{job.run_id}`, fold {job.fold}, backbone {job.backbone}, seed {job.seed}")
        procs.append((job,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    failures=[]
    for job,p,log in procs:
        rc=p.wait();log.close();
        if rc:failures.append((job.run_id,rc))
    append_log(f"\nWorker process failures: `{failures}`\n")
    if failures: raise RuntimeError(f"Worker failures: {failures}")
    consolidate()


def main():
    cfg=cli()
    if cfg.workers > 4: raise ValueError("workers per job capped at 4; requested value is too high")
    if cfg.mode=="preflight": preflight(write=False);return
    if cfg.mode=="consolidate": consolidate();return
    if cfg.mode=="worker":
        job=next((j for j in JOBS if j.run_id==cfg.run_id),None)
        if job is None: raise ValueError(cfg.run_id)
        worker(job,cfg);return
    orchestrate(cfg)


if __name__=="__main__": main()
