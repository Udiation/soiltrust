#!/usr/bin/env python3
"""North-America-to-Europe continental transfer test for MMEarth SOC.

The orchestrator launches five independent single-GPU SeCo-Eco runs and three
training-mean baseline jobs. It deliberately does not use DDP.
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
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchgeo.models import ResNet50_Weights, resnet50


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "soil_organic_carbon"
RUNS = ROOT / "continental_runs"
DESIGN_PATH = RUNS / "split_design.json"
RESULTS = ROOT / "continental_transfer.csv"
SUMMARY = ROOT.parent / "continental-log.md"

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
NO_DATA = 65535
BLOCK_KM = 50.0
KM_PER_DEG_LAT = 111.32
BASE_SEED = 20260901


@dataclass(frozen=True)
class Job:
    run_id: str
    gpu: int
    seed: int
    kind: str


JOBS = [Job(f"deep_seed_{i}", i, BASE_SEED + i, "deep") for i in range(5)] + [
    Job(f"mean_seed_{i}", 5 + i, BASE_SEED + i, "mean_baseline") for i in range(3)
]


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["orchestrate", "preflight", "worker", "consolidate"], default="orchestrate")
    p.add_argument("--run-id")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stopping", type=int, default=8)
    return p.parse_args()


def append_summary(text: str) -> None:
    with SUMMARY.open("a") as f:
        f.write(text.rstrip() + "\n")


def load_geo_targets() -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(DATA / "soil_organic_carbon.h5", "r") as f:
        geo = f["geolocation"][:]
    y = np.load(DATA / "soc_gkg.npy", mmap_mode="r")
    if len(geo) != len(y):
        raise RuntimeError(f"geolocation/target mismatch: {len(geo)} vs {len(y)}")
    return geo, np.asarray(y, dtype=np.float32)


def block_ids(lonlat: np.ndarray) -> np.ndarray:
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    lat_step = BLOCK_KM / KM_PER_DEG_LAT
    lat_bin = np.floor((lat + 90.0) / lat_step).astype(np.int64)
    mid_lat = -90.0 + (lat_bin + 0.5) * lat_step
    lon_step = BLOCK_KM / (KM_PER_DEG_LAT * np.maximum(np.cos(np.deg2rad(mid_lat)), 0.1))
    lon_bin = np.floor((lon + 180.0) / lon_step).astype(np.int64)
    return np.asarray([f"{a}:{b}" for a, b in zip(lat_bin, lon_bin)])


def make_design() -> dict:
    geo, _ = load_geo_targets()
    lon, lat = geo[:, 0], geo[:, 1]
    north_america = np.flatnonzero((lat >= 25) & (lat <= 70) & (lon >= -170) & (lon <= -50))
    europe = np.flatnonzero((lat >= 35) & (lat <= 70) & (lon >= -10) & (lon <= 40))
    if len(set(north_america) & set(europe)):
        raise RuntimeError("North America and Europe bbox membership overlaps")

    blocks = block_ids(geo)
    unique, counts = np.unique(blocks[north_america], return_counts=True)
    rng = np.random.default_rng(BASE_SEED)
    order = rng.permutation(len(unique))
    cumulative = np.cumsum(counts[order])
    target = 0.15 * len(north_america)
    n_blocks = int(np.argmin(np.abs(cumulative - target))) + 1
    held_blocks = set(unique[order[:n_blocks]])
    within = np.asarray([i for i in north_america if blocks[i] in held_blocks], dtype=np.int64)
    train = np.asarray([i for i in north_america if blocks[i] not in held_blocks], dtype=np.int64)
    if set(blocks[train]) & set(blocks[within]):
        raise RuntimeError("spatial-block leakage in North America split")
    if not train.size or not within.size or not europe.size:
        raise RuntimeError("empty train or evaluation split")

    def bbox(idx: np.ndarray) -> dict:
        g = geo[idx]
        return {"lon_min": float(g[:, 0].min()), "lat_min": float(g[:, 1].min()),
                "lon_max": float(g[:, 0].max()), "lat_max": float(g[:, 1].max())}

    return {
        "coordinate_order": "geolocation[:,0]=longitude, geolocation[:,1]=latitude",
        "north_america_bbox": {"lat": [25, 70], "lon": [-170, -50], "inclusive": True},
        "europe_bbox": {"lat": [35, 70], "lon": [-10, 40], "inclusive": True},
        "block_km": BLOCK_KM,
        "block_method": "50/111.32 degree latitude bands; longitude width cosine-adjusted at band midpoint",
        "split_seed": BASE_SEED,
        "north_america_total": int(len(north_america)),
        "europe_total": int(len(europe)),
        "north_america_train_indices": train.tolist(),
        "north_america_holdout_indices": within.tolist(),
        "europe_indices": europe.tolist(),
        "north_america_train_blocks": int(len(set(blocks[train]))),
        "north_america_holdout_blocks": int(len(set(blocks[within]))),
        "north_america_holdout_fraction": float(len(within) / len(north_america)),
        "north_america_train_bbox": bbox(train),
        "north_america_holdout_bbox": bbox(within),
        "europe_observed_bbox": bbox(europe),
        "r2_convention": "1 - SSE/sum((y - mean(y_evaluation_split))^2); never clipped",
        "bias_convention": "mean(prediction - truth)",
    }


def preflight() -> dict:
    if RUNS.exists() or SUMMARY.exists() or RESULTS.exists():
        raise FileExistsError("continental output already exists; refusing to overwrite")
    design = make_design()
    RUNS.mkdir(parents=True)
    DESIGN_PATH.write_text(json.dumps(design, indent=2))
    mapping = "\n".join(f"| {j.gpu} | {j.run_id} | {j.seed} | {j.kind} |" for j in JOBS)
    text = f"""# Continental transfer test

## Preflight

- North America bbox count before splitting: **{design['north_america_total']:,}**.
- Europe transfer bbox count: **{design['europe_total']:,}**.
- North America training: **{len(design['north_america_train_indices']):,} samples**, **{design['north_america_train_blocks']:,} blocks**.
- North America spatial holdout: **{len(design['north_america_holdout_indices']):,} samples** ({100*design['north_america_holdout_fraction']:.3f}%), **{design['north_america_holdout_blocks']:,} blocks**.
- Approximate block size: **50 km × 50 km**; {design['block_method']}.
- R² convention: {design['r2_convention']}.
- Mean baseline: North America training-target arithmetic mean predicted on both evaluation sets.
- Loader workers: **2 per deep job** (10 total); baseline jobs do not create loaders.

| GPU | Run | Seed | Kind |
|---:|---|---:|---|
{mapping}

Deep hyperparameters match `train_soc.py`: SeCo-Eco ResNet-50, end-to-end Huber loss on training z-scores, AdamW lr 2e-4 and weight decay 1e-4, cosine schedule, 30 epochs, batch 32, gradient clipping 5, BF16 autocast, patience 8.
"""
    SUMMARY.write_text(text)
    print(text, flush=True)
    return design


class SOCDataset(Dataset):
    def __init__(self, indices: list[int], train: bool):
        self.images = np.load(DATA / "sentinel2.npy", mmap_mode="r")
        self.targets = np.load(DATA / "soc_gkg.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        x = torch.from_numpy(np.array(self.images[idx], dtype=np.float32, copy=True))
        x[x == NO_DATA] = 0
        x = x.clamp_(0, 10000).div_(10000.0)
        x = TF.resize(x, [224, 224], antialias=True)
        if self.train:
            if random.random() < 0.5:
                x = TF.hflip(x)
            if random.random() < 0.5:
                x = TF.vflip(x)
            x = torch.rot90(x, random.randrange(4), dims=(-2, -1))
        return x, torch.tensor(float(self.targets[idx]), dtype=torch.float32)


def loader(indices: list[int], train: bool, batch: int, workers: int) -> DataLoader:
    return DataLoader(SOCDataset(indices, train), batch_size=batch, shuffle=train,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      drop_last=train)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    denom = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / denom), "bias": float(np.mean(err))}


@torch.no_grad()
def evaluate(model, dl: DataLoader, mean: float, std: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).flatten()
        ys.append(y.numpy())
        ps.append((p.float().cpu().numpy() * std + mean))
    return np.concatenate(ys), np.concatenate(ps)


def worker(cfg: argparse.Namespace, job: Job) -> None:
    out = RUNS / job.run_id
    out.mkdir()
    design = json.loads(DESIGN_PATH.read_text())
    targets = np.load(DATA / "soc_gkg.npy", mmap_mode="r")
    train_idx = design["north_america_train_indices"]
    within_idx = design["north_america_holdout_indices"]
    europe_idx = design["europe_indices"]
    mean = float(np.mean(targets[train_idx], dtype=np.float64))
    std = float(np.std(targets[train_idx], dtype=np.float64))
    common = {"run_id": job.run_id, "kind": job.kind, "seed": job.seed,
              "training_mean": mean, "training_std": std}

    if job.kind == "mean_baseline":
        result = {**common, "best_epoch": None, "splits": {}}
        for name, idx in [("north_america_holdout", within_idx), ("europe_transfer", europe_idx)]:
            y = np.asarray(targets[idx], float)
            result["splits"][name] = metrics(y, np.full(len(y), mean))
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print("FINAL " + json.dumps(result), flush=True)
        return

    random.seed(job.seed); np.random.seed(job.seed); torch.manual_seed(job.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", 0)
    train_dl = loader(train_idx, True, cfg.batch_size, cfg.workers)
    within_dl = loader(within_idx, False, cfg.batch_size, cfg.workers)
    europe_dl = loader(europe_idx, False, cfg.batch_size, cfg.workers)
    weights = ResNet50_Weights.SENTINEL2_ALL_SECO_ECO
    if list(weights.meta["bands"]) != BANDS:
        raise RuntimeError(f"weight bands mismatch: {weights.meta['bands']}")
    model = resnet50(weights=weights, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr / 50)
    best_mae, stale, history = float("inf"), 0, []
    for epoch in range(1, cfg.epochs + 1):
        started = time.time(); model.train(); total = count = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            yz = (y - mean) / std
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.huber_loss(model(x).flatten(), yz, delta=1.0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach()) * y.numel(); count += y.numel()
        sched.step()
        wy, wp = evaluate(model, within_dl, mean, std, device)
        wm = metrics(wy, wp)
        row = {"epoch": epoch, "train_loss": total / count, **{f"within_{k}": v for k, v in wm.items()},
               "lr": sched.get_last_lr()[0], "epoch_seconds": time.time() - started}
        history.append(row)
        with (out / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerows(history)
        print("METRIC " + json.dumps(row), flush=True)
        if wm["mae"] < best_mae:
            best_mae, stale = wm["mae"], 0
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                        "within": wm, "config": {**common, "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size, "lr": cfg.lr, "weight_decay": cfg.weight_decay}}, out / "best_model.pt")
        else:
            stale += 1
        if stale >= cfg.early_stopping:
            break

    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    result = {**common, "best_epoch": ckpt["epoch"], "splits": {}}
    for name, dl in [("north_america_holdout", within_dl), ("europe_transfer", europe_dl)]:
        y, p = evaluate(model, dl, mean, std, device)
        result["splits"][name] = metrics(y, p)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL " + json.dumps(result), flush=True)


def consolidate() -> None:
    records = []
    results = []
    for job in JOBS:
        path = RUNS / job.run_id / "result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text()); results.append(result)
        for split, m in result["splits"].items():
            records.append({"run_id": job.run_id, "kind": job.kind, "seed": job.seed,
                            "aggregation": "individual", "split": split, **m})
    for kind in ["deep", "mean_baseline"]:
        group = [r for r in records if r["kind"] == kind]
        for split in ["north_america_holdout", "europe_transfer"]:
            rows = [r for r in group if r["split"] == split]
            for agg, ddof in [("mean", 0), ("std", 1)]:
                vals = {k: float(np.mean([r[k] for r in rows])) if agg == "mean" else
                        float(np.std([r[k] for r in rows], ddof=ddof)) for k in ["mae", "rmse", "r2", "bias"]}
                records.append({"run_id": f"{kind}_{agg}", "kind": kind, "seed": "",
                                "aggregation": agg, "split": split, "n": rows[0]["n"], **vals})

    fields = ["run_id", "kind", "seed", "aggregation", "split", "n", "mae", "rmse", "r2", "bias"]
    with RESULTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)

    def summary(kind: str, split: str, metric: str) -> tuple[float, float]:
        vals = [r[metric] for r in records if r["kind"] == kind and r["split"] == split and r["aggregation"] == "individual"]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    deep_rmse, deep_rmse_sd = summary("deep", "europe_transfer", "rmse")
    base_rmse, base_rmse_sd = summary("mean_baseline", "europe_transfer", "rmse")
    ratio = deep_rmse / base_rmse
    lines = ["", "## Final results", "",
             "All R² values use the corresponding evaluation set's own mean denominator and are reported without clipping.", "",
             "| Model | Split | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD | Bias mean ± SD |",
             "|---|---|---:|---:|---:|---:|"]
    for kind in ["deep", "mean_baseline"]:
        for split in ["north_america_holdout", "europe_transfer"]:
            values = [summary(kind, split, m) for m in ["mae", "rmse", "r2", "bias"]]
            fmt = [f"{a:.4f} ± {b:.4f}" for a, b in values]
            lines.append(f"| {kind} | {split} | " + " | ".join(fmt) + " |")
    lines += ["", "## Headline Europe transfer comparison", "",
              f"- Deep Europe RMSE: **{deep_rmse:.4f} ± {deep_rmse_sd:.4f} g/kg**.",
              f"- North-America-training-mean baseline Europe RMSE: **{base_rmse:.4f} ± {base_rmse_sd:.4f} g/kg**.",
              f"- RMSE ratio (deep / baseline): **{ratio:.4f}**.",
              f"- Relative RMSE change: **{100*(1-ratio):.2f}% reduction** versus the baseline." if ratio < 1 else
              f"- Relative RMSE change: **{100*(ratio-1):.2f}% increase** versus the baseline.",
              ("- The mean ratio is below 1.0, so the deep model shows transferred predictive skill by the requested criterion."
               if ratio < 1 else
               "- The mean ratio is above 1.0, so the deep model does not beat the transfer baseline by the requested criterion.")]
    append_summary("\n".join(lines))
    print("\n".join(lines), flush=True)


def orchestrate(cfg: argparse.Namespace) -> None:
    preflight()
    procs = []
    for job in JOBS:
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        log_path = RUNS / f"{job.run_id}.launcher.log"
        log = log_path.open("w")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--run-id", job.run_id,
               "--epochs", str(cfg.epochs), "--batch-size", str(cfg.batch_size), "--workers", str(cfg.workers),
               "--lr", str(cfg.lr), "--weight-decay", str(cfg.weight_decay),
               "--early-stopping", str(cfg.early_stopping)]
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((job, p, log))
        print(f"LAUNCHED gpu={job.gpu} pid={p.pid} run={job.run_id} log={log_path}", flush=True)
    failed = []
    for job, p, log in procs:
        rc = p.wait(); log.close()
        print(f"COMPLETED run={job.run_id} rc={rc}", flush=True)
        if rc:
            failed.append((job.run_id, rc))
    if failed:
        append_summary(f"\n## Failure\n\nJobs failed: `{failed}`. See launcher logs in `{RUNS}`.")
        raise RuntimeError(f"jobs failed: {failed}")
    consolidate()


def main() -> None:
    cfg = cli()
    if cfg.mode == "preflight":
        print(json.dumps(make_design(), indent=2))
    elif cfg.mode == "consolidate":
        consolidate()
    elif cfg.mode == "worker":
        if not cfg.run_id:
            raise ValueError("--run-id is required")
        job = next((j for j in JOBS if j.run_id == cfg.run_id), None)
        if job is None:
            raise ValueError(f"unknown run id {cfg.run_id}")
        worker(cfg, job)
    else:
        orchestrate(cfg)


if __name__ == "__main__":
    main()
