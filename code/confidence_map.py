#!/usr/bin/env python3
"""Five-model SOC prediction/agreement maps for continental transfer splits.

Inference only. Five independent subprocesses each load one already-trained
North-America SeCo-Eco checkpoint on one GPU; the parent consolidates outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchgeo.models import resnet50


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "soil_organic_carbon"
RUNS = ROOT / "continental_runs"
DESIGN = RUNS / "split_design.json"
OUT = ROOT / "artifacts" / "confidence_map"
FIGURE = OUT / "soil_confidence_map.png"
TABLE = OUT / "tile_predictions.csv"
SUMMARY_JSON = OUT / "summary.json"
REPORT = ROOT.parent / "confidence-map.md"
NO_DATA = 65535

CHECKPOINTS = [RUNS / f"deep_seed_{i}" / "best_model.pt" for i in range(5)]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int, help="Checkpoint/GPU slot 0..4")
    p.add_argument("--consolidate", action="store_true", help="Reuse completed worker outputs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


class TileDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.images = np.load(DATA / "sentinel2.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        x = torch.from_numpy(np.array(self.images[idx], dtype=np.float32, copy=True))
        x[x == NO_DATA] = 0
        x = TF.resize(x.clamp_(0, 10000).div_(10000), [224, 224], antialias=True)
        return x, idx


def selected_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = json.loads(DESIGN.read_text())
    north_america = np.asarray(d["north_america_holdout_indices"], dtype=np.int64)
    europe = np.asarray(d["europe_indices"], dtype=np.int64)
    return north_america, europe, np.concatenate([north_america, europe])


def worker(slot: int, batch_size: int, workers: int) -> None:
    checkpoint = CHECKPOINTS[slot]
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = resnet50(weights=None, in_chans=12, num_classes=1)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda:0")
    model.eval().to(device)
    _, _, indices = selected_indices()
    dl = DataLoader(TileDataset(indices), batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    predictions, observed_indices = [], []
    mean = float(ckpt["config"]["training_mean"])
    std = float(ckpt["config"]["training_std"])
    with torch.no_grad():
        for x, idx in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x).flatten()
            predictions.append(z.float().cpu().numpy() * std + mean)
            observed_indices.append(idx.numpy())
    prediction = np.concatenate(predictions)
    observed = np.concatenate(observed_indices)
    if not np.array_equal(observed, indices):
        raise RuntimeError("inference output order mismatch")
    np.savez(OUT / f"seed_{slot}_predictions.npz", indices=observed, prediction=prediction,
             seed=int(ckpt["config"]["seed"]), best_epoch=int(ckpt["epoch"]),
             checkpoint=str(checkpoint))
    print(json.dumps({"slot": slot, "seed": int(ckpt["config"]["seed"]),
                      "best_epoch": int(ckpt["epoch"]), "tiles": len(prediction),
                      "negative_predictions": int(np.sum(prediction < 0))}), flush=True)


def region_outline(ax, bounds: tuple[float, float, float, float]) -> None:
    shp = (Path(gpd.__file__).parent.parent / "pyogrio" / "tests" / "fixtures" /
           "naturalearth_lowres" / "naturalearth_lowres.shp")
    if shp.exists():
        world = gpd.read_file(shp)
        world.boundary.plot(ax=ax, color="#777777", linewidth=.45, zorder=0)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=.18)


def consolidate() -> dict:
    north_america, europe, indices = selected_indices()
    arrays, metadata = [], []
    for slot in range(5):
        path = OUT / f"seed_{slot}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        z = np.load(path)
        if not np.array_equal(z["indices"], indices):
            raise RuntimeError(f"index mismatch in {path}")
        arrays.append(z["prediction"])
        metadata.append({"slot": slot, "seed": int(z["seed"]), "best_epoch": int(z["best_epoch"]),
                         "checkpoint": str(z["checkpoint"])})
    predictions = np.stack(arrays, axis=1)
    ensemble_mean = predictions.mean(axis=1)
    disagreement = predictions.std(axis=1, ddof=1)
    any_negative = np.any(predictions < 0, axis=1)
    mean_negative = ensemble_mean < 0

    n_na = len(north_america)
    region = np.asarray(["North America"] * n_na + ["Europe"] * len(europe))
    threshold = float(np.quantile(disagreement[:n_na], .75))
    low_confidence = disagreement > threshold
    with h5py.File(DATA / "soil_organic_carbon.h5", "r") as h5:
        # h5py fancy indices must be monotonic; NumPy indexing has no such
        # restriction and preserves the exact saved region order.
        geo = h5["geolocation"][:][indices]
    targets = np.asarray(np.load(DATA / "soc_gkg.npy", mmap_mode="r")[indices], float)

    fields = ["region", "sample_index", "longitude", "latitude", "observed_soc_gkg",
              *[f"seed_{i}_soc_gkg" for i in range(5)], "mean_soc_gkg", "model_std_gkg",
              "low_confidence", "any_seed_negative", "ensemble_mean_negative"]
    with TABLE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for j, idx in enumerate(indices):
            row = {"region": region[j], "sample_index": int(idx), "longitude": float(geo[j, 0]),
                   "latitude": float(geo[j, 1]), "observed_soc_gkg": targets[j],
                   "mean_soc_gkg": float(ensemble_mean[j]), "model_std_gkg": float(disagreement[j]),
                   "low_confidence": bool(low_confidence[j]), "any_seed_negative": bool(any_negative[j]),
                   "ensemble_mean_negative": bool(mean_negative[j])}
            row.update({f"seed_{i}_soc_gkg": float(predictions[j, i]) for i in range(5)})
            w.writerow(row)

    masks = {"North America": np.arange(len(indices)) < n_na,
             "Europe": np.arange(len(indices)) >= n_na}
    summary = {"checkpoints": metadata, "checkpoint_source": "continental transfer five-seed run",
               "regions": {"north_america_holdout": n_na, "europe_transfer": len(europe)},
               "low_confidence_definition": "sample SD > North America holdout 75th percentile",
               "low_confidence_threshold_std_gkg": threshold}
    for name, mask in masks.items():
        key = "north_america" if name == "North America" else "europe"
        summary[key] = {
            "n": int(mask.sum()), "median_disagreement_gkg": float(np.median(disagreement[mask])),
            "mean_disagreement_gkg": float(np.mean(disagreement[mask])),
            "low_confidence_fraction": float(np.mean(low_confidence[mask])),
            "any_seed_negative_fraction": float(np.mean(any_negative[mask])),
            "ensemble_mean_negative_fraction": float(np.mean(mean_negative[mask])),
        }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    pred_lo, pred_hi = np.quantile(ensemble_mean, [.02, .98])
    dis_hi = max(float(np.quantile(disagreement, .98)), threshold)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.subplots_adjust(left=.05, right=.96, bottom=.07, top=.88, wspace=.27, hspace=.34)
    fig.suptitle("Soil Confidence Map — Prediction and Model Agreement", fontsize=20, fontweight="bold", y=.97)
    fig.text(.5, .925, "Green = five models agree  •  Red = models disagree  •  × = at least one impossible negative prediction",
             ha="center", fontsize=11,
             bbox={"boxstyle": "round,pad=.4", "facecolor": "#f2f2f2", "edgecolor": "#777777"})
    confidence_norm = Normalize(vmin=0, vmax=dis_hi, clip=True)
    prediction_norm = Normalize(vmin=pred_lo, vmax=pred_hi, clip=True)
    region_bounds = {"North America": (-175, -45, 20, 75), "Europe": (-15, 45, 32, 73)}
    for row, (name, mask) in enumerate(masks.items()):
        xy = geo[mask]; neg = any_negative[mask]
        region_outline(axes[row, 0], region_bounds[name])
        p = axes[row, 0].scatter(xy[:, 0], xy[:, 1], c=ensemble_mean[mask], cmap="viridis",
                                 norm=prediction_norm, s=20, alpha=.85, edgecolors="none", zorder=2)
        axes[row, 0].set_title(f"{name}: mean SOC prediction ({mask.sum()} tiles)")
        cb = fig.colorbar(p, ax=axes[row, 0], fraction=.046, pad=.03); cb.set_label("Mean predicted SOC (g/kg)")

        region_outline(axes[row, 1], region_bounds[name])
        c = axes[row, 1].scatter(xy[:, 0], xy[:, 1], c=disagreement[mask], cmap="RdYlGn_r",
                                 norm=confidence_norm, s=22, alpha=.9, edgecolors="none", zorder=2)
        if np.any(neg):
            axes[row, 1].scatter(xy[neg, 0], xy[neg, 1], marker="x", s=48, c="black", linewidths=1.3,
                                 label="Any seed < 0", zorder=3)
            axes[row, 1].legend(loc="lower left", fontsize=8)
        axes[row, 1].set_title(f"{name}: disagreement / confidence")
        cb = fig.colorbar(c, ax=axes[row, 1], fraction=.046, pad=.03); cb.set_label("Across-model SD (g/kg)")

        axes[row, 2].hist(disagreement[mask], bins=28, color="#597f9f", alpha=.85, edgecolor="white")
        axes[row, 2].axvline(threshold, color="#b32121", linestyle="--", linewidth=2,
                             label=f"LOW threshold = {threshold:.2f}")
        med = np.median(disagreement[mask])
        axes[row, 2].axvline(med, color="black", linewidth=2, label=f"Median = {med:.2f}")
        axes[row, 2].set(xlabel="Across-model SD (g/kg)", ylabel="Number of tiles",
                         title=f"{name}: model-disagreement distribution")
        axes[row, 2].grid(axis="y", alpha=.2); axes[row, 2].legend(fontsize=9)
    fig.savefig(FIGURE, dpi=180, facecolor="white"); plt.close(fig)

    na, eu = summary["north_america"], summary["europe"]
    report = f"""# Soil Confidence Map

## What was run

This inference-only demonstration used all five already-trained SeCo-Eco
ResNet-50 checkpoints from the continental transfer experiment. Each model was
trained independently on the same North America training region with a different
seed. No model was trained or adjusted for this confidence map.

The aligned map covers **{n_na} North America spatial-holdout tiles**
(in-distribution) and **{len(europe)} Europe transfer tiles**
(out-of-distribution). Each plotted point is a tile centroid, not a continuous
wall-to-wall soil raster.

For every tile:

- **Prediction** is the arithmetic mean SOC prediction from the five models.
- **Disagreement** is the sample standard deviation of their five predictions.
- **LOW confidence** means disagreement exceeds **{threshold:.2f} g/kg**, the
  75th percentile measured on the North America holdout. Europe was not used to
  choose this threshold.
- If any model predicts negative SOC, the tile receives a hard black “do not
  trust” marker. Negative values are not clipped or hidden.

## Punchline

| Quantity | North America holdout | Europe transfer |
|---|---:|---:|
| Tiles | {na['n']} | {eu['n']} |
| Median model disagreement | {na['median_disagreement_gkg']:.3f} g/kg | {eu['median_disagreement_gkg']:.3f} g/kg |
| LOW-confidence fraction | {100*na['low_confidence_fraction']:.2f}% | {100*eu['low_confidence_fraction']:.2f}% |
| Any seed predicts impossible negative SOC | {100*na['any_seed_negative_fraction']:.2f}% | {100*eu['any_seed_negative_fraction']:.2f}% |
| Ensemble mean itself is negative | {100*na['ensemble_mean_negative_fraction']:.2f}% | {100*eu['ensemble_mean_negative_fraction']:.2f}% |

Disagreement is an honest empirical uncertainty signal: independently trained
models seeing the same tile should give similar answers where the learned
relationship is stable. Large spread exposes sensitivity to training randomness
and warns that the prediction is not robust. Agreement does **not** prove
accuracy—all five models can share the same bias—but disagreement is direct
evidence not to trust a point estimate.

## Product pitch

> **The only soil map that tells you where to trust it, and where to send ground samples instead.**

## Outputs

- Figure: `{FIGURE}`
- Per-tile predictions: `{TABLE}`
- Machine-readable summary: `{SUMMARY_JSON}`
"""
    REPORT.write_text(report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"WROTE {FIGURE}\nWROTE {TABLE}\nWROTE {REPORT}", flush=True)
    return summary


def orchestrate(batch_size: int, workers: int) -> None:
    missing = [str(p) for p in CHECKPOINTS if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Fewer than five usable checkpoints; missing: {missing}")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)
    procs = []
    for slot in range(5):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(slot)
        log = (OUT / f"seed_{slot}.log").open("w")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(slot),
               "--batch-size", str(batch_size), "--workers", str(workers)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((slot, proc, log))
        print(f"LAUNCHED slot={slot} gpu={slot} pid={proc.pid} checkpoint={CHECKPOINTS[slot]}", flush=True)
    failed = []
    for slot, proc, log in procs:
        rc = proc.wait(); log.close()
        print(f"COMPLETED slot={slot} rc={rc}", flush=True)
        if rc:
            failed.append((slot, rc))
    if failed:
        raise RuntimeError(f"Inference worker failures: {failed}; inspect {OUT}/seed_*.log")
    consolidate()


def main() -> None:
    cfg = args()
    if cfg.consolidate:
        consolidate()
    elif cfg.worker is not None:
        if cfg.worker not in range(5):
            raise ValueError("--worker must be 0..4")
        worker(cfg.worker, cfg.batch_size, cfg.workers)
    else:
        orchestrate(cfg.batch_size, cfg.workers)


if __name__ == "__main__":
    main()
