#!/usr/bin/env python3
"""DDP fine-tuning for MMEarth-Bench Sentinel-2 -> soil organic carbon.

This intentionally uses ordinary eager PyTorch. It does not import or enable
FlashAttention, FlashInfer, xFormers, or scaled-dot-product attention kernels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF
from torchgeo.models import ResNet50_Weights, resnet50


BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
WEIGHT_NAME = "SENTINEL2_ALL_SECO_ECO"
NO_DATA = 65535


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/soil_organic_carbon"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--log-file", type=Path, default=Path("../overnight-log2.md"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--early-stopping", type=int, default=8)
    return p.parse_args()


def ddp_setup() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def prepare_arrays(data_dir: Path, rank: int, world: int) -> tuple[Path, Path]:
    """Extract only the two needed HDF5 arrays once for fast mmap DDP reads."""
    image_path = data_dir / "sentinel2.npy"
    target_path = data_dir / "soc_gkg.npy"
    if rank == 0 and (not image_path.exists() or not target_path.exists()):
        h5_path = data_dir / "soil_organic_carbon.h5"
        with h5py.File(h5_path, "r") as f:
            images = f["Sentinel2"][:]
            targets = f["soil_organic_carbon"][:].astype("float32").reshape(-1)
        np.save(image_path, images)
        np.save(target_path, targets)
        del images, targets
    barrier(world)
    return image_path, target_path


class SOCDataset(Dataset):
    def __init__(self, image_path: Path, target_path: Path, indices: list[int], train: bool):
        self.images = np.load(image_path, mmap_mode="r")
        self.targets = np.load(target_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        # MMEarth Sentinel-2 is uint16 surface reflectance scaled by 10,000.
        x = torch.from_numpy(np.array(self.images[idx], dtype=np.float32, copy=True))
        x[x == NO_DATA] = 0
        x = x.clamp_(0, 10000).div_(10000.0)
        # The selected TorchGeo SeCo-Eco checkpoint expects 224x224 and /10000.
        x = TF.resize(x, [224, 224], antialias=True)
        if self.train:
            if random.random() < 0.5:
                x = TF.hflip(x)
            if random.random() < 0.5:
                x = TF.vflip(x)
            x = torch.rot90(x, random.randrange(4), dims=(-2, -1))
        return x, torch.tensor(float(self.targets[idx]), dtype=torch.float32)


def make_loader(ds: Dataset, batch: int, workers: int, world: int, rank: int, shuffle: bool):
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False) if world > 1 else None
    return DataLoader(ds, batch_size=batch, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      drop_last=shuffle), sampler


@torch.no_grad()
def evaluate(model, loader, target_mean: float, target_std: float, device, world: int) -> dict[str, float]:
    model.eval()
    # n, abs error, squared error, y sum, y squared sum
    acc = torch.zeros(5, dtype=torch.float64, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_z = model(x).flatten()
        pred = pred_z.float() * target_std + target_mean
        err = pred - y
        acc += torch.stack([torch.tensor(y.numel(), device=device), err.abs().sum(), err.square().sum(), y.sum(), y.square().sum()]).double()
    if world > 1:
        dist.all_reduce(acc)
    n, sae, sse, sy, sy2 = acc.tolist()
    denom = max(sy2 - sy * sy / n, 1e-12)
    return {"n": int(n), "mae": sae / n, "rmse": math.sqrt(sse / n), "r2": 1.0 - sse / denom}


def plot_history(rows: list[dict], path: Path) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train Huber (z-score)")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[0].grid(alpha=.25); axes[0].legend()
    axes[1].plot(epochs, [r["val_mae"] for r in rows], marker="o", label="validation MAE")
    axes[1].plot(epochs, [r["val_rmse"] for r in rows], marker="o", label="validation RMSE")
    axes[1].set(xlabel="Epoch", ylabel="SOC (g/kg)", title="Held-out validation error")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    cfg = args()
    rank, world, local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)
    random.seed(cfg.seed + rank); np.random.seed(cfg.seed + rank); torch.manual_seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    image_path, target_path = prepare_arrays(cfg.data_dir, rank, world)
    split = json.loads((cfg.data_dir / "soil_organic_carbon_split_data.json").read_text())
    train_idx = split["train_100%_indices"]
    val_idx = split["val_indices"]
    geo_idx = split["geographic_test_indices"]
    targets = np.load(target_path, mmap_mode="r")
    target_mean = float(np.mean(targets[train_idx], dtype=np.float64))
    target_std = float(np.std(targets[train_idx], dtype=np.float64))

    train_ds = SOCDataset(image_path, target_path, train_idx, True)
    # Shard evaluation without DistributedSampler padding so every held-out tile
    # contributes exactly once to the reported metrics.
    val_ds = SOCDataset(image_path, target_path, val_idx[rank::world], False)
    geo_ds = SOCDataset(image_path, target_path, geo_idx[rank::world], False)
    train_loader, train_sampler = make_loader(train_ds, cfg.batch_size, cfg.workers, world, rank, True)
    val_loader, _ = make_loader(val_ds, cfg.batch_size, cfg.workers, 1, 0, False)
    geo_loader, _ = make_loader(geo_ds, cfg.batch_size, cfg.workers, 1, 0, False)

    weights = ResNet50_Weights.SENTINEL2_ALL_SECO_ECO
    assert list(weights.meta["bands"]) == BANDS, (weights.meta["bands"], BANDS)
    model = resnet50(weights=weights, num_classes=1).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 50)

    if rank == 0:
        metadata = {
            "dataset": "MMEarth-Bench soil_organic_carbon", "target": "SOC_g_per_kg",
            "bands": BANDS, "backbone": "torchgeo resnet50", "weights": WEIGHT_NAME,
            "pretraining": weights.meta, "world_size": world, "per_gpu_batch": cfg.batch_size,
            "epochs_requested": cfg.epochs, "target_mean": target_mean, "target_std": target_std,
            "split_counts": {"train": len(train_idx), "val": len(val_idx), "geographic_test": len(geo_idx)},
        }
        (cfg.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str))
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    history: list[dict] = []
    best_mae, stale = float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        model.train(); loss_sum = torch.zeros(2, dtype=torch.float64, device=device)
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            yz = (y - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x).flatten()
                loss = F.huber_loss(pred, yz, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += torch.tensor([loss.detach().double() * y.numel(), y.numel()], device=device)
        if world > 1: dist.all_reduce(loss_sum)
        scheduler.step()
        val = evaluate(model, val_loader, target_mean, target_std, device, world)
        elapsed = time.time() - started
        row = {"epoch": epoch, "train_loss": float(loss_sum[0] / loss_sum[1]),
               "val_mae": val["mae"], "val_rmse": val["rmse"], "val_r2": val["r2"],
               "lr": scheduler.get_last_lr()[0], "epoch_seconds": elapsed}
        if rank == 0:
            history.append(row)
            with (cfg.output_dir / "metrics.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerows(history)
            plot_history(history, cfg.output_dir / "training_curve.png")
            print("METRIC " + json.dumps(row), flush=True)
            with cfg.log_file.open("a") as f:
                f.write(f"- Epoch {epoch}: train Huber={row['train_loss']:.5f}, "
                        f"val MAE={row['val_mae']:.3f} g/kg, val RMSE={row['val_rmse']:.3f} g/kg, "
                        f"val R²={row['val_r2']:.4f}, time={elapsed:.1f}s.\n")
            if val["mae"] < best_mae:
                best_mae, stale = val["mae"], 0
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "val": val, "config": metadata}, cfg.output_dir / "best_model.pt")
            else:
                stale += 1
        # Broadcast early-stopping decision.
        stop = torch.tensor([stale >= cfg.early_stopping if rank == 0 else False], device=device)
        if world > 1: dist.broadcast(stop, src=0)
        if stop.item(): break

    # Evaluate the selected best checkpoint on the benchmark geographic holdout.
    barrier(world)
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint = torch.load(cfg.output_dir / "best_model.pt", map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["model"])
    geographic = evaluate(model, geo_loader, target_mean, target_std, device, world)
    if rank == 0:
        y_val = np.asarray(targets[val_idx], dtype=np.float64)
        baseline = {"mae": float(np.mean(np.abs(y_val - target_mean))),
                    "rmse": float(np.sqrt(np.mean((y_val - target_mean) ** 2)))}
        final = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["val"],
                 "geographic_test": geographic, "train_mean_baseline_on_validation": baseline}
        (cfg.output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
        with cfg.log_file.open("a") as f:
            f.write("\n## Final measured result\n\n")
            f.write(f"- Best epoch: {final['best_epoch']}\n")
            f.write(f"- Validation: MAE {final['validation']['mae']:.3f} g/kg; RMSE {final['validation']['rmse']:.3f} g/kg; R² {final['validation']['r2']:.4f}.\n")
            f.write(f"- Geographic holdout: MAE {geographic['mae']:.3f} g/kg; RMSE {geographic['rmse']:.3f} g/kg; R² {geographic['r2']:.4f}.\n")
            f.write(f"- Train-mean validation baseline: MAE {baseline['mae']:.3f} g/kg; RMSE {baseline['rmse']:.3f} g/kg.\n")
        print("FINAL " + json.dumps(final), flush=True)
    barrier(world)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
