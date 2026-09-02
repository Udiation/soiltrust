#!/usr/bin/env python3
"""Load the trained checkpoint and predict SOC for a 12-band NumPy tile."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchgeo.models import ResNet50_Weights, resnet50


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tile", type=Path, help=".npy array shaped (12,H,W), reflectance scaled by 10000")
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    a = p.parse_args()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = resnet50(weights=None, in_chans=12, num_classes=1)
    model.load_state_dict(ckpt["model"]); model.eval().cuda()
    x = torch.from_numpy(np.load(a.tile).astype("float32"))
    if x.ndim != 3 or x.shape[0] != 12:
        raise ValueError(f"Expected (12,H,W), got {tuple(x.shape)}")
    x[x == 65535] = 0
    x = TF.resize(x.clamp_(0, 10000).div_(10000), [224, 224], antialias=True)[None].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model(x).float().item()
    prediction = z * ckpt["config"]["target_std"] + ckpt["config"]["target_mean"]
    print(json.dumps({"soil_organic_carbon_g_per_kg": prediction,
                      "bands": ckpt["config"]["bands"],
                      "checkpoint_epoch": ckpt["epoch"]}, indent=2))


if __name__ == "__main__":
    main()
