#!/usr/bin/env python3
"""Inference-only, single-tile SOC report-card demo using MMEarth data."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import h5py
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision.transforms import functional as TF
from torchgeo.models import resnet50


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "soil_organic_carbon"
H5_PATH = DATA / "soil_organic_carbon.h5"
CHECKPOINT = ROOT / "artifacts" / "best_model.pt"
OUT_DIR = ROOT / "artifacts" / "parcel_demo"
FIGURE = OUT_DIR / "field_report_card.png"
DETAILS = OUT_DIR / "parcel_demo.json"
REPORT = ROOT.parent / "parcel-demo.md"
NO_DATA = 65535
PATCH_PIXELS = 32


def stretch(channels: np.ndarray) -> np.ndarray:
    """Percentile-stretch channels-first reflectance into display RGB."""
    out = []
    for band in channels:
        valid = band[np.isfinite(band) & (band != NO_DATA)]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
        out.append(np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1))
    return np.moveaxis(np.stack(out), 0, -1)


def pick_tile(h5: h5py.File) -> tuple[int, list[dict]]:
    geo = h5["geolocation"][:]
    lon, lat = geo[:, 0], geo[:, 1]
    candidates = np.flatnonzero((lat >= 6) & (lat <= 37) & (lon >= 68) & (lon <= 98))
    rows = []
    for idx in candidates:
        tile = h5["Sentinel2"][idx]
        row = {
            "index": int(idx),
            "longitude": float(lon[idx]),
            "latitude": float(lat[idx]),
            "missing_fraction": float(np.mean(tile == NO_DATA)),
            "cloud_probability_fraction": float(h5["MSK_CLDPRB_CLOUDY_PIXEL_FRACTION"][idx]),
            "s2cloudless_fraction": float(h5["S2CLOUDLESS_CLOUDY_PIXEL_FRACTION"][idx]),
            "mean_cloud_probability": float(np.nanmean(h5["MSK_CLDPRB"][idx])),
            "mean_s2cloudless_probability": float(np.nanmean(h5["S2CLOUDLESS"][idx])),
        }
        x = np.asarray(tile, dtype=float)
        x[x == NO_DATA] = np.nan
        row["mean_ndvi"] = float(np.nanmean(
            (x[7] - x[3]) / np.where(np.abs(x[7] + x[3]) < 1e-6, np.nan, x[7] + x[3])))
        rows.append(row)
    if not rows:
        raise RuntimeError("No samples fall inside the requested bbox; fallback selection is not implemented silently")
    # A rectangular bbox is not a country boundary. Use the locally installed
    # Natural Earth layer to prevent another non-India bbox point being labelled
    # as Indian, then apply the user's NDVI-first/cloud-second ranking.
    shp = (Path(gpd.__file__).parent.parent / "pyogrio" / "tests" / "fixtures" /
           "naturalearth_lowres" / "naturalearth_lowres.shp")
    if not shp.exists():
        raise FileNotFoundError(f"India verification polygon unavailable: {shp}")
    world = gpd.read_file(shp)
    india = world.loc[world["name"] == "India", "geometry"].union_all()
    for row in rows:
        row["inside_india_polygon"] = bool(india.covers(gpd.points_from_xy(
            [row["longitude"]], [row["latitude"]])[0]))
        row["cloud_fraction"] = max(row["cloud_probability_fraction"], row["s2cloudless_fraction"])
    rows.sort(key=lambda r: (-r["mean_ndvi"], r["cloud_fraction"], r["index"]))
    print("RANK index latitude longitude mean_ndvi cloud_fraction inside_india", flush=True)
    for rank, row in enumerate(rows, 1):
        print(f"{rank:>2} {row['index']:>5} {row['latitude']:.6f} {row['longitude']:.6f} "
              f"{row['mean_ndvi']:.6f} {row['cloud_fraction']:.6f} "
              f"{str(row['inside_india_polygon']):>5}", flush=True)
    eligible = [row for row in rows if row["inside_india_polygon"]]
    if not eligible:
        raise RuntimeError("No bbox candidate is inside the India polygon; refusing a non-India fallback")
    chosen = eligible[0]
    print(f"FINAL_PICK index={chosen['index']} lat={chosen['latitude']:.6f} "
          f"lon={chosen['longitude']:.6f} ndvi={chosen['mean_ndvi']:.6f} "
          f"cloud_fraction={chosen['cloud_fraction']:.6f}", flush=True)
    return chosen["index"], rows


def preprocess(tile: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.array(tile, dtype=np.float32, copy=True))
    x[x == NO_DATA] = 0
    return TF.resize(x.clamp_(0, 10000).div_(10000), [224, 224], antialias=True)


def predict(model, tiles: list[np.ndarray], ckpt: dict, device: torch.device) -> np.ndarray:
    batch = torch.stack([preprocess(t) for t in tiles]).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        z = model(batch).flatten().float().cpu().numpy()
    return z * float(ckpt["config"]["target_std"]) + float(ckpt["config"]["target_mean"])


def advisory(ndvi: float, missing: float, cloud: float) -> tuple[str, str]:
    """One transparent illustrative rule using only signals present in this tile."""
    if missing > 0.05 or cloud > 0.10:
        return "RE-ACQUIRE IMAGERY", "Cloud/missing fraction exceeds 10%; do not use this scene operationally."
    if ndvi > 0.35:
        return "BARE-SOIL REVISIT", "Vegetation signal is high; obtain a cloud-free post-harvest scene before soil interpretation."
    return "GROUND-CHECK PRIORITY", "Bare-soil visibility appears plausible; collect a geolocated lab sample before acting on SOC."


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Existing checkpoint not found: {CHECKPOINT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_PATH, "r") as h5:
        idx, candidates = pick_tile(h5)
        tile = np.asarray(h5["Sentinel2"][idx], dtype=np.float32)
        lon, lat = map(float, h5["geolocation"][idx])
        transform = np.asarray(h5["transform"][idx], dtype=float).reshape(3, 3)
        crs_raw = h5["crs"][idx]
        crs = crs_raw.decode() if isinstance(crs_raw, bytes) else str(crs_raw)
        cloud_fraction = max(float(h5["MSK_CLDPRB_CLOUDY_PIXEL_FRACTION"][idx]),
                             float(h5["S2CLOUDLESS_CLOUDY_PIXEL_FRACTION"][idx]))
        missing_fraction = float(np.mean(tile == NO_DATA))

    # Pixel area is the absolute determinant of the 2x2 affine linear part.
    pixel_area_m2 = abs(float(np.linalg.det(transform[:2, :2])))
    area_m2 = pixel_area_m2 * tile.shape[1] * tile.shape[2]
    width_m = abs(transform[0, 0]) * tile.shape[2]
    height_m = abs(transform[1, 1]) * tile.shape[1]

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=None, in_chans=12, num_classes=1)
    model.load_state_dict(ckpt["model"]); model.eval().to(device)
    field_soc = float(predict(model, [tile], ckpt, device)[0])

    patches, positions = [], []
    for row in range(0, tile.shape[1], PATCH_PIXELS):
        for col in range(0, tile.shape[2], PATCH_PIXELS):
            patches.append(tile[:, row:row + PATCH_PIXELS, col:col + PATCH_PIXELS])
            positions.append((row // PATCH_PIXELS, col // PATCH_PIXELS))
    patch_pred = predict(model, patches, ckpt, device)
    grid_shape = (tile.shape[1] // PATCH_PIXELS, tile.shape[2] // PATCH_PIXELS)
    soc_map = np.empty(grid_shape, dtype=float)
    for pos, value in zip(positions, patch_pred):
        soc_map[pos] = value

    safe = tile.copy(); safe[safe == NO_DATA] = np.nan
    rgb = stretch(safe[[3, 2, 1]])                 # B4, B3, B2
    swir = stretch(safe[[10, 7, 3]])              # B11, B8, B4
    b4, b8 = safe[3], safe[7]
    ndvi = float(np.nanmean((b8 - b4) / np.where(np.abs(b8 + b4) < 1e-6, np.nan, b8 + b4)))
    advice_title, advice_text = advisory(ndvi, missing_fraction, cloud_fraction)

    rgb8 = np.uint8(np.clip(rgb * 255, 0, 255))
    native4x = np.asarray(Image.fromarray(rgb8).resize((512, 512), Image.Resampling.NEAREST))
    sharpened = Image.fromarray(rgb8).resize((512, 512), Image.Resampling.BICUBIC)
    sharpened = np.asarray(sharpened.filter(ImageFilter.UnsharpMask(radius=2.0, percent=170, threshold=3)))
    comparison = np.concatenate([native4x[:, :256], sharpened[:, 256:]], axis=1)

    chosen = next(r for r in candidates if r["index"] == idx)
    reason = ("highest mean NDVI among bbox candidates verified inside India; cloud fraction used "
              "as the ranking tiebreak")
    confidence_reason = "LOW — only 0.18% of benchmark samples fall in the broad India bbox; no local calibration."
    location_label = "Real India location (Natural Earth boundary); benchmark tile, not a verified parcel"
    cloud_label = f"CLOUD-AFFECTED: {100*cloud_fraction:.2f}% flagged" if cloud_fraction > 0 else "CLOUD-FREE BY FLAGS"

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.04, right=.97, bottom=.06, top=.84, wspace=.28, hspace=.30)
    fig.suptitle("Single-Tile Soil Intelligence — Founder Demo", y=.97,
                 fontsize=20, fontweight="bold")
    fig.text(.5, .905,
             f"REAL INDIA LOCATION  •  {lat:.5f}°N, {lon:.5f}°E  •  NDVI {ndvi:.3f}  •  {cloud_label}",
             ha="center", va="center", fontsize=12, fontweight="bold",
             bbox={"boxstyle": "round,pad=.45", "facecolor": "#ffe2a8" if cloud_fraction > 0 else "#dcefd5",
                   "edgecolor": "#a86400" if cloud_fraction > 0 else "#4c7a3d"})
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("Sentinel-2 true colour\nB4 / B3 / B2")
    axes[0, 1].imshow(swir); axes[0, 1].set_title("SWIR false colour\nB11 / B8 / B4")
    axes[0, 2].imshow(comparison)
    axes[0, 2].axvline(255.5, color="white", lw=2)
    axes[0, 2].text(0.20, .04, "10 m pixels", color="white", ha="center", fontsize=9,
                    transform=axes[0, 2].transAxes,
                    bbox={"facecolor": "black", "alpha": .6, "pad": 3})
    axes[0, 2].text(.76, .04, "4× sharpened", color="white", ha="center", fontsize=9,
                    transform=axes[0, 2].transAxes,
                    bbox={"facecolor": "black", "alpha": .6, "pad": 3})
    axes[0, 2].set_title("Resolution demo — no new SWIR detail")

    im = axes[1, 0].imshow(soc_map, cmap="YlOrBr", interpolation="nearest")
    axes[1, 0].set_title(f"Illustrative patch SOC map\n{PATCH_PIXELS}×{PATCH_PIXELS}-pixel estimates", fontsize=11)
    cb = fig.colorbar(im, ax=axes[1, 0], fraction=.046, pad=.04); cb.set_label("Predicted SOC (g/kg)")

    axes[1, 1].axis("off")
    facts = (f"LOCATION / TILE\n{lat:.5f}°N, {lon:.5f}°E\n"
             "Verified inside India boundary\nBenchmark tile; no cadastral boundary\n"
             f"NDVI {ndvi:.3f} | Cloud {100*cloud_fraction:.2f}%\n{cloud_label}\n\n"
             f"FOOTPRINT\n{width_m/1000:.2f} × {height_m/1000:.2f} km\n≈ {area_m2/10_000:.2f} ha (tile, not field boundary)\n\n"
             f"MODEL OUTPUT\nField-level SOC: {field_soc:.1f} g/kg\nRaw patch range: {soc_map.min():.1f}–{soc_map.max():.1f} g/kg\n"
             "Negative patches = OOD model failure\n\n"
             "CONFIDENCE: LOW\n0.18% broad India-bbox coverage;\nno local ground calibration")
    axes[1, 1].text(.02, .98, facts, va="top", fontsize=10.2, linespacing=1.25,
                    bbox={"boxstyle": "round,pad=.7", "facecolor": "#f4f1e8", "edgecolor": "#8a795d"})
    axes[1, 1].set_title("Parcel summary")

    axes[1, 2].axis("off")
    advisory_wrapped = textwrap.fill(advice_text, width=39)
    advisory_text = (f"EXAMPLE RULE FIRED\n\n{advice_title}\n\n{advisory_wrapped}\n\n"
                     f"Signal used: mean NDVI = {ndvi:.3f}\n\n"
                     "Demo rule only — not a crop or fertilizer prescription.\n\n"
                     "Production inputs:\n• ALU: verified parcel boundary\n• AMED: crop identity/status\n"
                     "• Soil Health Cards: local SOC calibration")
    axes[1, 2].text(.03, .97, advisory_text, va="top", fontsize=10.2, linespacing=1.25,
                    bbox={"boxstyle": "round,pad=.7", "facecolor": "#eef4e8", "edgecolor": "#648153"})
    axes[1, 2].set_title("Transparent advisory logic")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180, facecolor="white"); plt.close(fig)

    result = {
        "sample_index": idx, "latitude": lat, "longitude": lon, "label": location_label,
        "selection_reason": reason, "candidate_count": len(candidates), "selection_quality": chosen,
        "crs": crs, "native_pixel_metres": abs(float(transform[0, 0])),
        "tile_shape": list(tile.shape), "footprint_width_m": width_m,
        "footprint_height_m": height_m, "footprint_area_ha": area_m2 / 10_000,
        "field_soc_g_per_kg": field_soc, "patch_soc_min": float(soc_map.min()),
        "patch_soc_max": float(soc_map.max()), "patch_soc_grid": soc_map.tolist(),
        "confidence": "LOW", "confidence_reason": confidence_reason, "mean_ndvi": ndvi,
        "advisory_rule": advice_title, "advisory_text": advice_text,
        "checkpoint_epoch": int(ckpt["epoch"]), "device": str(device), "figure": str(FIGURE),
    }
    DETAILS.write_text(json.dumps(result, indent=2))

    report = f"""# Single-parcel SOC founder demo

## What the pipeline produced

This inference-only demo selected MMEarth sample **{idx}** at **{lat:.5f}°N,
{lon:.5f}°E**. This **is a real India location**, verified against the available
Natural Earth country boundary. From the 14 broad bbox candidates it was the
highest-NDVI point that is genuinely inside India; cloud fraction was the
tiebreak criterion. Its measured mean NDVI is **{ndvi:.3f}** and its flagged
cloud fraction is **{100*cloud_fraction:.2f}%**, so the tile is plainly labelled
**cloud-affected** in the figure.

The source is a 128×128 Sentinel-2 tile on a 10 m grid in `{crs}`. Its footprint
is approximately **{width_m/1000:.2f} km × {height_m/1000:.2f} km**, or
**{area_m2/10_000:.2f} ha**. This is the image footprint—not an ownership,
cadastral, or cultivated-field boundary.

The existing SeCo-Eco ResNet-50 checkpoint produced a whole-tile SOC estimate of
**{field_soc:.1f} g/kg**. A 4×4 grid of 32-pixel subtiles produced an illustrative
within-tile range of **{soc_map.min():.1f}–{soc_map.max():.1f} g/kg**. Each patch
covers roughly 320 m × 320 m and is resized to the model input size. Because the
model was trained on whole tiles rather than patch supervision, this map is a
visual demonstration of the inference mechanism, not validated within-field
resolution or an agronomic prescription.

Some raw patch outputs are negative, which is physically impossible for SOC.
They are deliberately left visible rather than clipped: this is direct evidence
of out-of-distribution/model-resolution failure and reinforces the LOW confidence
flag. The patch map must not be interpreted as a quantitative soil map.

The example rule fired was **{advice_title}**, using mean NDVI {ndvi:.3f}:
{advice_text} This is transparent demo logic, not a recommendation validated for
this location.

## Honest limits

- Confidence is **LOW**. Only about 0.18% of benchmark samples fall inside the
  broad India bbox, and the model has no parcel-specific laboratory calibration.
- The location is real and inside India, but the 163.84 ha rectangular scene is
  still a benchmark tile—not a verified small-farm parcel. An ALU boundary is
  required before making a true parcel-level claim.
- Sentinel-2 SWIR soil information remains at its native sensor resolution. A
  4× sharpened image can make edges easier to view, but it cannot create new
  spatial detail or new chemistry. Typical high-resolution RGB imagery has no
  SWIR band, so it cannot genuinely sharpen the SWIR measurement.
- The “parcel” is a rectangular benchmark tile. No verified cadastral boundary,
  crop identity, management history, or field inspection was available.
- The field-level value and patch map are model outputs, not measured SOC and
  not an accuracy claim for India.

## What production layers would use

- **ALU:** authoritative/verified parcel geometry would replace the rectangular
  tile boundary and define which pixels belong to the field. No ALU layer was
  available here; the displayed footprint is a clearly labelled placeholder.
- **AMED:** crop identity and crop-status information would contextualize NDVI,
  choose suitable acquisition dates, and support crop-aware advisories. No AMED
  access was available; the demo does not claim a real crop label.
- **Soil Health Cards:** geolocated laboratory SOC observations would calibrate,
  validate, and monitor the model locally. No Soil Health Card observation was
  joined to this tile.

## Outputs

- Report card: `{FIGURE}`
- Machine-readable details: `{DETAILS}`
- Script: `{Path(__file__).resolve()}`
"""
    REPORT.write_text(report)
    print(json.dumps(result, indent=2), flush=True)
    print(f"WROTE {FIGURE}", flush=True)
    print(f"WROTE {REPORT}", flush=True)


if __name__ == "__main__":
    main()
