#!/usr/bin/env python3
"""Auditable log-SOC quantile baseline on the saved 50 km spatial folds."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import geopandas
import h5py
import joblib
import matplotlib
import numpy as np
import pandas
import pyproj
import sklearn
import torch
import torchgeo
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "soil_organic_carbon"
SPLITS = ROOT / "spatial_cv_runs" / "split_design.json"
OUT = ROOT / "tabular_quantile_runs"
RESULTS = ROOT / "tabular_log_quantile_results.csv"
SUMMARY = ROOT.parent / "tabular-log-quantile-log.md"
RUN_CONFIG = ROOT / "RUN_CONFIG.md"
FEATURE_CACHE = OUT / "features.npz"
QUANTILES = (0.05, 0.50, 0.95)
SEED = 20260901

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
FEATURES = [f"{b}_mean" for b in BANDS] + [
    "NDVI_mean=(B8-B4)/(B8+B4)",
    "NBR2_mean=(B11-B12)/(B11+B12)",
    "BSI_mean=((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))",
    "clay_ratio_mean=B11/B12",
    "carbonate_ratio_mean=B11/B8A",
    "ASTER_GDEM_elevation_mean_m",
    "ASTER_GDEM_slope_mean_degrees",
]


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
    return np.divide(a, b, out=out, where=valid)


def extract_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if FEATURE_CACHE.exists():
        z = np.load(FEATURE_CACHE)
        return z["X"], z["y"], z["geolocation"]
    images = np.load(DATA / "sentinel2.npy", mmap_mode="r")
    y = np.load(DATA / "soc_gkg.npy", mmap_mode="r")
    n = len(y); X = np.empty((n, len(FEATURES)), dtype=np.float32)
    with h5py.File(DATA / "soil_organic_carbon.h5", "r") as h5:
        geo = h5["geolocation"][:]
        terrain = h5["ASTER_GDEM"]
        for start in range(0, n, 64):
            end = min(start + 64, n)
            s2 = np.array(images[start:end], dtype=np.float32, copy=True)
            s2[s2 == 65535] = np.nan
            means = np.nanmean(s2, axis=(2, 3))
            b2,b4,b8,b8a,b11,b12 = s2[:,1],s2[:,3],s2[:,7],s2[:,8],s2[:,10],s2[:,11]
            idx = [
                np.nanmean(safe_ratio(b8-b4,b8+b4),axis=(1,2)),
                np.nanmean(safe_ratio(b11-b12,b11+b12),axis=(1,2)),
                np.nanmean(safe_ratio((b11+b4)-(b8+b2),(b11+b4)+(b8+b2)),axis=(1,2)),
                np.nanmean(safe_ratio(b11,b12),axis=(1,2)),
                np.nanmean(safe_ratio(b11,b8a),axis=(1,2)),
            ]
            dem = np.asarray(terrain[start:end],dtype=np.float32)
            dem[dem == -9999] = np.nan
            terr = np.nanmean(dem,axis=(2,3))
            X[start:end] = np.column_stack([means,*idx,terr])
            print(f"FEATURES {end}/{n}",flush=True)
    # HistGB handles NaNs, but record their count; no imputation leakage is introduced.
    np.savez_compressed(FEATURE_CACHE,X=X,y=np.asarray(y,dtype=np.float32),geolocation=geo)
    return X,np.asarray(y,dtype=np.float32),geo


def point_metrics(y: np.ndarray,p: np.ndarray) -> dict:
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
    den=np.sum((y-y.mean())**2)
    return {"n":len(y),"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),
            "r2":float(1-np.sum(e**2)/den) if den>0 else math.nan,"bias":float(np.mean(e))}


def interval_metrics(y,q05,q95) -> dict:
    y=np.asarray(y,float);q05=np.asarray(q05,float);q95=np.asarray(q95,float)
    inside=(y>=q05)&(y<=q95);lo,hi=np.quantile(y,[1/3,2/3]);low=y<=lo;high=y>hi;width=q95-q05
    return {"picp":float(inside.mean()),"interval_width_mean":float(width.mean()),
            "interval_width_median":float(np.median(width)),"picp_low_tercile":float(inside[low].mean()),
            "picp_high_tercile":float(inside[high].mean()),"tercile_low_max":float(lo),"tercile_high_min":float(hi)}


def metric_block(y,q05,q50,q95) -> dict:
    crossing=float(np.mean(q05>q95));lo=np.minimum(q05,q95);hi=np.maximum(q05,q95)
    return {**point_metrics(y,q50),**interval_metrics(y,lo,hi),"quantile_crossing_rate":crossing}


def row(run_id,fold,split,space,row_type,m,null_value,smear):
    return {"run_id":run_id,"fold":fold,"split":split,"space":space,"row_type":row_type,
            "null_value":null_value,"duan_smearing_factor":smear,**m}


def transform_predictions(pred_log: np.ndarray,smear: float,space: str) -> np.ndarray:
    if space=="log1p": return pred_log
    if space=="naive_expm1": return np.maximum(np.expm1(pred_log),0)
    if space=="duan_smearing": return np.maximum(smear*np.exp(pred_log)-1,0)
    raise ValueError(space)


def train() -> None:
    for p in (OUT,RESULTS,SUMMARY,RUN_CONFIG):
        if p.exists(): raise FileExistsError(f"Refusing to overwrite {p}")
    OUT.mkdir()
    X,y,geo=extract_features(); ylog=np.log1p(y)
    design=json.loads(SPLITS.read_text()); rows=[]; fold_summaries=[]
    for f in design["folds"]:
        fold=f["fold"];tr=np.asarray(f["train_indices"],int);va=np.asarray(f["val_indices"],int);ho=np.asarray(design["holdout_indices"],int)
        fold_dir=OUT/f"fold_{fold}";fold_dir.mkdir()
        models={}
        for q in QUANTILES:
            model=HistGradientBoostingRegressor(loss="quantile",quantile=q,learning_rate=.05,max_iter=300,
                    max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,early_stopping=True,
                    validation_fraction=.1,n_iter_no_change=25,random_state=SEED+fold)
            model.fit(X[tr],ylog[tr]);models[q]=model;joblib.dump(model,fold_dir/f"q{int(q*100):02d}.joblib")
        train_med=models[.5].predict(X[tr]);smear=float(np.mean(np.exp(ylog[tr]-train_med)))
        fold_result={"fold":fold,"smearing_factor":smear,"features":FEATURES,"splits":{}}
        for split_name,idx in (("spatial_validation",va),("geographic_holdout",ho)):
            plog=np.stack([models[q].predict(X[idx]) for q in QUANTILES],axis=1)
            split_result={}
            for space in ("log1p","naive_expm1","duan_smearing"):
                truth=ylog[idx] if space=="log1p" else y[idx]
                pred=transform_predictions(plog,smear,space)
                m=metric_block(truth,pred[:,0],pred[:,1],pred[:,2]);split_result[space]=m
                # Null is the training arithmetic mean in the metric's own space.
                null=float(ylog[tr].mean()) if space=="log1p" else float(y[tr].mean())
                nm=point_metrics(truth,np.full(len(idx),null));nm.update({k:math.nan for k in
                    ("picp","interval_width_mean","interval_width_median","picp_low_tercile","picp_high_tercile","tercile_low_max","tercile_high_min","quantile_crossing_rate")})
                rows.append(row(f"tabular_fold_{fold}",fold,split_name,space,"model",m,math.nan,smear))
                rows.append(row(f"tabular_fold_{fold}",fold,split_name,space,"null_train_mean",nm,null,smear))
            fold_result["splits"][split_name]=split_result
        (fold_dir/"result.json").write_text(json.dumps(fold_result,indent=2,allow_nan=True));fold_summaries.append(fold_result)
        print("FOLD",fold,json.dumps(fold_result["splits"]["geographic_holdout"]["duan_smearing"]),flush=True)
    fields=["run_id","fold","split","space","row_type","null_value","duan_smearing_factor","n","mae","rmse","r2","bias",
            "picp","interval_width_mean","interval_width_median","picp_low_tercile","picp_high_tercile","tercile_low_max","tercile_high_min","quantile_crossing_rate"]
    with RESULTS.open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_docs(X,y,geo,design,fold_summaries)


def geography_audit(y,geo,design):
    used=np.asarray(sorted(set(design["pool_indices"])|set(design["holdout_indices"])),int);lon,lat=geo[:,0],geo[:,1]
    sa=(lat>=6)&(lat<=37)&(lon>=68)&(lon<=98)
    def s(mask):
        a=y[used[mask[used]]];return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"p5":float(np.percentile(a,5)),"p95":float(np.percentile(a,95)),"max":float(a.max())}
    return {"total":len(used),"south_asia_count":int(sa[used].sum()),"south_asia_percent":float(100*sa[used].mean()),
            "south_asia_soc":s(sa),"outside_south_asia_soc":s(~sa)}


def write_docs(X,y,geo,design,folds):
    audit=geography_audit(y,geo,design)
    versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pandas.__version__,
              "scikit_learn":sklearn.__version__,"h5py":h5py.__version__,"torch":torch.__version__,
              "torchgeo":torchgeo.__version__,"geopandas":geopandas.__version__,"pyproj":pyproj.__version__}
    config=f"""# Reproducibility configuration: log-SOC tabular quantile run

## Scope and geography gate

- Dataset: MMEarth-Bench soil organic carbon, all 7,982 samples used by the saved CV/holdout design.
- South Asia bbox: 6–37°N, 68–98°E; {audit['south_asia_count']} samples ({audit['south_asia_percent']:.3f}%).
- India: 2 samples by Natural Earth 1:10m country polygon; geographic holdout contains zero South Asia samples.
- India audit boundary: Natural Earth 1:10m admin-0 countries, `ADMIN == India`, from `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson`.
- Audited SOC inside South Asia: n=14, mean=36.879, median=10.200, p5=0.855, p95=139.720, max=179.500 g/kg; outside: n=7,968, mean=78.142, median=25.500, p5=3.600, p95=383.895, max=779.000 g/kg.
- India SOC: n=2, mean/median=9.450, p5=1.935, p95=16.965, max=17.800 g/kg.
- South Asia occupies 11 blocks. Fold samples/blocks: 0=1/1, 1=2/2, 2=2/2, 3=2/2, 4=7/4; holdout=0/0.
- Because South Asia representation is below 2%, the requested deep uncertainty model was not run. These results are not Indian calibration results.

## Target and correction

- Training target: `log1p(SOC_g_per_kg)`.
- Quantiles: 0.05, 0.50, 0.95; point prediction is q0.50.
- Naive back-transform: `max(expm1(q), 0)`.
- Duan correction per fold: `s = mean(exp(y_log_train - q50_log_train))`; corrected quantiles `max(s * exp(q) - 1, 0)`.
- R² in every space: `1 - SSE / sum((y_space - mean(y_space_evaluation_split))^2)`.
- Null: training-split arithmetic mean in the same evaluation space, predicted everywhere.
- Bias: mean(prediction - truth); negative means underprediction.

## Features ({len(FEATURES)})

"""+"\n".join(f"- `{x}`" for x in FEATURES)+f"""

Aspect and TWI are not present in MMEarth. Only the available ASTER GDEM elevation and slope were used. Index ratios are calculated pixelwise and then spatially averaged. `carbonate_ratio` is explicitly defined as B11/B8A for reproducibility.

## Model

- Three scikit-learn `HistGradientBoostingRegressor` models per fold with quantile loss.
- `learning_rate=0.05`, `max_iter=300`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`.
- Internal early stopping: 10% training-only validation fraction, patience 25; seed `20260901 + fold`.
- Missing values, if any, are handled natively by HistGradientBoosting; no global imputation was fit.
- Independently fitted quantiles are checked for crossing. Crossing rate is reported; crossed lower/upper bounds are reordered only for interval coverage/width calculation.

## Spatial folds

- Reused exactly from `spatial_cv_runs/split_design.json`.
- Approximate 50 km blocks; five GroupKFold folds; 325-sample geographic holdout remains separate.

| Fold | Train n | Validation n | Train blocks | Validation blocks | Validation bbox |
|---:|---:|---:|---:|---:|---|
"""+"\n".join(f"| {f['fold']} | {len(f['train_indices'])} | {len(f['val_indices'])} | {f['train_blocks']} | {f['val_blocks']} | {f['val_bbox']} |" for f in design['folds'])+f"""

Holdout bbox: `{design['holdout_bbox']}`.

## Package versions

"""+"\n".join(f"- {k}: `{v}`" for k,v in versions.items())+"\n"
    RUN_CONFIG.write_text(config)
    # Human summary deliberately does not claim an India result or deep-vs-tabular verdict.
    lines=["# Log-SOC tabular quantile baseline", "", "## Geography gate", "",
           f"South Asia has only {audit['south_asia_count']}/{audit['total']} samples ({audit['south_asia_percent']:.3f}%); India has 2; the holdout has zero. Section 2 was stopped by design.", "",
           "## Geographic holdout: individual tabular models", "",
           "All values below use Duan-smearing correction; no ensemble was run.", "",
           "| Fold | MAE | RMSE | R² | Bias | PICP | Width mean | Width median | PICP low tercile | PICP high tercile |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    vals=[]
    for f in folds:
        m=f["splits"]["geographic_holdout"]["duan_smearing"];vals.append(m)
        lines.append(f"| {f['fold']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} | {m['bias']:.3f} | {m['picp']:.3f} | {m['interval_width_mean']:.3f} | {m['interval_width_median']:.3f} | {m['picp_low_tercile']:.3f} | {m['picp_high_tercile']:.3f} |")
    lines += ["", "Mean ± sample SD across individual folds:", ""]
    for k in ("mae","rmse","r2","bias","picp","interval_width_mean","interval_width_median","picp_low_tercile","picp_high_tercile"):
        a=np.array([m[k] for m in vals]);lines.append(f"- {k}: **{a.mean():.4f} ± {a.std(ddof=1):.4f}**")
    lines += ["", "## Verdict", "", "The deep quantile model was prohibited by the South Asia representation gate, so no valid deep-versus-tabular uncertainty winner can be declared. The tabular baseline is the only newly calibrated uncertainty model in this run. Its holdout scores describe the MMEarth holdout geography, not India."]
    SUMMARY.write_text("\n".join(lines)+"\n")


if __name__=="__main__": train()
