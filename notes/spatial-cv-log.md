# Spatial block cross-validation run

## Spatial split preflight

- Block target: **50 km × 50 km** (approximately).
- Choice: 50 km was user-specified as the target scale for suppressing local spatial autocorrelation.
- Construction: latitude bands of 50/111.32 degrees; longitude cell width corrected by cosine of band midpoint.
- R² convention: R2 = 1 - SSE / sum((y - mean(y_evaluation_split))^2); fold/split mean denominator for every row.
- Dataloader workers: **2 per job**, 16 total across eight concurrent jobs.

| Fold | Train samples | Validation samples | Train blocks | Validation blocks | Validation bbox (lon_min, lat_min, lon_max, lat_max) |
|---:|---:|---:|---:|---:|---|
| 0 | 6125 | 1532 | 2287 | 571 | (-157.050, -42.798, 153.376, 72.950) |
| 1 | 6125 | 1532 | 2287 | 571 | (-157.731, -63.871, 172.731, 69.487) |
| 2 | 6126 | 1531 | 2286 | 572 | (-156.731, -46.590, 169.251, 73.000) |
| 3 | 6126 | 1531 | 2286 | 572 | (-156.463, -42.768, 175.449, 72.983) |
| 4 | 6126 | 1531 | 2286 | 572 | (-161.958, -42.792, 153.219, 72.900) |

Geographic holdout: **325 samples in 186 blocks**; bbox (-16.861, -34.236, 45.370, 36.923).

Whole-block disjointness assertions passed for all folds.

## Backbone availability

- seco_eco: available
- ssl4eo_moco: available
- imagenet: available

## GPU job launch
- GPU 0: `cv_fold_0`, fold 0, backbone seco_eco, seed 20260901
- GPU 1: `cv_fold_1`, fold 1, backbone seco_eco, seed 20260902
- GPU 2: `cv_fold_2`, fold 2, backbone seco_eco, seed 20260903
- GPU 3: `cv_fold_3`, fold 3, backbone seco_eco, seed 20260904
- GPU 4: `cv_fold_4`, fold 4, backbone seco_eco, seed 20260905
- GPU 5: `compare_seco_seed`, fold 0, backbone seco_eco, seed 20265901
- GPU 6: `compare_ssl4eo_moco`, fold 0, backbone ssl4eo_moco, seed 20260901
- GPU 7: `compare_imagenet`, fold 0, backbone imagenet, seed 20260901

Worker process failures: `[]`

## Final consolidated results

R² uses the mean of the evaluation split in its denominator for every model and null row. The null predictor outputs that run's training-set mean SOC.

### Headline: individual deployable control models on geographic holdout

Mean ± sample SD across the five separately trained control-fold models:

- MAE: **20.8772 ± 1.3511**
- RMSE: **41.5084 ± 0.7919**
- R2: **0.0784 ± 0.0351**
- BIAS: **-3.6727 ± 3.4774**

### Secondary: what a five-model ensemble buys

{
  "n": 325,
  "mae": 19.21418737796637,
  "rmse": 40.44855579174359,
  "r2": 0.12515948477605798,
  "bias": -3.672672367646144
}

### Pooled out-of-fold spatial CV

{
  "n": 7657,
  "mae": 48.27566143388017,
  "rmse": 91.41604942912996,
  "r2": 0.44715014366386474,
  "bias": -11.269625154553879
}

### GPU 0 versus GPU 5 seed-only gap (GPU5 - GPU0)

{
  "validation": {
    "mae": 1.2494103755247536,
    "rmse": 2.2372714082548697,
    "r2": -0.025732798982929794,
    "bias": 4.914281043309145
  },
  "geographic_holdout": {
    "mae": -0.052193925930900775,
    "rmse": 2.6355609173760257,
    "r2": -0.11883645835813428,
    "bias": -3.232308161808894
  }
}

Best run selected by spatial-validation R²: `cv_fold_0`.

Skipped jobs: `[]`
