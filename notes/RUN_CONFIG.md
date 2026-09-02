# Reproducibility configuration: log-SOC tabular quantile run

## Scope and geography gate

- Dataset: MMEarth-Bench soil organic carbon, all 7,982 samples used by the saved CV/holdout design.
- South Asia bbox: 6–37°N, 68–98°E; 14 samples (0.175%).
- India: 2 samples by Natural Earth 1:10m country polygon; geographic holdout contains zero South Asia samples.
- India boundary source: `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson`, feature `ADMIN == India`.
- South Asia SOC: n=14, mean=36.879, median=10.200, p5=0.855, p95=139.720, max=179.500 g/kg.
- Outside South Asia SOC: n=7,968, mean=78.142, median=25.500, p5=3.600, p95=383.895, max=779.000 g/kg.
- India SOC: n=2, mean/median=9.450, p5=1.935, p95=16.965, max=17.800 g/kg.
- South Asia occupies 11 distinct 50 km blocks. Validation-fold allocation (samples/blocks): fold 0 1/1; fold 1 2/2; fold 2 2/2; fold 3 2/2; fold 4 7/4. The holdout has 0/0.
- Because South Asia representation is below 2%, the requested deep uncertainty model was not run. These results are not Indian calibration results.

## Target and correction

- Training target: `log1p(SOC_g_per_kg)`.
- Quantiles: 0.05, 0.50, 0.95; point prediction is q0.50.
- Naive back-transform: `max(expm1(q), 0)`.
- Duan correction per fold: `s = mean(exp(y_log_train - q50_log_train))`; corrected quantiles `max(s * exp(q) - 1, 0)`.
- R² in every space: `1 - SSE / sum((y_space - mean(y_space_evaluation_split))^2)`.
- Null: training-split arithmetic mean in the same evaluation space, predicted everywhere.
- Bias: mean(prediction - truth); negative means underprediction.

## Features (19)

- `B1_mean`
- `B2_mean`
- `B3_mean`
- `B4_mean`
- `B5_mean`
- `B6_mean`
- `B7_mean`
- `B8_mean`
- `B8A_mean`
- `B9_mean`
- `B11_mean`
- `B12_mean`
- `NDVI_mean=(B8-B4)/(B8+B4)`
- `NBR2_mean=(B11-B12)/(B11+B12)`
- `BSI_mean=((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))`
- `clay_ratio_mean=B11/B12`
- `carbonate_ratio_mean=B11/B8A`
- `ASTER_GDEM_elevation_mean_m`
- `ASTER_GDEM_slope_mean_degrees`

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
| 0 | 6125 | 1532 | 2287 | 571 | {'lon_min': -157.05027574114948, 'lat_min': -42.79849710919871, 'lon_max': 153.37605642110282, 'lat_max': 72.9499759282339} |
| 1 | 6125 | 1532 | 2287 | 571 | {'lon_min': -157.73099321641823, 'lat_min': -63.87085286909405, 'lon_max': 172.73092280999853, 'lat_max': 69.48690904077844} |
| 2 | 6126 | 1531 | 2286 | 572 | {'lon_min': -156.73101641100934, 'lat_min': -46.59018666275372, 'lon_max': 169.25127260804206, 'lat_max': 73.00001934845483} |
| 3 | 6126 | 1531 | 2286 | 572 | {'lon_min': -156.4629553062303, 'lat_min': -42.76794197788533, 'lon_max': 175.44922925822803, 'lat_max': 72.98330989448999} |
| 4 | 6126 | 1531 | 2286 | 572 | {'lon_min': -161.95795410521438, 'lat_min': -42.791588304999664, 'lon_max': 153.219314854277, 'lat_max': 72.90002575351475} |

Holdout bbox: `{'lon_min': -16.860536147286354, 'lat_min': -34.235550488899975, 'lon_max': 45.370277540815906, 'lat_max': 36.92332336026316}`.

## Package versions

- python: `3.12.3`
- numpy: `2.5.2`
- pandas: `3.0.5`
- scikit_learn: `1.9.0`
- h5py: `3.16.0`
- torch: `2.13.0+cu130`
- torchgeo: `0.10.0`
- geopandas: `1.1.4`
- pyproj: `3.7.2`
