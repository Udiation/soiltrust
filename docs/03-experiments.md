# 03 — Experiments

## 1. Initial feasibility run

A Sentinel-2-pretrained SeCo-Eco ResNet-50 was fine-tuned end to end to predict one SOC value per 12-band, 128×128 MMEarth tile. The network used a scalar regression head, standardized SOC targets, Huber loss, AdamW, cosine scheduling, flips/rotations, BF16 autocast, and eager PyTorch.

The ordinary validation split reached R² 0.5107, but a separate geographic holdout reached only 0.0212. That contrast motivated spatially stricter testing; the random-style number is not used as a deployment claim.

## 2. Spatial block cross-validation

Samples were grouped into approximate 50 km cells and whole cells assigned to five folds, preventing local neighbours from being split individually across train and validation. Five independently deployable control models produced geographic-holdout R² **0.0784 ± 0.0351**. Their ensemble reached 0.1252, but the individual-model mean remains the honest deployment headline.

## 3. Continental transfer test

Five single-GPU seeds used the same North America training set: 3,161 training samples, a spatially blocked 558-sample North America holdout, and an untouched 289-sample Europe transfer set. The baseline predicted the North America training mean everywhere. R² was computed against each evaluation split's own mean and was never clipped.

This design tests the actual risk: a model that learns within one data-rich continent may not survive a geographic shift.

## 4. Model-disagreement confidence

No new training was performed. The five continental checkpoints ran on every North America holdout and Europe tile. The confidence threshold—42.25 g/kg disagreement—was fixed as the North America holdout's 75th percentile; Europe was not used to choose it. Negative outputs received a separate hard warning.

## 5. India parcel-style demonstration

Sample 3075 is at 11.41942°N, 79.79233°E and inside the available India country polygon. Its measured NDVI was 0.541 and cloud fraction 6.37%. The report displays RGB, SWIR false colour, native versus sharpened imagery, field-level inference and patch inference. It explicitly states that the 163.84 ha image footprint is not an ownership parcel and that negative patch values are invalid.

## 6. Cuddalore integration base

Public Census/LGD boundaries produced one valid Cuddalore district polygon and 870 valid village polygons. Sample 3075 falls inside the district polygon. ALU, AMED, and Soil Health Card data were not anonymously acquired, so the map carries empty, labelled placeholders for them.

## Reproducibility trail

- Exact run configuration: [`notes/RUN_CONFIG.md`](../notes/RUN_CONFIG.md)
- Spatial CV narrative and metrics: [`notes/spatial-cv-log.md`](../notes/spatial-cv-log.md)
- Continental run narrative: [`notes/continental-log.md`](../notes/continental-log.md)
- Machine-readable tables: [`notes/results/`](../notes/results/)
- Executable implementations: [`code/`](../code/)

