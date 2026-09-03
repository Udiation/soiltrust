# Soil Confidence Map

## What was run

This inference-only demonstration used all five already-trained SeCo-Eco
ResNet-50 checkpoints from the continental transfer experiment. Each model was
trained independently on the same North America training region with a different
seed. No model was trained or adjusted for this confidence map.

The aligned map covers **558 North America spatial-holdout tiles**
(in-distribution) and **289 Europe transfer tiles**
(out-of-distribution). Each plotted point is a tile centroid, not a continuous
wall-to-wall soil raster.

For every tile:

- **Prediction** is the arithmetic mean SOC prediction from the five models.
- **Disagreement** is the sample standard deviation of their five predictions.
- **LOW confidence** means disagreement exceeds **42.25 g/kg**, the
  75th percentile measured on the North America holdout. Europe was not used to
  choose this threshold.
- If any model predicts negative SOC, the tile receives a hard black “do not
  trust” marker. Negative values are not clipped or hidden.

## Punchline

| Quantity | North America holdout | Europe transfer |
|---|---:|---:|
| Tiles | 558 | 289 |
| Median model disagreement | 15.978 g/kg | 40.196 g/kg |
| LOW-confidence fraction | 25.09% | 45.33% |
| Any seed predicts impossible negative SOC | 25.27% | 5.54% |
| Ensemble mean itself is negative | 2.87% | 0.35% |

Disagreement is an honest empirical uncertainty signal: independently trained
models seeing the same tile should give similar answers where the learned
relationship is stable. Large spread exposes sensitivity to training randomness
and warns that the prediction is not robust. Agreement does **not** prove
accuracy—all five models can share the same bias—but disagreement is direct
evidence not to trust a point estimate.

## Product pitch

> **The only soil map that tells you where to trust it, and where to send ground samples instead.**

## Outputs

- Figure: `../figures/soil_confidence_map.png`
- Per-tile predictions: `results/tile_predictions.csv`
- Machine-readable summary: `results/confidence_summary.json`
