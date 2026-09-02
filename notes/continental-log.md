# Continental transfer test

## Preflight

- North America bbox count before splitting: **3,719**.
- Europe transfer bbox count: **289**.
- North America training: **3,161 samples**, **1,297 blocks**.
- North America spatial holdout: **558 samples** (15.004%), **223 blocks**.
- Approximate block size: **50 km × 50 km**; 50/111.32 degree latitude bands; longitude width cosine-adjusted at band midpoint.
- R² convention: 1 - SSE/sum((y - mean(y_evaluation_split))^2); never clipped.
- Mean baseline: North America training-target arithmetic mean predicted on both evaluation sets.
- Loader workers: **2 per deep job** (10 total); baseline jobs do not create loaders.

| GPU | Run | Seed | Kind |
|---:|---|---:|---|
| 0 | deep_seed_0 | 20260901 | deep |
| 1 | deep_seed_1 | 20260902 | deep |
| 2 | deep_seed_2 | 20260903 | deep |
| 3 | deep_seed_3 | 20260904 | deep |
| 4 | deep_seed_4 | 20260905 | deep |
| 5 | mean_seed_0 | 20260901 | mean_baseline |
| 6 | mean_seed_1 | 20260902 | mean_baseline |
| 7 | mean_seed_2 | 20260903 | mean_baseline |

Deep hyperparameters match `train_soc.py`: SeCo-Eco ResNet-50, end-to-end Huber loss on training z-scores, AdamW lr 2e-4 and weight decay 1e-4, cosine schedule, 30 epochs, batch 32, gradient clipping 5, BF16 autocast, patience 8.

## Final results

All R² values use the corresponding evaluation set's own mean denominator and are reported without clipping.

| Model | Split | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD | Bias mean ± SD |
|---|---|---:|---:|---:|---:|
| deep | north_america_holdout | 66.9254 ± 1.2085 | 117.9920 ± 1.8772 | 0.3342 ± 0.0211 | -15.2807 ± 6.0873 |
| deep | europe_transfer | 106.9072 ± 18.4652 | 139.1520 ± 15.7792 | -0.2216 ± 0.2759 | 20.6381 ± 22.0480 |
| mean_baseline | north_america_holdout | 121.7432 ± 0.0000 | 146.4347 ± 0.0000 | -0.0253 ± 0.0000 | 22.9885 ± 0.0000 |
| mean_baseline | europe_transfer | 109.0526 ± 0.0000 | 128.8704 ± 0.0000 | -0.0370 ± 0.0000 | 24.3541 ± 0.0000 |

## Headline Europe transfer comparison

- Deep Europe RMSE: **139.1520 ± 15.7792 g/kg**.
- North-America-training-mean baseline Europe RMSE: **128.8704 ± 0.0000 g/kg**.
- RMSE ratio (deep / baseline): **1.0798**.
- Relative RMSE change: **7.98% increase** versus the baseline.
- The mean ratio is above 1.0, so the deep model does not beat the transfer baseline by the requested criterion.

## Interpretation

- Within North America, the deep model clearly beats the training-mean floor:
  RMSE 117.9920 versus 146.4347 g/kg, with mean R² 0.3342.
- Across the continental shift, results are highly seed-sensitive. Two of five
  seeds beat the baseline Europe RMSE (seeds 20260903 and 20260904), while three
  did not. Individual Europe R² ranges from -0.5750 to 0.0861.
- Averaged across the five independently trained models, Europe RMSE is 7.98%
  higher than the baseline. This experiment therefore does not demonstrate
  reliable North-America-to-Europe transfer, despite isolated seeds showing
  some skill.
- Negative baseline R² (-0.0370) is expected and retained: the North America
  training mean differs from Europe's evaluation mean.
