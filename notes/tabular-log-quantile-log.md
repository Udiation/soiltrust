# Log-SOC tabular quantile baseline

## Geography gate

South Asia has only 14/7982 samples (0.175%); India has 2; the holdout has zero. Section 2 was stopped by design.

## Geographic holdout: individual tabular models

All values below use Duan-smearing correction; no ensemble was run.

| Fold | MAE | RMSE | R² | Bias | PICP | Width mean | Width median | PICP low tercile | PICP high tercile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31.962 | 49.158 | -0.2922 | 14.817 | 0.538 | 186.653 | 164.928 | 0.118 | 0.944 |
| 1 | 28.651 | 45.542 | -0.1091 | 9.994 | 0.560 | 189.040 | 174.905 | 0.109 | 0.944 |
| 2 | 29.329 | 46.892 | -0.1758 | 11.071 | 0.572 | 182.194 | 155.724 | 0.155 | 0.954 |
| 3 | 29.815 | 46.434 | -0.1529 | 12.603 | 0.502 | 181.551 | 155.247 | 0.100 | 0.870 |
| 4 | 31.060 | 48.158 | -0.2401 | 13.179 | 0.511 | 199.235 | 180.148 | 0.100 | 0.898 |

Mean ± sample SD across individual folds:

- mae: **30.1635 ± 1.3366**
- rmse: **47.2369 ± 1.4297**
- r2: **-0.1940 ± 0.0725**
- bias: **12.3326 ± 1.8722**
- picp: **0.5366 ± 0.0305**
- interval_width_mean: **187.7346 ± 7.1425**
- interval_width_median: **166.1905 ± 11.1987**
- picp_low_tercile: **0.1164 ± 0.0226**
- picp_high_tercile: **0.9222 ± 0.0362**

## Verdict

The deep quantile model was prohibited by the South Asia representation gate, so no valid deep-versus-tabular uncertainty winner can be declared. The tabular baseline is the only newly calibrated uncertainty model in this run. Its holdout scores describe the MMEarth holdout geography, not India.
