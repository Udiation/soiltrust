# 01 — The India data gap

## The client question

Can a globally trained satellite model be used as an India soil map? The measured answer is **not yet**. The training geography is too uneven, and a rectangular India query finds almost no calibration data.

## What we measured

MMEarth's SOC task contains 7,982 geolocated samples. Applying the same inclusive bounding boxes used by the diagnostic produced:

| Region | Count | Dataset share |
|---|---:|---:|
| Broad India box: 6–37°N, 68–98°E | 14 | 0.1754% |
| South Asia: 5–40°N, 60–100°E | 21 | 0.2631% |
| Europe: 35–70°N, 10°W–40°E | 289 | 3.6206% |
| North America: 25–70°N, 170°W–50°W | 3,719 | 46.5923% |

The India rectangle is deliberately described as a **broad-box upper bound**, not as 14 confirmed Indian samples. It includes neighbouring countries. A separate low-resolution country-polygon check found only two points inside India.

![Coverage audit](../figures/india_coverage_audit.png)

## Why this matters

Satellite reflectance does not map to SOC in isolation. Soil mineralogy, moisture, crop cover, management, season, atmosphere, sampling depth, and laboratory methods can all change the relationship. Spatially nearby samples also resemble each other, so a random train/validation split can leak local geographic structure and make performance look more transferable than it is.

This dataset imbalance is why a North-America-trained model was tested on Europe and why India outputs are marked LOW confidence. The right next step is not stronger marketing; it is local, geolocated laboratory calibration with spatially honest validation.

## Traceability

- Raw diagnostic: [`notes/india-coverage.md`](../notes/india-coverage.md)
- Geographic transfer run: [`notes/continental-log.md`](../notes/continental-log.md)
- Coverage numbers were computed directly from the HDF5 `geolocation` array; no training was used.

