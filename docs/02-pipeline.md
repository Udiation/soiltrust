# 02 — The seven-layer pipeline

SoilTrust is designed as an integration pipeline, not a single neural network. Each layer has a distinct owner and evidence standard.

| Layer | Role | Current status | Production input |
|---|---|---|---|
| 1. Field boundary | Defines whose field and which pixels belong to it | Access needed | ALU authoritative parcel polygons and stable IDs |
| 2. Crop + season | Selects meaningful dates and interprets vegetation | Access needed | AMED crop identity, season, and observation date |
| 3. Soil inference | Estimates SOC from multispectral context | Prototype complete | Sentinel-2 12-band tiles; SeCo-Eco ResNet-50 |
| 4. Condition | Separates transient field condition from soil signal | Next build | Radar and/or in-field sensor observations |
| 5. Calibration | Anchors predictions to local laboratory truth | Access needed | Geolocated Soil Health Cards with depth, units, date and method |
| 6. Advisory rule | Converts evidence into transparent action | Prototype only | Versioned agronomic rules with confidence gates |
| 7. Delivery | Makes uncertainty operational | Prototype complete | Dashboard, sampling worklist and field report |

## Data flow

An ALU polygon clips pixels to a real parcel. AMED identifies crop and season so the system can select cloud-free, bare-soil or agronomically relevant Sentinel dates. The model estimates SOC, while condition layers identify moisture or vegetation effects. Soil Health Card observations calibrate and validate locally. A transparent rule fires only when input quality and confidence permit it. The result is delivered to an officer with both the estimate and a sampling priority.

## Why the confidence layer is central

Five independently trained models produce five predictions per tile. Their mean is the displayed prediction; their sample standard deviation is disagreement. Low disagreement means the learned solution is stable across seeds. High disagreement is direct evidence that the point estimate is fragile and the location should move up the ground-sampling queue.

Agreement is not a guarantee—all models can share one systematic bias. Local laboratory calibration remains essential.

## Input contract

The existing client contract requires a 12-band Sentinel-2 tile, explicit CRS/affine metadata, nodata handling, and SOC labels expressed with units and laboratory context. See [`notes/CLIENT_DATA_CONTRACT.md`](../notes/CLIENT_DATA_CONTRACT.md). The Cuddalore layer-specific join plan is in [`notes/cuddalore-base.md`](../notes/cuddalore-base.md).

