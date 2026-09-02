# Single-parcel SOC founder demo

## What the pipeline produced

This inference-only demo selected MMEarth sample **3075** at **11.41942°N,
79.79233°E**. This **is a real India location**, verified against the available
Natural Earth country boundary. From the 14 broad bbox candidates it was the
highest-NDVI point that is genuinely inside India; cloud fraction was the
tiebreak criterion. Its measured mean NDVI is **0.541** and its flagged
cloud fraction is **6.37%**, so the tile is plainly labelled
**cloud-affected** in the figure.

The source is a 128×128 Sentinel-2 tile on a 10 m grid in `EPSG:32644`. Its footprint
is approximately **1.28 km × 1.28 km**, or
**163.84 ha**. This is the image footprint—not an ownership,
cadastral, or cultivated-field boundary.

The existing SeCo-Eco ResNet-50 checkpoint produced a whole-tile SOC estimate of
**27.2 g/kg**. A 4×4 grid of 32-pixel subtiles produced an illustrative
within-tile range of **-3.8–16.2 g/kg**. Each patch
covers roughly 320 m × 320 m and is resized to the model input size. Because the
model was trained on whole tiles rather than patch supervision, this map is a
visual demonstration of the inference mechanism, not validated within-field
resolution or an agronomic prescription.

Some raw patch outputs are negative, which is physically impossible for SOC.
They are deliberately left visible rather than clipped: this is direct evidence
of out-of-distribution/model-resolution failure and reinforces the LOW confidence
flag. The patch map must not be interpreted as a quantitative soil map.

The example rule fired was **BARE-SOIL REVISIT**, using mean NDVI 0.541:
Vegetation signal is high; obtain a cloud-free post-harvest scene before soil interpretation. This is transparent demo logic, not a recommendation validated for
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

- Report card: `<workdir>/work/soil-poc/artifacts/parcel_demo/field_report_card.png`
- Machine-readable details: `<workdir>/work/soil-poc/artifacts/parcel_demo/parcel_demo.json`
- Script: `<workdir>/work/soil-poc/parcel_demo.py`
