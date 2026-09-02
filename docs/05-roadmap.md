# 05 — Roadmap and access requirements

## Immediate access needed

1. **ALU parcel boundaries:** authoritative geometry and stable parcel IDs for Cuddalore. Village polygons cannot substitute for fields.
2. **AMED crop and season labels:** crop identity, observation date/season, and either parcel ID or geometry. These determine which satellite dates are meaningful.
3. **Soil Health Card laboratory records:** start with one district. Required fields are sample ID, coordinates, sample date, depth, units, laboratory method, and SOC/nutrient values.
4. **Multi-date Sentinel archive:** cloud-screened surface reflectance covering pre-sowing/post-harvest bare-soil windows, including SWIR bands B11 and B12.

## Fast integration path

- Reproject authoritative inputs to a documented common CRS while preserving their original copies and IDs.
- Validate geometry and intersect ALU fields with the existing Cuddalore district base.
- Join AMED by parcel ID where possible; otherwise record spatial overlap or match distance for audit.
- Normalize Soil Health Card units and depths, reject invalid coordinates, then link lab points to parcels.
- Define spatial blocks before model selection. Keep a final geographic holdout untouched until the deployment candidate is fixed.
- Calibrate both point estimates and uncertainty against India data. Measure interval coverage and error by district, soil range, season, crop and imagery quality.
- Publish confidence gates that convert weak coverage, model disagreement, clouds, vegetation and impossible predictions into sampling priorities or “do not advise” states.

## Product milestones

| Milestone | Evidence required |
|---|---|
| Cuddalore ingestion complete | ALU + AMED + laboratory schemas validated and joins audited |
| India calibration baseline | Simple tabular and vision models compared on identical spatial folds |
| Trustworthy uncertainty | Nominal intervals assessed for coverage, width and low/high-SOC terciles |
| Officer pilot | Sampling worklist tested against field-team constraints and new lab returns |
| Deployment claim | Locked, untouched district/geographic evaluation with documented provenance |

## Non-negotiable safeguards

Do not clip away impossible predictions in evaluation, call a benchmark tile a parcel, infer crop labels without AMED evidence, or market broad-box India coverage as local calibration. The confidence layer should route scarce sampling—not excuse missing validation.

