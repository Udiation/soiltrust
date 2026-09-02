# Cuddalore district base kit

This directory is a public-data integration shell for Cuddalore, Tamil Nadu. It contains real administrative boundaries and the footprint of the existing MMEarth/Sentinel-2 anchor tile. ALU, AMED, and Soil Health Card layers are visibly empty placeholders—not simulated data.

## Present now

| Layer | Local file | Source / status |
|---|---|---|
| Cuddalore district | `cuddalore_district.geojson` | Census 2011 district boundary from `yashveeeeeeer/india-geodata`; one feature, EPSG:4326 |
| Cuddalore villages | `cuddalore_villages.geojson` | LGD village release from the same repository; 870 named features, EPSG:4326 |
| Anchor location | `sample_3075_anchor.geojson` | MMEarth HDF5 geolocation; sample 3075 at 11.419416°N, 79.792326°E |
| Sentinel-2 footprint | `sample_3075_sentinel2_footprint.geojson` | Derived from MMEarth's stored affine transform and CRS; 128×128 pixels at 10 m, 163.84 ha |
| Integration map | `cuddalore_base_map.png` | District/village context, anchor and image footprint, plus explicit empty-layer legend entries |

The point-in-polygon check confirms sample 3075 is inside the downloaded Cuddalore district polygon. `build_summary.json` records this machine-readable result. `connectivity_and_download_log.md` records the network and government-data attempts. Boundary provenance is pinned to upstream commit `6e5a00c781a6d50e580bd8752526293ef18a54da`.

## Pending access

- **ALU field boundaries:** pending API/data-sharing access. No parcel boundaries are inferred from village polygons.
- **AMED crop labels:** pending API/data-sharing access. No crop identity is guessed from imagery.
- **Soil Health Card observations:** pending department/API export. Portal reachability did not yield an anonymous Cuddalore data export.
- **Agriculture Census land holdings:** portal reachable, but no machine-readable Cuddalore extract was acquired in this run.

## Plug-in plan

1. **ALU:** place the supplied parcel polygons in `incoming/alu_fields.*`; retain its stable field/parcel ID, reproject to EPSG:4326, validate/fix geometries, and spatially intersect with `cuddalore_district.geojson`. Join imagery and later predictions by that stable parcel ID. Preserve the raw source separately and record its vintage.
2. **AMED:** place crop observations in `incoming/amed_crop_labels.*`. Require crop label, observation/season date, and either parcel ID or geometry. Prefer a direct ALU parcel-ID join; otherwise spatially join points/polygons to ALU fields and retain match distance/overlap as QA fields.
3. **Soil Health Cards:** place a department export in `incoming/soil_health_cards.*`. Require sample ID, coordinates, sample date, depth, units, and laboratory method plus nutrient/SOC columns. Normalize units, reject invalid coordinates, then spatially join samples to ALU parcels. Use these only for India calibration/validation, with spatially separated evaluation.
4. **Agriculture Census:** place the official Cuddalore table in `incoming/ag_census.*`; retain year, administrative codes, holding-size class, count and area units. Aggregate/join through LGD village/subdistrict codes rather than fuzzy names wherever codes overlap.
5. Rebuild the map by running `source ~/work/.venv/bin/activate && python build_base.py`, then add each validated layer as a real plot layer and remove only its corresponding EMPTY legend entry.

## Important interpretation notes

Village polygons are administrative units, not farm parcels. The Sentinel-2 footprint is approximately 1.28 km × 1.28 km and does not establish a cadastral field boundary. Dataset vintages differ (2011 Census district versus the later LGD village compilation), so boundary discrepancies at edges must be treated as source-version differences, not silently edited.
