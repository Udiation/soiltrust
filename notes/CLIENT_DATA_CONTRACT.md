# Client data contract for this pipeline

The training sample is one field/lab observation paired to one cloud-screened
Sentinel-2 L2A image tile. A production delivery needs:

1. **Laboratory target:** soil organic carbon in g/kg (or an explicitly recorded
   convertible unit), sampling depth, analytical method, collection date, QA/QC
   flags, and a stable sample/field ID. Do not mix depths or lab methods without
   modeling them explicitly.
2. **Geometry:** point coordinates or field polygons in a declared CRS, with
   consent and provenance. Coordinates must be accurate enough to locate the
   sampled management unit; spatially overlapping observations must remain in
   the same train/validation group.
3. **Sentinel-2:** L2A surface reflectance bands B1, B2, B3, B4, B5, B6, B7, B8,
   B8A, B9, B11, B12, resampled/aligned to a common grid and stored in that exact
   order. Reflectance is expected as uint16 scaled by 10,000. B10 is deliberately
   absent because it is not an L2A surface-reflectance band.
4. **Temporal matching:** acquisition date and soil sampling date, ideally bare
   soil or controlled phenological windows. Define a maximum time offset and use
   cloud/shadow/snow masks (SCL plus cloud probability). Record compositing rules.
5. **Tile creation:** a consistent physical footprint around the sample/field,
   not merely a fixed number of pixels across mixed resolutions. Include valid
   pixel masks and field masks where polygons exist.
6. **Leakage-safe splits:** hold out entire farms/fields and geographic regions;
   repeated visits and nearby tiles cannot cross splits. Report both in-region
   and out-of-region performance.
7. **Covariates for a production model:** terrain, precipitation, temperature,
   soil texture, land management, and acquisition season can materially improve
   SOC prediction, but must be available under the same operational conditions.

The MMEarth proof predicts one SOC value per tile. Producing a dense soil map
requires dense/calibrated labels or a carefully validated weak-supervision
strategy; an image-level regressor should not be presented as pixel-level truth.
