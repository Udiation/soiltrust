# MMEarth SOC geographic coverage diagnostic

Date: 2026-09-02

## Result

The MMEarth soil-organic-carbon HDF5 file contains **7,982 samples**.

| Region definition (inclusive bounding box) | Samples | Fraction of dataset |
|---|---:|---:|
| India approximation: 6–37°N, 68–98°E | **14** | **0.1754%** |
| South Asia: 5–40°N, 60–100°E | **21** | **0.2631%** |
| Europe: 35–70°N, 10°W–40°E | **289** | **3.6206%** |
| North America: 25–70°N, 170°W–50°W | **3,719** | **46.5923%** |

The requested India box therefore contains only about **0.18%** of the SOC
dataset. This is a broad rectangular approximation, not an India-border test:
some of its 14 samples may lie in neighbouring countries. It is an upper-bound
style coverage diagnostic and should not be presented as 14 confirmed Indian
samples. The previous polygon audit found only two points inside India's actual
Natural Earth boundary.

## Samples by continent

Continents below were assigned with a point-in-polygon join against the locally
available Natural Earth low-resolution country polygons. Bars are scaled to the
largest category (North America = 40 characters).

```text
North America       3,809  ########################################
Oceania             2,951  ###############################
South America         355  ####
Europe                 348  ####
Africa                 324  ###
Asia                    80  #
Unassigned / ocean     115  #
                     -----
Total                7,982
```

The continent counts use country polygons, whereas the Europe and North America
figures in the table use the exact rectangular boxes requested. They therefore
should not be expected to match. The 115 unassigned points do not fall within a
country polygon in the low-resolution layer; this can include islands/coastline
edge cases and is not silently assigned to a continent.

## Coordinate availability and loading

Coordinates are stored directly in
`data/soil_organic_carbon/soil_organic_carbon.h5` under the dataset
`geolocation`, with shape `(7982, 2)` and `float64` values. The existing loaders
in `train_soc_spatial.py` and `train_soc_tabular_quantile.py` read this dataset
and consistently interpret it as:

```python
geolocation = h5["geolocation"][:]
longitude = geolocation[:, 0]
latitude = geolocation[:, 1]
```

Observed ranges are longitude **-161.958 to 175.449°** and latitude **-63.871
to 73.000°**. The HDF5 dataset itself has no descriptive attributes, so the
column-order interpretation comes from the project's existing loader and split
logic rather than an HDF5 attribute.

All bounding-box comparisons used inclusive lower and upper bounds. No model was
loaded and no training was run for this diagnostic.
