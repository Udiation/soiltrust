#!/usr/bin/env python3
"""Build the Cuddalore public-data base kit. No model training."""

from pathlib import Path
import json
import h5py
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "sources"
REPO = SRC / "india-geodata"
H5 = ROOT.parent / "data/soil_organic_carbon/soil_organic_carbon.h5"


def main():
    districts = gpd.read_file(
        REPO / "data/administrative/districts/census-2011/2011_Dist.shp"
    ).to_crs("EPSG:4326")
    district = districts[
        districts["ST_NM"].str.casefold().eq("tamil nadu")
        & districts["DISTRICT"].str.casefold().eq("cuddalore")
    ].copy()
    if len(district) != 1:
        raise RuntimeError(f"Expected one Cuddalore district, found {len(district)}")
    district = district[["DISTRICT", "ST_NM", "ST_CEN_CD", "DT_CEN_CD", "censuscode", "geometry"]]
    # The upstream Census ring has a self-intersection. Repair it explicitly so
    # downstream spatial joins operate on a valid polygon, while retaining all
    # original administrative attributes.
    district["geometry"] = district.geometry.make_valid()
    district.to_file(ROOT / "cuddalore_district.geojson", driver="GeoJSON")

    cols = ["OBJECTID", "vil_lgd", "vilname11", "stname", "dtname", "sdtname",
            "gp_code", "gp_name", "subdt_lgd", "block_name", "block_lgd", "geometry"]
    villages = gpd.read_parquet(SRC / "LGD_Villages.parquet", columns=cols)
    villages = villages[
        villages["stname"].str.casefold().eq("tamil nadu")
        & villages["dtname"].str.casefold().eq("cuddalore")
    ].copy().to_crs("EPSG:4326")
    if villages.empty:
        raise RuntimeError("No LGD villages matched Tamil Nadu / Cuddalore")
    villages.to_file(ROOT / "cuddalore_villages.geojson", driver="GeoJSON")

    with h5py.File(H5, "r") as h5:
        lon, lat = map(float, h5["geolocation"][3075])
        affine = h5["transform"][3075].reshape(3, 3)
        crs_raw = h5["crs"][3075]
        tile_crs = crs_raw.decode() if isinstance(crs_raw, bytes) else str(crs_raw)
        _, height, width = h5["Sentinel2"][3075].shape

    point = gpd.GeoDataFrame(
        {"sample_index": [3075], "latitude": [lat], "longitude": [lon]},
        geometry=[Point(lon, lat)], crs="EPSG:4326"
    )
    inside = bool(district.geometry.iloc[0].covers(point.geometry.iloc[0]))
    point["inside_cuddalore"] = inside
    point.to_file(ROOT / "sample_3075_anchor.geojson", driver="GeoJSON")

    def xy(col, row):
        return (affine[0, 0] * col + affine[0, 1] * row + affine[0, 2],
                affine[1, 0] * col + affine[1, 1] * row + affine[1, 2])
    footprint = gpd.GeoDataFrame(
        {"sample_index": [3075], "source": ["MMEarth Sentinel-2"]},
        geometry=[Polygon([xy(0, 0), xy(width, 0), xy(width, height), xy(0, height)])],
        crs=tile_crs,
    ).to_crs("EPSG:4326")
    footprint.to_file(ROOT / "sample_3075_sentinel2_footprint.geojson", driver="GeoJSON")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=180)
    villages.boundary.plot(ax=ax, color="#a8b4ad", linewidth=0.22, alpha=.75)
    district.boundary.plot(ax=ax, color="#183a2b", linewidth=2.0)
    footprint.plot(ax=ax, facecolor="#3690c0", edgecolor="#034e7b", alpha=.65, linewidth=1.2)
    point.plot(ax=ax, color="#ffd92f", edgecolor="black", marker="*", markersize=190, zorder=5)
    ax.annotate("MMEarth sample 3075\n11.4194°N, 79.7923°E", (lon, lat),
                xytext=(-125, 25), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#333333"},
                bbox={"boxstyle": "round,pad=.3", "fc": "white", "alpha": .9})
    placeholder = [
        Line2D([], [], color="#183a2b", lw=2, label="Cuddalore district — PRESENT"),
        Line2D([], [], color="#a8b4ad", lw=1, label="LGD village boundaries — PRESENT"),
        Patch(facecolor="#3690c0", edgecolor="#034e7b", alpha=.65,
              label="Sentinel-2 footprint — PRESENT"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#ffd92f",
               markeredgecolor="black", markersize=13, label="MMEarth sample 3075 — PRESENT"),
        Line2D([], [], color="#8c510a", lw=2, ls="--", label="ALU field boundaries — EMPTY / API pending"),
        Patch(facecolor="#80cdc1", edgecolor="#01665e", hatch="///",
              label="AMED crop labels — EMPTY / API pending"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#d73027", markersize=7,
               label="Soil Health Card samples — EMPTY / department data pending"),
    ]
    ax.legend(handles=placeholder, loc="lower left", fontsize=8.5, framealpha=.96)
    ax.set(title="Cuddalore District Base Kit\nPublic layers ready; operational India layers shown as explicit placeholders",
           xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.grid(color="#dddddd", lw=.4)
    fig.tight_layout()
    fig.savefig(ROOT / "cuddalore_base_map.png", bbox_inches="tight")
    plt.close(fig)

    result = {
        "district_features": len(district), "village_features": len(villages),
        "sample_3075": {"longitude": lon, "latitude": lat,
                        "inside_cuddalore_polygon": inside},
        "sentinel2_footprint": {"source_crs": tile_crs, "pixels": [width, height],
                                "pixel_size_metres": [abs(float(affine[0, 0])), abs(float(affine[1, 1]))],
                                "area_hectares": abs(float(affine[0, 0] * affine[1, 1])) * width * height / 10000},
    }
    (ROOT / "build_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
