"""Deck figures from the finished map products (read-only use of outputs/maps).

Seasonal VHI tiles, one per year 2014-2026: the April-October mean VHI (30 m, vegetation only),
classified with Kogan's thresholds into the deck's drought colours, over a light province silhouette.

Usage:  .venv/bin/python docs/sunum/make_figures.py
Output: docs/sunum/figures/vhi_season_<year>.png
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parents[2]
MAPS = ROOT / "outputs" / "maps"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
WIDTH = 440

# deck colours: severe, moderate, mild, none, province background
COLORS = {0: (122, 50, 22), 1: (200, 100, 59), 2: (233, 185, 143), 3: (47, 111, 107)}
BG = (227, 220, 203)

aoi = json.loads((ROOT / "data" / "aoi" / "malatya.geojson").read_text())["geometry"]

for year in range(2014, 2027):
    with rasterio.open(MAPS / f"vhi_season_{year}.tif") as src:
        h = round(src.height * WIDTH / src.width)
        vhi = src.read(1, out_shape=(h, WIDTH), resampling=rasterio.enums.Resampling.nearest)
        transform = src.transform * src.transform.scale(src.width / WIDTH, src.height / h)
        geom = transform_geom("EPSG:4326", src.crs, aoi)
    inside = rasterize([(geom, 1)], out_shape=(h, WIDTH), transform=transform, fill=0).astype(bool)
    rgba = np.zeros((h, WIDTH, 4), np.uint8)
    rgba[inside] = (*BG, 255)
    valid = (vhi != 255) & inside
    cls = np.digitize(vhi, [20, 30, 40])          # 0 severe ... 3 none
    for k, c in COLORS.items():
        m = valid & (cls == k)
        rgba[m] = (*c, 255)
    Image.fromarray(rgba, "RGBA").save(OUT / f"vhi_season_{year}.png", optimize=True)
    print(year, f"{h}x{WIDTH}", f"valid {valid.sum() / inside.sum():.0%}")
