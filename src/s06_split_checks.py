"""Step 11b: is the frozen split usable?

  1. Class counts in each set that will actually be scored (not just province-wide).
  2. Land cover of test vs training cells (WorldCover vegetated / cropland share per cell).
     Land cover is not outcome data, so looking at it leaks nothing.
  3. How SPEI-3 variance splits between months and between cells within a month,
     which tells us how much the spatial test can show.

Output: data/climate/cells_split.csv gains veg_frac and crop_frac columns.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from config import CLASS_NAMES, CLIMATE_DIR, SEASON_MONTHS, SPEI_CUTS, ee_init

warnings.filterwarnings("ignore", category=DeprecationWarning)
cells = pd.read_csv(CLIMATE_DIR / "cells_split.csv")
sp = xr.open_dataset(CLIMATE_DIR / "spei3.nc")
t = sp.time.to_index()
TEST_YEARS = [2021, 2022]

# ---- 1. class counts per scored set (growing season, 2014-2025)
season = t.month.isin(SEASON_MONTHS) & (t.year >= 2014)
S = sp.spei3.values[:, cells.iy, cells.ix][season]
yrs = t.year[season].values
cls = np.digitize(S, SPEI_CUTS)                       # 0 severe .. 3 none
ty = np.isin(yrs, TEST_YEARS)[:, None]
blk = (cells.split.values == "test")[None, :]
trn = (cells.split.values == "train")[None, :]
sets = {"train (train cells x train years)": trn & ~ty,
        "temporal test (train cells x test years)": trn & ty,
        "spatial test (test cells x train years)": blk & ~ty,
        "strict test (test cells x test years)": blk & ty}
rows = []
for name, m in sets.items():
    m = np.broadcast_to(m, cls.shape)
    rows.append({"set": name, **{CLASS_NAMES[3 - k]: int((cls[m] == k).sum()) for k in range(4)}})
print("1. cell-months per class in each scored set")
print(pd.DataFrame(rows).to_string(index=False))

# ---- 2. land cover per cell
ee = ee_init()
wc = ee.ImageCollection("ESA/WorldCover/v200").first()
lc = ee.Image.cat(wc.remap([10, 20, 30, 40], [1, 1, 1, 1], 0).rename("veg"),
                  wc.eq(40).rename("crop")).toFloat()
import geopandas as gpd
g = gpd.read_file(CLIMATE_DIR / "cells.gpkg")[["cell_id", "geometry"]]
g = g[g.cell_id.isin(cells.cell_id)]
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Polygon(list(geom.exterior.coords)), {"cell_id": int(cid)})
                           for cid, geom in zip(g.cell_id, g.geometry)])
res = lc.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=100, crs="EPSG:32637").getInfo()
lcdf = pd.DataFrame([f["properties"] for f in res["features"]])
cells = cells.drop(columns=["veg_frac", "crop_frac"], errors="ignore").merge(
    lcdf.rename(columns={"veg": "veg_frac", "crop": "crop_frac"}), on="cell_id", how="left")
cells.to_csv(CLIMATE_DIR / "cells_split.csv", index=False)
print("\n2. land cover by split (mean share of each cell)")
print(cells.groupby("split")[["veg_frac", "crop_frac", "elev"]].agg(["mean", "min", "max"]).round(2).to_string())

# ---- 3. variance decomposition (growing season, 2014-2025, cells inside the province)
inside = cells.frac_in.values >= 0.5
Si = S[:, inside]
between = np.nanvar(np.nanmean(Si, axis=1))
within = np.nanmean(np.nanvar(Si, axis=1))
print(f"\n3. SPEI-3 variance between months: {between:.3f} | within month (between cells): {within:.3f} "
      f"| share between months: {100 * between / (between + within):.0f}%")
