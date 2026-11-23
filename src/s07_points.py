"""Day 2, step 1: a fixed set of vegetation pixels per climate cell.

Every satellite feature is read at these exact 30 m pixel centres for every month,
so a pixel's history is comparable with itself (needed for VCI/TCI and anomalies).

- Candidate pixels: on the Landsat-aligned 30 m UTM grid, inside Malatya, at least
  2/3 of the pixel is WorldCover tree/shrub/grass/cropland.
- 25 pixels per climate cell, stratified random, fixed seed.
- Each pixel carries its cell_id (computed from the pixel-centre lon/lat with the
  same formula as config.cell_rc), WorldCover class, and Copernicus DEM elevation.

Outputs: data/points/points.gpkg (+ Earth Engine asset projects/<project>/assets/corak_points)
"""
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (AOI_GEOJSON, CLIM_RES, CLIM_X0, CLIM_Y0, CRS_UTM, DATA, GEE_PROJECT,
                    UTM_TRANSFORM, cell_rc, ee_init)

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()
PTS_DIR = DATA / "points"
PTS_DIR.mkdir(parents=True, exist_ok=True)
N_PER_CELL = 25
SEED = 42
ASSET = f"projects/{GEE_PROJECT}/assets/corak_points"

aoi = ee.Geometry(json.loads(AOI_GEOJSON.read_text())["geometry"])
proj30 = ee.Projection(CRS_UTM, UTM_TRANSFORM)

wc = ee.ImageCollection("ESA/WorldCover/v200").first()
veg_frac = (wc.remap([10, 20, 30, 40], [1, 1, 1, 1], 0).toFloat()
            .reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(proj30))
lc_mode = wc.reduceResolution(ee.Reducer.mode(), maxPixels=64).reproject(proj30).rename("lc")
dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic()
       .setDefaultProjection(ee.Projection("EPSG:4326").atScale(30)).rename("elev"))

ll = ee.Image.pixelLonLat().reproject(proj30)
row = ll.select("latitude").multiply(-1).add(CLIM_Y0).divide(CLIM_RES).floor()
col = ll.select("longitude").subtract(CLIM_X0).divide(CLIM_RES).floor()
cell_id = row.multiply(100000).add(col).toInt().rename("cell_id")

stack = (cell_id.addBands(lc_mode).addBands(dem.reproject(proj30))
         .updateMask(veg_frac.gte(2 / 3)).clip(aoi))
pts = stack.stratifiedSample(numPoints=N_PER_CELL, classBand="cell_id", region=aoi,
                             projection=proj30, seed=SEED, geometries=True, tileScale=8)

df = ee.data.computeFeatures({"expression": pts, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
df = df.set_crs("EPSG:4326", allow_override=True)
df["lon"], df["lat"] = df.geometry.x, df.geometry.y
u = df.to_crs(CRS_UTM)
df["x"], df["y"] = u.geometry.x, u.geometry.y
df = df.sort_values(["cell_id", "x", "y"]).reset_index(drop=True)
df["pid"] = np.arange(len(df))
df["elev"] = df.elev.round(1)

# --- checks
r, c = cell_rc(df.lon.values, df.lat.values)
assert np.all(r * 100000 + c == df.cell_id.values), "cell_id mismatch with config.cell_rc"
# pixel corners sit at 15 (mod 30), so pixel centres sit at 0 (mod 30)
on_grid = (np.allclose(np.mod(df.x + 1e-6, 30), 0, atol=0.01)
           and np.allclose(np.mod(df.y + 1e-6, 30), 0, atol=0.01))
print(f"points: {len(df)} | cells: {df.cell_id.nunique()} | on 30 m pixel centres: {on_grid}")
print("points per cell:", df.groupby("cell_id").size().describe()[["min", "50%", "max"]].to_dict())
print("WorldCover class shares:", (df.lc.value_counts(normalize=True).round(3)).to_dict())

df = df[["pid", "cell_id", "lc", "elev", "lon", "lat", "x", "y", "geometry"]]
df.to_file(PTS_DIR / "points.gpkg", driver="GPKG")
df.drop(columns="geometry").to_csv(PTS_DIR / "points.csv", index=False)

# --- also store as an Earth Engine asset so later scripts sample exactly these pixels
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]), {"pid": int(p)})
                           for a, b, p in zip(df.lon, df.lat, df.pid)])
try:
    ee.data.deleteAsset(ASSET)
except Exception:
    pass
task = ee.batch.Export.table.toAsset(collection=fc, description="corak_points", assetId=ASSET)
task.start()
print("asset export started:", ASSET)
