"""Day 4, step 4: finish the map products (run after s18 for all years).

1. Vegetation mask, same rule as the sample points (s07): at least 2/3 of the 30 m pixel is
   WorldCover 2021 tree / shrub / grass / cropland. VHI is only evaluated on such pixels, so
   water, built-up land and bare rock are set to no data (255) in every VHI map.
2. Colour table embedded in every uint8 raster (VHI and the class rasters), so QGIS and ArcGIS
   both show the drought classes without a style file. The .qml files stay for QGIS.
3. Seasonal maps per year (the scale at which VHI and SPEI-3 agree: r = 0.86 between years):
   vhi_season_YYYY.tif   mean of the monthly VHI, April-October, where >= MIN_MONTHS months are
                         valid (averaging also damps the monthly scene seams of the thermal band)
   spei3_season_YYYY.tif mean monthly SPEI-3 over the same months (float, climate grid)
   spei3_season_YYYY_class.tif and spei3_YYYY_MM_class.tif: SPEI-3 classes as uint8 with colours
4. Check: province-mean seasonal VHI (vegetation pixels) against seasonal SPEI-3, all years.

Output: outputs/maps/*, outputs/tables/season_vhi_spei.csv
"""
import numpy as np
import pandas as pd
import rasterio

from config import CRS_UTM, ROOT, SEASON_MONTHS, STUDY_END, STUDY_START, UTM_TRANSFORM
from s18_maps import (NODATA, OUT, REF_DIR, SPEI_STYLE, UTM_TRANSFORM_MAP, VHI_STYLE, aoi, ee, fetch_all, qml,
                      write_cog)

MIN_MONTHS = 4
YEARS = range(STUDY_START, STUDY_END + 1)
COLORS = {0: "#7f0000", 1: "#e34a33", 2: "#fdbb84", 3: "#31a354"}


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) + (255,)


VHI_CMAP = {v: rgb(COLORS[int(np.digitize(v, [20, 30, 40]))]) if v <= 100 else (0, 0, 0, 0) for v in range(256)}
CLASS_CMAP = {v: rgb(COLORS[v]) if v < 4 else (0, 0, 0, 0) for v in range(256)}
CLASS_STYLE = [(0, COLORS[0], "severe"), (1, COLORS[1], "moderate"), (2, COLORS[2], "mild"), (3, COLORS[3], "none")]

# ---- 1. vegetation mask on the map grid
mask_path = REF_DIR / "vegmask.tif"
if not mask_path.exists():
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    proj30 = ee.Projection(CRS_UTM, UTM_TRANSFORM)
    veg = (wc.remap([10, 20, 30, 40], [1, 1, 1, 1], 0).toFloat()
           .reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(proj30).gte(2 / 3).clip(aoi))
    m = fetch_all(veg.rename("veg"), ["veg"], 0, 1024)[0].astype(np.uint8)
    write_cog(mask_path, m, UTM_TRANSFORM_MAP, CRS_UTM, None)
with rasterio.open(mask_path) as ds:
    VEG = ds.read(1).astype(bool)
print(f"vegetation mask: {VEG.mean():.0%} of the box", flush=True)


def read(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform, ds.crs


# ---- 2. mask + colours on every monthly VHI map
monthly = sorted(OUT.glob("vhi_20??_??.tif"))
for p in monthly:
    a, tr, crs = read(p)
    a = np.where(VEG, a, NODATA).astype(np.uint8)
    write_cog(p, a, tr, crs, NODATA, colormap=VHI_CMAP)
print(f"masked and coloured {len(monthly)} monthly VHI maps", flush=True)

# SPEI-3 class rasters for every monthly SPEI-3 map
for p in sorted(OUT.glob("spei3_20??_??.tif")):
    s, tr, crs = read(p)
    c = np.where(s == -9999, 255, np.digitize(s, [-1.5, -1.0, -0.5])).astype(np.uint8)
    q = p.with_name(p.stem + "_class.tif")
    write_cog(q, c, tr, crs, 255, colormap=CLASS_CMAP)
    qml(q.with_suffix(".qml"), CLASS_STYLE, 255)

# ---- 3. seasonal maps
rows = []
for y in YEARS:
    files = [OUT / f"vhi_{y}_{m:02d}.tif" for m in SEASON_MONTHS if (OUT / f"vhi_{y}_{m:02d}.tif").exists()]
    if not files:
        continue
    total = np.zeros(VEG.shape, np.float32)
    n = np.zeros(VEG.shape, np.uint8)
    for p in files:
        a, _, _ = read(p)
        ok = a != NODATA
        total[ok] += a[ok]
        n += ok
    season = np.where(n >= MIN_MONTHS, np.round(total / np.maximum(n, 1)), NODATA).astype(np.uint8)
    out = OUT / f"vhi_season_{y}.tif"
    write_cog(out, season, UTM_TRANSFORM_MAP, CRS_UTM, NODATA, colormap=VHI_CMAP)
    qml(out.with_suffix(".qml"), VHI_STYLE, NODATA)

    sp_files = [OUT / f"spei3_{y}_{m:02d}.tif" for m in SEASON_MONTHS if (OUT / f"spei3_{y}_{m:02d}.tif").exists()]
    stack = np.stack([read(p)[0] for p in sp_files]).astype(np.float32)
    stack[stack == -9999] = np.nan
    with np.errstate(all="ignore"):
        smean = np.nanmean(stack, 0)
    _, tr, crs = read(sp_files[0])
    write_cog(OUT / f"spei3_season_{y}.tif", np.nan_to_num(smean, nan=-9999).astype(np.float32), tr, crs, -9999)
    qml(OUT / f"spei3_season_{y}.qml", SPEI_STYLE, -9999)
    c = np.where(np.isnan(smean), 255, np.digitize(smean, [-1.5, -1.0, -0.5])).astype(np.uint8)
    write_cog(OUT / f"spei3_season_{y}_class.tif", c, tr, crs, 255, colormap=CLASS_CMAP)
    qml(OUT / f"spei3_season_{y}_class.qml", CLASS_STYLE, 255)

    v = season[season != NODATA]
    rows.append({"year": y, "months": len(files), "vhi_valid_share_of_vegetation": (season != NODATA)[VEG].mean(),
                 "vhi_mean": v.mean(), "vhi_share_below_40": (v < 40).mean(), "spei3_mean": np.nanmean(smean)})
    print(f"{y}: {len(files)} months, seasonal VHI valid on {rows[-1]['vhi_valid_share_of_vegetation']:.0%} of "
          f"vegetation, mean {v.mean():.1f}, SPEI-3 mean {np.nanmean(smean):+.2f}", flush=True)

# ---- 4. check
t = pd.DataFrame(rows)
t.to_csv(ROOT / "outputs" / "tables" / "season_vhi_spei.csv", index=False)
tr_ = t[~t.year.isin([2021, 2022])]
print(f"\nprovince seasonal VHI vs SPEI-3: r = {t.vhi_mean.corr(t.spei3_mean):.2f} all {len(t)} years, "
      f"{tr_.vhi_mean.corr(tr_.spei3_mean):.2f} training years")
print(t.round(2).to_string(index=False))
