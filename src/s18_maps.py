"""Day 4, step 2: province-wide maps as Cloud-Optimized GeoTIFFs (open in QGIS or ArcGIS).

VHI (30 m, UTM 37N, Landsat grid), for each month April-October of the map years:
  monthly median NDVI / LST exactly as at the sample points (s08 / s08b processing)
  VCI = 100 (NDVI - NDVImin) / (NDVImax - NDVImin), TCI = 100 (LSTmax - LST) / (LSTmax - LSTmin),
  with min/max from the training-year reference (s17: the EE asset when it exists, otherwise the
  same image computed block by block and cached in data/ref/); VHI = mean of VCI and TCI, each
  clipped to 0-100. Stored as uint8 (0-100, 255 = no data: cloud, snow, too few reference
  years, outside the province).
SPEI-3 and SPEI-6 (0.05 deg CHIRPS grid, EPSG:4326), same months, float32, cells touching the province.
QGIS styles (.qml, same name as each raster) colour both by the drought classes
(VHI <20 / 20-30 / 30-40 / >=40; SPEI < -1.5 / -1.0 / -0.5 / above).

Usage: python s18_maps.py [YEAR ...]     default: 2021 2022 2025
Output: outputs/maps/vhi_YYYY_MM.tif, spei3_YYYY_MM.tif, spei6_YYYY_MM.tif (+ .qml)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import rasterio
import rasterio.shutil
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from config import AOI_GEOJSON, CLIMATE_DIR, CLIM_RES, CRS_UTM, DATA, ROOT, SEASON_MONTHS, ee_init
from s03_spei import glo_spei
from s17_vhi_climatology import aoi, asset_id, monthly, reference

ee = ee_init()
ee.data.setDeadline(300_000)       # 5 min per request: a hung request raises and is retried instead of blocking forever
OUT = ROOT / "outputs" / "maps"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = [int(a) for a in sys.argv[1:]] or [2021, 2022, 2025]
NODATA = 255

# ---- 30 m grid covering the province, snapped to the Landsat pixel corners (15 m mod 30)
geo = json.loads(AOI_GEOJSON.read_text())["geometry"]
lon = [c[0] for ring in (geo["coordinates"] if geo["type"] == "Polygon" else [r for p in geo["coordinates"] for r in p])
       for c in ring]
lat = [c[1] for ring in (geo["coordinates"] if geo["type"] == "Polygon" else [r for p in geo["coordinates"] for r in p])
       for c in ring]
xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", CRS_UTM, min(lon), min(lat), max(lon), max(lat))
X0 = np.floor((xmin - 15) / 30) * 30 + 15
Y0 = np.ceil((ymax - 15) / 30) * 30 + 15
W = int(np.ceil((xmax - X0) / 30))
H = int(np.ceil((Y0 - ymin) / 30))


def fetch(img, bands, r0, c0, h, w, fill):
    """One block of `img` on the 30 m grid; splits itself in four on 'memory limit' errors."""
    grid = {"dimensions": {"width": w, "height": h}, "crsCode": CRS_UTM,
            "affineTransform": {"scaleX": 30, "shearX": 0, "translateX": X0 + 30 * c0,
                                "shearY": 0, "scaleY": -30, "translateY": Y0 - 30 * r0}}
    for attempt in range(5):
        try:
            a = ee.data.computePixels({"expression": img.unmask(fill), "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            return np.stack([a[b] for b in bands])
        except Exception as e:
            msg = str(e)
            if "memory" in msg.lower() and min(h, w) > 128:
                out = np.empty((len(bands), h, w), np.float32)
                h2, w2 = h // 2, w // 2
                for dr, dc, hh, ww in [(0, 0, h2, w2), (0, w2, h2, w - w2), (h2, 0, h - h2, w2), (h2, w2, h - h2, w - w2)]:
                    out[:, dr:dr + hh, dc:dc + ww] = fetch(img, bands, r0 + dr, c0 + dc, hh, ww, fill)
                return out
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
            print(f"    retry block {r0},{c0}: {msg[:80]}", flush=True)


def fetch_all(img, bands, fill, block):
    blocks = [(r, c, min(block, H - r), min(block, W - c)) for r in range(0, H, block) for c in range(0, W, block)]
    out = np.full((len(bands), H, W), fill, np.float32)
    with ThreadPoolExecutor(6) as ex:
        for (r0, c0, h, w), a in zip(blocks, ex.map(lambda b: fetch(img, bands, *b, fill), blocks)):
            out[:, r0:r0 + h, c0:c0 + w] = a
    return out


REF_DIR = DATA / "ref"
REF_DIR.mkdir(exist_ok=True)
REF_BANDS = ["NDVI_min", "NDVI_max", "LST_min", "LST_max"]


def reference_local(month):
    """Training-year min/max (s17 definition), from the EE asset if it exists, else computed
    block by block; cached as data/ref/vhi_ref_MM.tif (int16, -32768 = no data)."""
    path = REF_DIR / f"vhi_ref_{month:02d}.tif"
    if not path.exists():
        try:
            ee.data.getAsset(asset_id(month))
            img = ee.Image(asset_id(month))
        except Exception:
            img = reference(month)
        a = fetch_all(img, REF_BANDS, -32768, 512).astype(np.int16)
        with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=4, dtype="int16", crs=CRS_UTM,
                           transform=Affine(30, 0, X0, 0, -30, Y0), nodata=-32768, compress="deflate",
                           tiled=True) as ds:
            ds.write(a)
    with rasterio.open(path) as ds:
        a = ds.read().astype(np.float32)
    a[a == -32768] = np.nan
    return a


def vhi_array(year, month):
    """VHI exactly as s10 (VCI, TCI clipped to 0-100, averaged), computed locally."""
    nmin, nmax, lmin, lmax = reference_local(month)
    m = fetch_all(monthly(year, month).clip(aoi), ["NDVI", "LST"], -9999, 1024)
    ndvi, lst = np.where(m[0] == -9999, np.nan, m[0] * 10000), np.where(m[1] == -9999, np.nan, m[1] * 100)
    with np.errstate(all="ignore"):
        vci = np.clip(100 * (ndvi - nmin) / (nmax - nmin), 0, 100)
        tci = np.clip(100 * (lmax - lst) / (lmax - lmin), 0, 100)
        vhi = 0.5 * vci + 0.5 * tci
    return np.where(np.isfinite(vhi), np.round(vhi), NODATA).astype(np.uint8)


def write_cog(path, arr, transform, crs, nodata, colormap=None):
    """Cloud-Optimized GeoTIFF. `colormap` (uint8 rasters only) is embedded as a colour table,
    which QGIS and ArcGIS both display without any style file."""
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype.name,
                crs=crs, transform=transform, nodata=nodata)
    with MemoryFile() as mem:
        with mem.open(**prof) as ds:
            ds.write(arr, 1)
            if colormap is not None:
                ds.write_colormap(1, colormap)
        with mem.open() as src:
            rasterio.shutil.copy(src, path, driver="COG", compress="DEFLATE", overview_resampling="nearest")


def qml(path, items, nodata):
    """QGIS discrete style: each item (upper value, colour, label) colours values <= upper value."""
    rows = "\n".join(f'          <item alpha="255" value="{v}" color="{c}" label="{escape(l, {chr(34): "&quot;"})}"/>'
                     for v, c, l in items)
    path.write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" nodataColor="">
      <rastershader>
        <colorrampshader colorRampType="DISCRETE" classificationMode="1" clip="0">
{rows}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
""")


VHI_STYLE = [(19, "#7f0000", "severe (VHI < 20)"), (29, "#e34a33", "moderate (20-30)"),
             (39, "#fdbb84", "mild (30-40)"), (100, "#31a354", "none (>= 40)")]
SPEI_STYLE = [(-1.5, "#7f0000", "severe (< -1.5)"), (-1.0, "#e34a33", "moderate (-1.5 to -1.0)"),
              (-0.5, "#fdbb84", "mild (-1.0 to -0.5)"), (9, "#31a354", "none (>= -0.5)")]

UTM_TRANSFORM_MAP = Affine(30, 0, X0, 0, -30, Y0)


if __name__ == "__main__":
    # ---- VHI maps
    print(f"VHI grid {W} x {H} px at 30 m", flush=True)
    for year in YEARS:
        for month in SEASON_MONTHS:
            path = OUT / f"vhi_{year}_{month:02d}.tif"
            if path.exists():
                continue
            t0 = time.time()
            arr = vhi_array(year, month)
            write_cog(path, arr, UTM_TRANSFORM_MAP, CRS_UTM, NODATA)
            qml(path.with_suffix(".qml"), VHI_STYLE, NODATA)
            valid = arr != NODATA
            print(f"{path.name}: valid {valid.mean():.0%} of the box, median VHI {np.median(arr[valid]):.0f}, "
                  f"{time.time() - t0:.0f} s", flush=True)

    # ---- SPEI-3 and SPEI-6 maps on the climate grid
    sp = xr.open_dataset(CLIMATE_DIR / "spei3.nc")
    t = sp.time.to_index()
    cells = pd.read_csv(CLIMATE_DIR / "cells.csv")
    cells = cells[cells.frac_in > 0]                      # cells.csv is the whole box; keep the 597 touching cells
    inside = np.zeros(sp.spei3.shape[1:], bool)
    inside[cells.iy.values, cells.ix.values] = True
    grids = {"spei3": sp.spei3.values, "spei6": glo_spei(sp.D.rolling(time=6).sum().values, t.month.values, t.year.values)}
    lat0 = float(sp.lat.values[0]) + CLIM_RES / 2         # lat descending, cell centres
    lon0 = float(sp.lon.values[0]) - CLIM_RES / 2
    assert sp.lat.values[0] > sp.lat.values[-1]
    clim_transform = Affine(CLIM_RES, 0, lon0, 0, -CLIM_RES, lat0)
    for year in YEARS:
        for month in SEASON_MONTHS:
            i = np.flatnonzero((t.year == year) & (t.month == month))
            if len(i) == 0:
                continue
            for name, g in grids.items():
                a = np.where(inside, g[i[0]], np.nan).astype("float32")
                path = OUT / f"{name}_{year}_{month:02d}.tif"
                write_cog(path, np.nan_to_num(a, nan=-9999).astype("float32"), clim_transform, "EPSG:4326", -9999)
                qml(path.with_suffix(".qml"), SPEI_STYLE, -9999)
    print("SPEI-3 / SPEI-6 maps written")

    # ---- check: map VHI at the sample points must equal the point table (2021-2022 use the same reference)
    pts = pd.read_csv(DATA / "points" / "points.csv")[["pid", "x", "y"]]
    feat = pd.read_csv(DATA / "features" / "features.csv.gz", usecols=["pid", "year", "month", "VHI_clip"])
    for year in [y for y in YEARS if y in (2021, 2022)]:
        rows = []
        for month in SEASON_MONTHS:
            with rasterio.open(OUT / f"vhi_{year}_{month:02d}.tif") as ds:
                v = np.array([x[0] for x in ds.sample(zip(pts.x, pts.y))]).astype(float)
            d = pts.assign(map=np.where(v == NODATA, np.nan, v), month=month)
            rows.append(d)
        m = pd.concat(rows).merge(feat[feat.year == year], on=["pid", "month"])
        both = m.dropna(subset=["map", "VHI_clip"])
        diff = (both["map"] - both.VHI_clip).abs()
        print(f"check {year}: {len(both)} point-months valid in both; |map - table| median {diff.median():.2f}, "
              f"99th pct {diff.quantile(0.99):.2f}, max {diff.max():.1f}; valid in table but not map: "
              f"{(m['map'].isna() & m.VHI_clip.notna()).sum()}, in map but not table: {(m['map'].notna() & m.VHI_clip.isna()).sum()}")
