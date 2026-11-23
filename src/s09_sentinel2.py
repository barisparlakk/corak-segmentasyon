"""Day 2, step 3: monthly Sentinel-2 values at the same fixed 30 m pixels (Model B, 2019-2026).

  - scenes: COPERNICUS/S2_SR_HARMONIZED (L2A, offset-harmonized across the 2022 baseline change)
  - mask:   Cloud Score+ cs_cdf >= 0.6 (clear), and scene-classification snow (SCL 11)
  - per scene, reflectance is first AVERAGED over each Landsat 30 m pixel footprint
    (area-weighted), then indices are computed, so they describe the same ground as Landsat
  - indices: NDVI (B8/B4), EVI (B8/B4/B2), NDMI (B8A/B11), NDRE (B8A/B5, red edge; Landsat has
    no red-edge band, so this is what Sentinel-2 adds beyond resolution)
  - monthly MEDIAN; NOBS kept for diagnostics only

Usage:  python s09_sentinel2.py pilot | python s09_sentinel2.py
Output: data/features/s2_<year>.csv.gz
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from config import CRS_UTM, DATA, S2_START, STUDY_END, UTM_TRANSFORM, ee_init
from s08_landsat import MONTHS, NODATA, aoi, points_fc, pts_local

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()
OUT = DATA / "features"
BANDS = ["S2_NDVI", "S2_EVI", "S2_NDMI", "S2_NDRE", "S2_NOBS"]
proj30 = ee.Projection(CRS_UTM, UTM_TRANSFORM)
CS_CLEAR = 0.6
N_CHUNKS = 4

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
      .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"]))


def prep(img):
    ok = img.select("cs_cdf").gte(CS_CLEAR).And(img.select("SCL").neq(11))
    refl = (img.select(["B2", "B4", "B5", "B8", "B8A", "B11"]).divide(10000).updateMask(ok)
            .reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(proj30))
    b2, b4, b5, b8, b8a, b11 = (refl.select(b) for b in ["B2", "B4", "B5", "B8", "B8A", "B11"])
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("S2_NDVI")
    evi = (b8.subtract(b4).multiply(2.5).divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
           .clamp(-1, 1).rename("S2_EVI"))
    ndmi = b8a.subtract(b11).divide(b8a.add(b11)).rename("S2_NDMI")
    ndre = b8a.subtract(b5).divide(b8a.add(b5)).rename("S2_NDRE")
    return ee.Image.cat(ndvi, evi, ndmi, ndre)


def month_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    coll = s2.filterDate(start, start.advance(1, "month")).map(prep)
    img = coll.median().addBands(coll.select("S2_NDVI").count().rename("S2_NOBS"))
    empty = ee.Image.constant([0] * len(BANDS)).rename(BANDS).updateMask(0)
    img = ee.Image(ee.Algorithms.If(coll.size().gt(0), img, empty))
    return img.unmask(NODATA).rename([f"{b}_{month:02d}" for b in BANDS])


def year_table(year, fc):
    stack = ee.Image.cat([month_image(year, m) for m in MONTHS])
    # request the pixels in chunks: one request for all of them timed out for 2025
    n = int(pts_local.pid.max()) + 1
    edges = np.linspace(0, n, N_CHUNKS + 1).astype(int)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = fc.filter(ee.Filter.rangeContains("pid", int(lo), int(hi) - 1))
        samples = stack.sampleRegions(collection=sub, properties=["pid"], projection=proj30, tileScale=8)
        parts.append(ee.data.computeFeatures({"expression": samples, "fileFormat": "PANDAS_DATAFRAME"}))
    wide = pd.concat(parts, ignore_index=True)
    long = wide.melt(id_vars="pid", value_vars=[c for c in wide.columns if c[-3] == "_"])
    long["band"] = long.variable.str[:-3]
    long["month"] = long.variable.str[-2:].astype(int)
    out = long.pivot_table(index=["pid", "month"], columns="band", values="value").reset_index()
    out = out.replace(NODATA, np.nan)
    out.insert(1, "year", year)
    return out[["pid", "year", "month"] + BANDS]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        rng = np.random.default_rng(1)
        pids = rng.choice(pts_local.pid.values, 300, replace=False)
        fc = points_fc(pids)
        n_scenes = s2.filterDate("2019-03-01", "2019-11-01").size().getInfo()
        print(f"2019 Mar-Oct Sentinel-2 scenes over Malatya: {n_scenes}")
        t0 = time.time()
        tab = year_table(2019, fc)
        dt = time.time() - t0
        print(f"rows {len(tab)} (expected {300 * len(MONTHS)})")
        tab = tab.merge(pts_local[["pid", "lc", "elev"]], on="pid")
        print("lowland cropland medians by month:")
        print(tab[(tab.lc == 40) & (tab.elev < 1200)].groupby("month")[BANDS].median().round(2).T.to_string())
        print("valid share by month:", tab.groupby("month").S2_NDVI.apply(lambda s: s.notna().mean()).round(2).to_dict())
        print(f"time: {dt:.0f} s for 300 points x 1 year")
    else:
        fc = points_fc()
        for year in range(S2_START, STUDY_END + 1):
            f = OUT / f"s2_{year}.csv.gz"
            if f.exists():
                continue
            t0 = time.time()
            tab = year_table(year, fc)
            assert len(tab) == len(pts_local) * len(MONTHS), f"{year}: {len(tab)} rows"
            tmp = f.with_name(f.name + ".part")          # atomic: never leave a half-written year
            tab.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(f)
            print(f"{year}: {len(tab)} rows, NDVI valid {tab.S2_NDVI.notna().mean():.0%}, {time.time() - t0:.0f} s", flush=True)
