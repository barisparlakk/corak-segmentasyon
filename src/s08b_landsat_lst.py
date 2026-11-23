"""Day 2, step 2b: land surface temperature for EVERY pixel, from Landsat Level-1 TOA.

Why: the Collection 2 Level-2 surface temperature (ST_B10) is empty wherever the ASTER GED
emissivity database has no data. Over Malatya that is a north-south strip (about 38.0-38.65 E)
holding 9% of our pixels, in every year. Mixing L2 temperature with something else there
would make TCI inconsistent, so TCI uses this single method everywhere instead.

Method (single channel with NDVI-threshold emissivity, Sobrino et al. 2004), per scene:
  BT   = TOA brightness temperature of band 10 (K), from LANDSAT/LC0x/C02/T1_TOA
  NDVI = from TOA red and NIR of the same scene
  Pv   = ((NDVI - 0.2) / (0.5 - 0.2))^2, clamped to 0-1
  e    = 0.004 Pv + 0.986
  LST  = BT / (1 + (10.895 um * BT / 14388 um K) ln e) - 273.15   (deg C)
Cloud, shadow, snow and cirrus masked with QA_PIXEL as in s08. Monthly MEDIAN.
No atmospheric correction: absolute values are biased, but TCI compares each pixel with
its own history for the same month, so a stable bias cancels. Checked against L2 ST where
both exist (pilot).

Usage:  python s08b_landsat_lst.py pilot | python s08b_landsat_lst.py
Output: data/features/lsttoa_<year>.csv.gz  (pid, year, month, LST_TOA, NOBS_T)
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from config import CRS_UTM, DATA, STUDY_END, STUDY_START, UTM_TRANSFORM, ee_init
from s08_landsat import MONTHS, NODATA, aoi, points_fc, pts_local

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()
OUT = DATA / "features"
BANDS = ["LST_TOA", "NOBS_T"]
proj30 = ee.Projection(CRS_UTM, UTM_TRANSFORM)
TOA = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA"))
       .filterBounds(aoi))


def prep(img):
    clear = img.select("QA_PIXEL").bitwiseAnd(0b111110).eq(0)
    ndvi = img.normalizedDifference(["B5", "B4"])
    pv = ndvi.subtract(0.2).divide(0.3).clamp(0, 1).pow(2)
    emis = pv.multiply(0.004).add(0.986)
    bt = img.select("B10")
    lst = bt.divide(bt.multiply(10.895 / 14388).multiply(emis.log()).add(1)).subtract(273.15)
    return lst.rename("LST_TOA").updateMask(clear)


def month_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    coll = TOA.filterDate(start, start.advance(1, "month")).map(prep)
    img = coll.median().addBands(coll.count().rename("NOBS_T"))
    empty = ee.Image.constant([0, 0]).rename(BANDS).updateMask(0)
    img = ee.Image(ee.Algorithms.If(coll.size().gt(0), img, empty))
    return img.unmask(NODATA).rename([f"{b}_{month:02d}" for b in BANDS])


def year_table(year, fc):
    stack = ee.Image.cat([month_image(year, m) for m in MONTHS])
    samples = stack.sampleRegions(collection=fc, properties=["pid"], projection=proj30, tileScale=4)
    wide = ee.data.computeFeatures({"expression": samples, "fileFormat": "PANDAS_DATAFRAME"})
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
        t0 = time.time()
        tab = year_table(2019, points_fc(pids))
        print(f"rows {len(tab)} (expected {300 * len(MONTHS)}), {time.time() - t0:.0f} s")
        l2 = pd.read_csv(OUT / "landsat_2019.csv.gz")
        j = tab.merge(l2[["pid", "month", "LST", "NDVI"]], on=["pid", "month"])
        both = j.dropna(subset=["LST_TOA", "LST"])
        print(f"valid LST_TOA where NDVI valid: {j[j.NDVI.notna()].LST_TOA.notna().mean():.1%} "
              f"(L2 LST: {j[j.NDVI.notna()].LST.notna().mean():.1%})")
        print(f"vs L2 ST where both exist (n={len(both)}): r = {np.corrcoef(both.LST_TOA, both.LST)[0, 1]:.3f}, "
              f"mean bias (TOA - L2) = {(both.LST_TOA - both.LST).mean():+.2f} C, "
              f"SD of difference = {(both.LST_TOA - both.LST).std():.2f} C")
        # what TCI cares about: the anomaly within each pixel-month across pixels -> check per month
        print("per-month correlation:", both.groupby("month").apply(
            lambda d: round(np.corrcoef(d.LST_TOA, d.LST)[0, 1], 3)).to_dict())
    else:
        fc = points_fc()
        for year in range(STUDY_START, STUDY_END + 1):
            f = OUT / f"lsttoa_{year}.csv.gz"
            if f.exists():
                continue
            t0 = time.time()
            tab = year_table(year, fc)
            assert len(tab) == len(pts_local) * len(MONTHS), f"{year}: {len(tab)} rows"
            tmp = f.with_name(f.name + ".part")          # atomic: never leave a half-written year
            tab.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(f)
            print(f"{year}: {len(tab)} rows, LST_TOA valid {tab.LST_TOA.notna().mean():.0%}, {time.time() - t0:.0f} s", flush=True)
