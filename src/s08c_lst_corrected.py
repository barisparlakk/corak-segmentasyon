"""Day 5, step 1: LST_TOA with a per-scene air-temperature correction (fix for the scene seams).

Problem (spec, "Monthly scene seams"): each Landsat path is imaged on a different date, and the
TOA temperature carries that day's weather. In May 2021 the western path was imaged on cool days
(ERA5-Land 2 m air temperature at overpass 4-6 C below the month's 08 UTC mean) and the eastern
path on a hot day (+3-4 C), so TCI fell from about 100 to -10 across the path edge.

Correction (fixed before any evaluation, k = 1, no fitting):
  dTa(scene, pixel) = ERA5-Land T2m at the overpass hour - mean T2m at 08 UTC over that month
  LST_corr          = LST_TOA(scene) - dTa                 then the monthly MEDIAN as before
So a scene taken on a hot day is cooled by that day's excess. The month's own mean weather
(a hot, dry month) is kept, only the date-within-month part is removed.
ERA5-Land is about 11 km; dTa is resampled bilinearly. Where ERA5-Land has no land value (large
lakes), dTa takes the scene's mean over the province, so no pixel is lost.
Everything else (masks, emissivity, months, points) is s08b unchanged.

Usage:  python s08c_lst_corrected.py pilot | python s08c_lst_corrected.py [YEAR ...]
Output: data/features/lstcorr_<year>.csv.gz  (pid, year, month, LST_TOA, NOBS_T, DTA)
        LST_TOA is the corrected value; DTA is the median dTa of the month's clear scenes.
"""
import sys
import time

import numpy as np
import pandas as pd

from config import DATA, STUDY_END, STUDY_START
from s08_landsat import MONTHS, NODATA, aoi, points_fc, pts_local
from s08b_landsat_lst import TOA, ee, prep, proj30

ee.data.setDeadline(300_000)
OUT = DATA / "features"
BANDS = ["LST_TOA", "NOBS_T", "DTA"]
ERA = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select("temperature_2m")


def month_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    mean08 = ERA.filterDate(start, end).filter(ee.Filter.calendarRange(8, 8, "hour")).mean()

    def corr(img):
        t = img.date()
        ta = ERA.filterDate(t.advance(-30, "minute"), t.advance(30, "minute")).first()
        dta = ee.Image(ta).subtract(mean08)
        fill = dta.reduceRegion(ee.Reducer.mean(), aoi, 11132, maxPixels=1e8).values().get(0)
        dta = dta.resample("bilinear").unmask(ee.Image.constant(fill)).rename("DTA")
        lst = prep(img)
        return lst.subtract(dta).rename("LST_TOA").addBands(dta.updateMask(lst.mask()))

    coll = TOA.filterDate(start, end).map(corr)
    img = coll.median().addBands(coll.select("LST_TOA").count().rename("NOBS_T")).select(BANDS)
    empty = ee.Image.constant([0, 0, 0]).rename(BANDS).updateMask(0)
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
        tab = year_table(2021, points_fc(pids))
        print(f"rows {len(tab)} (expected {300 * len(MONTHS)}), {time.time() - t0:.0f} s")
        old = pd.read_csv(OUT / "lsttoa_2021.csv.gz")
        j = tab.merge(old, on=["pid", "year", "month"], suffixes=("", "_old"))
        print(f"valid: corrected {j.LST_TOA.notna().mean():.3f}, original {j.LST_TOA_old.notna().mean():.3f}; "
              f"same NOBS {np.mean(j.NOBS_T.fillna(-1) == j.NOBS_T_old.fillna(-1)):.3f}")
        print("median dTa by month:", j.groupby("month").DTA.median().round(1).to_dict())
        d = j.LST_TOA_old - j.LST_TOA
        print(f"original - corrected: mean {d.mean():+.2f}, SD {d.std():.2f} C (vs median dTa r = "
              f"{np.corrcoef(d.dropna(), j.loc[d.notna(), 'DTA'])[0, 1]:.2f})")
    else:
        years = [int(a) for a in sys.argv[1:]] or range(STUDY_START, STUDY_END + 1)
        fc = points_fc()
        for year in years:
            f = OUT / f"lstcorr_{year}.csv.gz"
            if f.exists():
                continue
            t0 = time.time()
            tab = year_table(year, fc)
            assert len(tab) == len(pts_local) * len(MONTHS), f"{year}: {len(tab)} rows"
            tmp = f.with_name(f.name + ".part")
            tab.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(f)
            print(f"{year}: {len(tab)} rows, LST_TOA valid {tab.LST_TOA.notna().mean():.0%}, "
                  f"{time.time() - t0:.0f} s", flush=True)
