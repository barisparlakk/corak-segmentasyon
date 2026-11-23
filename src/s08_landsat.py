"""Day 2, step 2: monthly Landsat 8/9 values at the fixed pixels.

For every month March-October, 2014-2026 (March is kept so April has a previous month):
  - scenes: Landsat 8 and 9 Collection 2 Level 2 (surface reflectance + surface temperature)
  - mask:   QA_PIXEL dilated cloud, cirrus, cloud, cloud shadow, snow; QA_RADSAT saturation;
            ST_B10 fill (0) for temperature
  - scale:  per-scene REFLECTANCE_MULT/ADD and TEMPERATURE_MULT/ADD metadata
  - per scene: NDVI, EVI, SAVI, NDMI, LST (deg C), then the monthly MEDIAN (robust to leftover
    cloud edges; the maximum would favour months with more scenes, e.g. once Landsat 9 starts)
  - NOBS / NOBS_LST: number of clear scenes, kept for diagnostics only (never a model feature)
Sampling uses the exact Landsat 30 m grid, so each value is one real pixel.

Usage:  python s08_landsat.py pilot      (300 points, 2020, with checks and timing)
        python s08_landsat.py            (all points, all years; one file per year, skips done years)
Output: data/features/landsat_<year>.csv.gz  (long format: pid, month, NDVI, ..., NOBS)
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from config import AOI_GEOJSON, CRS_UTM, DATA, GEE_PROJECT, STUDY_END, STUDY_START, UTM_TRANSFORM, ee_init

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()
OUT = DATA / "features"
OUT.mkdir(parents=True, exist_ok=True)
MONTHS = list(range(3, 11))
BANDS = ["NDVI", "EVI", "SAVI", "NDMI", "LST", "NOBS", "NOBS_LST"]
NODATA = -9999
proj30 = ee.Projection(CRS_UTM, UTM_TRANSFORM)
aoi = ee.Geometry(json.loads(AOI_GEOJSON.read_text())["geometry"])
pts_local = pd.read_csv(DATA / "points" / "points.csv")
L89 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
       .filterBounds(aoi))


def points_fc(pids=None):
    asset = f"projects/{GEE_PROJECT}/assets/corak_points"
    if pids is None:
        try:
            ee.data.getAsset(asset)
            return ee.FeatureCollection(asset)
        except Exception:
            pids = pts_local.pid.values
    sub = pts_local[pts_local.pid.isin(pids)]
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]), {"pid": int(p)})
                                 for a, b, p in zip(sub.lon, sub.lat, sub.pid)])


def prep(img):
    def num(k):
        return ee.Number(img.get(k))
    bands = ["SR_B2", "SR_B4", "SR_B5", "SR_B6"]
    mult = ee.Image.constant([num(f"REFLECTANCE_MULT_BAND_{b[-1]}") for b in bands])
    add = ee.Image.constant([num(f"REFLECTANCE_ADD_BAND_{b[-1]}") for b in bands])
    sr = img.select(bands).multiply(mult).add(add).clamp(0, 1).rename(["blue", "red", "nir", "swir1"])
    qa = img.select("QA_PIXEL")
    clear = qa.bitwiseAnd(0b111110).eq(0).And(img.select("QA_RADSAT").eq(0))   # bits 1-5
    blue, red, nir, swir1 = (sr.select(b) for b in ["blue", "red", "nir", "swir1"])
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = (nir.subtract(red).multiply(2.5)
           .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
           .clamp(-1, 1).rename("EVI"))
    savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    st = img.select("ST_B10")
    lst = (st.multiply(num("TEMPERATURE_MULT_BAND_ST_B10")).add(num("TEMPERATURE_ADD_BAND_ST_B10"))
           .subtract(273.15).rename("LST").updateMask(st.gt(0)))
    return (ee.Image.cat(ndvi, evi, savi, ndmi).updateMask(clear)
            .addBands(lst.updateMask(clear)))


def month_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    coll = L89.filterDate(start, start.advance(1, "month")).map(prep)
    med = coll.median()
    nobs = coll.select("NDVI").count().rename("NOBS")
    nobs_lst = coll.select("LST").count().rename("NOBS_LST")
    img = ee.Image.cat(med.select(["NDVI", "EVI", "SAVI", "NDMI", "LST"]), nobs, nobs_lst)
    # an empty month yields an image with no bands; give it all bands, fully masked
    empty = ee.Image.constant([0] * len(BANDS)).rename(BANDS).updateMask(0)
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
    pilot = len(sys.argv) > 1 and sys.argv[1] == "pilot"
    if pilot:
        rng = np.random.default_rng(1)
        pids = rng.choice(pts_local.pid.values, 300, replace=False)
        fc = points_fc(pids)

        # check 1: our grid reproduces a single scene's own pixels exactly
        scene = L89.filterDate("2020-07-01", "2020-08-01").first()
        a = scene.select("SR_B4").sampleRegions(collection=fc, properties=["pid"],
                                                 projection=scene.select("SR_B4").projection())
        b = scene.select("SR_B4").sampleRegions(collection=fc, properties=["pid"], projection=proj30)
        da = ee.data.computeFeatures({"expression": a, "fileFormat": "PANDAS_DATAFRAME"}).set_index("pid")
        db = ee.data.computeFeatures({"expression": b, "fileFormat": "PANDAS_DATAFRAME"}).set_index("pid")
        j = da.join(db, lsuffix="_native", rsuffix="_ours", how="inner")
        print(f"check 1: {len(j)} points in one scene; identical pixel values: "
              f"{(j.SR_B4_native == j.SR_B4_ours).mean():.0%}")

        t0 = time.time()
        tab = year_table(2020, fc)
        dt = time.time() - t0
        print(f"check 2: rows {len(tab)} = 300 points x {len(MONTHS)} months = {300 * len(MONTHS)}: "
              f"{len(tab) == 300 * len(MONTHS)}")
        tab = tab.merge(pts_local[["pid", "lc", "elev"]], on="pid")
        low_crop = tab[(tab.lc == 40) & (tab.elev < 1200)]
        print("check 3: lowland cropland medians by month (NDVI, LST, NOBS):")
        print(low_crop.groupby("month")[["NDVI", "LST", "NOBS"]].median().round(2).T.to_string())
        print("valid share by month:", tab.groupby("month").NDVI.apply(lambda s: s.notna().mean()).round(2).to_dict())
        print(f"time for 300 points x 1 year: {dt:.0f} s -> all {len(pts_local)} points x "
              f"{STUDY_END - STUDY_START + 1} years ~ {dt * len(pts_local) / 300 * (STUDY_END - STUDY_START + 1) / 60:.0f} min "
              "if it scaled linearly (it usually scales better)")
    else:
        fc = points_fc()
        for year in range(STUDY_START, STUDY_END + 1):
            f = OUT / f"landsat_{year}.csv.gz"
            if f.exists():
                continue
            t0 = time.time()
            tab = year_table(year, fc)
            assert len(tab) == len(pts_local) * len(MONTHS), f"{year}: {len(tab)} rows"
            tmp = f.with_name(f.name + ".part")          # atomic: never leave a half-written year
            tab.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(f)
            print(f"{year}: {len(tab)} rows, NDVI valid {tab.NDVI.notna().mean():.0%}, {time.time() - t0:.0f} s", flush=True)
